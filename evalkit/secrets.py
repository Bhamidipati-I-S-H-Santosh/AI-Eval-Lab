"""Secrets: keep credentials out of config files.

In any config value:
    ${VAR}                 environment variable (or .env)
    ${VAR:-default}        with a default
    ${secret:NAME}         the configured secrets provider
    ${secret:vault:NAME}   a named provider, when several are configured

config:
  secrets:                         # one provider ...
    type: azure_keyvault
    vault_url: https://my-vault.vault.azure.net
  # ... or several, by name:
  secrets:
    providers:
      kv:    {type: azure_keyvault, vault_url: https://my-vault.vault.azure.net}
      vault: {type: vault, url: https://vault.internal:8200, mount: secret}
    default: kv

Every resolved secret value is remembered (in memory only) so redact() can
mask it in logs and reports. Values are never logged.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any

from .errors import ConfigError, SecretNotFoundError
from .plugins import HealthStatus, Plugin, timed_health

log = logging.getLogger("evalkit.secrets")

_REF = re.compile(r"\$\{(secret:)?([A-Za-z0-9_.:/#\-]+?)(?::-([^}]*))?\}")
_lock = threading.Lock()
_PROVIDERS: dict[str, "SecretsProvider"] = {}
_DEFAULT = "env"
_SEEN: set[str] = set()          # resolved secret values, for redaction


class SecretsProvider(Plugin):
    """Return the secret's value, or raise SecretNotFoundError."""

    cache_ttl_s: float = 300.0

    def __init__(self):
        self._cache: dict[str, tuple[float, str]] = {}

    def fetch(self, name: str) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def get(self, name: str) -> str:
        hit = self._cache.get(name)
        if hit and time.time() - hit[0] < self.cache_ttl_s:
            return hit[1]
        value = self.fetch(name)
        self._cache[name] = (time.time(), value)
        return value


class EnvSecrets(SecretsProvider):
    """Environment variables (and .env, loaded by the runner)."""

    def __init__(self, prefix: str = "", **_):
        super().__init__()
        self.prefix = prefix

    def fetch(self, name: str) -> str:
        key = self.prefix + name
        if key not in os.environ:
            raise SecretNotFoundError(f"environment variable '{key}' is not set")
        return os.environ[key]


class AzureKeyVaultSecrets(SecretsProvider):
    """Azure Key Vault. Auth: DefaultAzureCredential (managed identity, az login,
    workload identity, env vars). Needs: pip install 'evalkit[azure]'."""

    def __init__(self, vault_url: str, client=None, **_):
        super().__init__()
        if client is None:
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.secrets import SecretClient
            client = SecretClient(vault_url=vault_url, credential=DefaultAzureCredential())
        self.client, self.vault_url = client, vault_url

    def fetch(self, name: str) -> str:
        try:
            return self.client.get_secret(name).value
        except Exception as e:
            raise SecretNotFoundError(f"Key Vault {self.vault_url}: cannot read '{name}': {type(e).__name__}") from e

    def health_check(self) -> HealthStatus:
        return timed_health(lambda: f"listed {sum(1 for _ in zip(range(1), self.client.list_properties_of_secrets()))} secret(s)")


class AwsSecretsManager(SecretsProvider):
    """AWS Secrets Manager. NAME may be 'secret-id' or 'secret-id#json_key'.
    Auth: the standard AWS credential chain (IAM role, SSO, env)."""

    def __init__(self, region: str | None = None, client=None, **_):
        super().__init__()
        if client is None:
            import boto3
            client = boto3.client("secretsmanager", region_name=region)
        self.client = client

    def fetch(self, name: str) -> str:
        sid, _, key = name.partition("#")
        try:
            value = self.client.get_secret_value(SecretId=sid)["SecretString"]
        except Exception as e:
            raise SecretNotFoundError(f"AWS Secrets Manager: cannot read '{sid}': {type(e).__name__}") from e
        if key:
            import json
            try:
                return str(json.loads(value)[key])
            except (ValueError, KeyError) as e:
                raise SecretNotFoundError(f"AWS secret '{sid}' has no JSON key '{key}'") from e
        return value


class GcpSecretManager(SecretsProvider):
    """GCP Secret Manager. Auth: Application Default Credentials / service account."""

    def __init__(self, project: str, version: str = "latest", client=None, **_):
        super().__init__()
        if client is None:
            from google.cloud import secretmanager
            client = secretmanager.SecretManagerServiceClient()
        self.client, self.project, self.version = client, project, version

    def fetch(self, name: str) -> str:
        path = f"projects/{self.project}/secrets/{name}/versions/{self.version}"
        try:
            return self.client.access_secret_version(request={"name": path}).payload.data.decode("utf-8")
        except Exception as e:
            raise SecretNotFoundError(f"GCP Secret Manager: cannot read '{name}': {type(e).__name__}") from e


class HashiCorpVault(SecretsProvider):
    """HashiCorp Vault KV v2 over its REST API (no extra package).
    NAME is 'path#key', e.g. 'evalkit/groq#api_key'. Token from VAULT_TOKEN
    unless token/token_env is set."""

    def __init__(self, url: str | None = None, token: str | None = None, token_env: str = "VAULT_TOKEN",
                 mount: str = "secret", namespace: str | None = None, verify: bool | str = True,
                 session=None, timeout: float = 10, **_):
        super().__init__()
        import requests
        self.url = (url or os.environ.get("VAULT_ADDR", "")).rstrip("/")
        if not self.url:
            raise ConfigError("vault secrets provider needs 'url' or VAULT_ADDR")
        self.token = token or os.environ.get(token_env, "")
        self.mount, self.namespace, self.verify, self.timeout = mount, namespace, verify, timeout
        self.session = session or requests.Session()

    def _headers(self):
        h = {"X-Vault-Token": self.token}
        if self.namespace:
            h["X-Vault-Namespace"] = self.namespace
        return h

    def fetch(self, name: str) -> str:
        path, _, key = name.partition("#")
        r = self.session.get(f"{self.url}/v1/{self.mount}/data/{path}", headers=self._headers(),
                             timeout=self.timeout, verify=self.verify)
        if r.status_code != 200:
            raise SecretNotFoundError(f"Vault: cannot read '{path}' (HTTP {r.status_code})")
        data = (r.json().get("data") or {}).get("data") or {}
        if key:
            if key not in data:
                raise SecretNotFoundError(f"Vault secret '{path}' has no key '{key}'")
            return str(data[key])
        if len(data) == 1:
            return str(next(iter(data.values())))
        raise SecretNotFoundError(f"Vault secret '{path}' has several keys; use '{path}#<key>'")

    def health_check(self) -> HealthStatus:
        def probe():
            r = self.session.get(f"{self.url}/v1/sys/health", timeout=self.timeout, verify=self.verify)
            return f"HTTP {r.status_code}"
        return timed_health(probe)


# ---------- configuration & resolution ----------
def configure(block: dict | None) -> None:
    """Install the providers from a config's 'secrets' block (None = env only)."""
    global _DEFAULT
    from .plugins import REGISTRY
    with _lock:
        _PROVIDERS.clear()
        _PROVIDERS["env"] = EnvSecrets()
        _DEFAULT = "env"
        if not block:
            return
        if "providers" in block:
            for name, cfg in block["providers"].items():
                _PROVIDERS[name] = _build(REGISTRY, cfg)
            _DEFAULT = block.get("default") or next(iter(block["providers"]))
        else:
            _PROVIDERS["default"] = _build(REGISTRY, block)
            _DEFAULT = "default"


def _build(registry, cfg: dict) -> SecretsProvider:
    cfg = dict(cfg)
    name = cfg.pop("type")
    cls = registry.get("secrets", name)
    # provider configs may only use env vars (a secret can't depend on itself)
    return cls(**{k: _expand_env_only(v) for k, v in cfg.items()})


def _expand_env_only(v):
    if isinstance(v, str):
        return re.sub(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}", lambda m: os.environ.get(m.group(1), m.group(2) or ""), v)
    return v


def get_secret(name: str, provider: str | None = None) -> str:
    if not _PROVIDERS:
        configure(None)
    pname = provider or _DEFAULT
    if pname not in _PROVIDERS:
        raise SecretNotFoundError(f"no secrets provider named '{pname}'; configured: {sorted(_PROVIDERS)}")
    value = _PROVIDERS[pname].get(name)
    if value and len(value) >= 6:
        with _lock:
            _SEEN.add(value)
    return value


def _sub(m: re.Match) -> str:
    is_secret, ref, default = m.group(1), m.group(2), m.group(3)
    if is_secret:
        provider, _, name = ref.partition(":") if ref.count(":") else (None, "", ref)
        if not name:
            provider, name = None, ref
        try:
            return get_secret(name, provider)
        except SecretNotFoundError:
            if default is not None:
                return default
            raise
    if not re.fullmatch(r"[A-Z0-9_]+", ref):
        return m.group(0)       # not an env reference (e.g. literal text); leave untouched
    value = os.environ.get(ref)
    if value is None:
        return default or ""
    if any(w in ref for w in ("KEY", "TOKEN", "SECRET", "PASSWORD")) and len(value) >= 6:
        with _lock:
            _SEEN.add(value)
    return value


def resolve_value(value: str) -> str:
    return _REF.sub(_sub, value)


def resolve_config(obj: Any) -> Any:
    """Recursively resolve ${VAR} and ${secret:...} references."""
    if isinstance(obj, str):
        return resolve_value(obj)
    if isinstance(obj, dict):
        return {k: resolve_config(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_config(v) for v in obj]
    return obj


def redact(text: str) -> str:
    """Mask every secret value this process has resolved."""
    if not text or not _SEEN:
        return text
    for v in sorted(_SEEN, key=len, reverse=True):
        text = text.replace(v, "***")
    return text
