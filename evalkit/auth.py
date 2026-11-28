"""Outbound authentication: how evalkit proves who it is to the system under
test (HTTP agents, REST retrievers, cloud APIs).

Each provider is a `requests` auth object (usable as `session.auth = provider`)
and also exposes `token()` / `headers()` for SDKs that take a token.

config (inside a system/retriever block):
  auth:
    type: oauth2_client_credentials
    token_url: https://login.microsoftonline.com/<tenant>/oauth2/v2.0/token
    client_id: ${CLIENT_ID}
    client_secret: ${secret:agent-client-secret}
    scope: api://my-agent/.default
"""
from __future__ import annotations

import threading
import time

from .errors import ConfigError
from .plugins import HealthStatus, Plugin, timed_health


class AuthProvider(Plugin):
    def headers(self) -> dict[str, str]:
        return {}

    def token(self) -> str:
        raise NotImplementedError(f"{type(self).__name__} does not issue tokens")

    def __call__(self, request):            # requests.auth.AuthBase protocol
        request.headers.update(self.headers())
        return request

    def health_check(self) -> HealthStatus:
        return timed_health(lambda: f"{len(self.headers())} header(s) produced")


class ApiKeyAuth(AuthProvider):
    def __init__(self, key: str, header: str = "Authorization", prefix: str = "", **_):
        if not key:
            raise ConfigError("api_key auth: 'key' is empty (is the env var / secret set?)")
        self.key, self.header, self.prefix = key, header, prefix

    def headers(self):
        return {self.header: f"{self.prefix}{self.key}"}

    def token(self):
        return self.key


class BearerAuth(AuthProvider):
    def __init__(self, token: str, **_):
        if not token:
            raise ConfigError("bearer auth: 'token' is empty")
        self._token = token

    def headers(self):
        return {"Authorization": f"Bearer {self._token}"}

    def token(self):
        return self._token


class _CachedToken(AuthProvider):
    """Caches a token until shortly before it expires. Thread-safe."""
    refresh_margin_s = 120

    def __init__(self):
        self._lock = threading.Lock()
        self._tok: str | None = None
        self._exp = 0.0

    def _fetch(self) -> tuple[str, float]:  # (token, expires_at epoch)
        raise NotImplementedError

    def token(self) -> str:
        with self._lock:
            if not self._tok or time.time() > self._exp - self.refresh_margin_s:
                self._tok, self._exp = self._fetch()
            return self._tok

    def headers(self):
        return {"Authorization": f"Bearer {self.token()}"}

    def health_check(self) -> HealthStatus:
        return timed_health(lambda: f"token acquired, expires in {int(self._exp - time.time()) if self.token() else 0}s")


class OAuth2ClientCredentials(_CachedToken):
    """OAuth 2.0 client-credentials grant against any standards-compliant
    token endpoint: Microsoft Entra ID, Okta, Auth0, Keycloak, Ping, Google."""

    def __init__(self, token_url: str, client_id: str, client_secret: str, scope: str | None = None,
                 audience: str | None = None, extra_params: dict | None = None, timeout: float = 15,
                 session=None, **_):
        super().__init__()
        import requests
        if not (token_url and client_id and client_secret):
            raise ConfigError("oauth2_client_credentials needs token_url, client_id and client_secret")
        self.token_url, self.client_id, self.client_secret = token_url, client_id, client_secret
        self.scope, self.audience, self.extra, self.timeout = scope, audience, extra_params or {}, timeout
        self.session = session or requests.Session()

    def _fetch(self):
        data = {"grant_type": "client_credentials", "client_id": self.client_id,
                "client_secret": self.client_secret, **self.extra}
        if self.scope:
            data["scope"] = self.scope
        if self.audience:
            data["audience"] = self.audience
        r = self.session.post(self.token_url, data=data, timeout=self.timeout)
        if r.status_code != 200:
            raise ConfigError(f"OAuth token request to {self.token_url} failed: HTTP {r.status_code} "
                              f"{r.text[:200]}")
        body = r.json()
        return body["access_token"], time.time() + float(body.get("expires_in", 3600))


class EntraIdAuth(_CachedToken):
    """Microsoft Entra ID via azure-identity's DefaultAzureCredential (managed
    identity, workload identity, az login, env service principal)."""

    def __init__(self, scope: str, credential=None, **_):
        super().__init__()
        if credential is None:
            from azure.identity import DefaultAzureCredential
            credential = DefaultAzureCredential()
        self.scope, self.credential = scope, credential

    def _fetch(self):
        t = self.credential.get_token(self.scope)
        return t.token, float(t.expires_on)


class GcpServiceAccountAuth(_CachedToken):
    """Google credentials (service account key, workload identity, ADC).
    Set target_audience to get an ID token (Cloud Run / IAP); otherwise an
    OAuth access token for `scopes` is used."""

    def __init__(self, scopes: list[str] | None = None, target_audience: str | None = None,
                 credentials=None, **_):
        super().__init__()
        self.scopes = scopes or ["https://www.googleapis.com/auth/cloud-platform"]
        self.target_audience = target_audience
        self._creds = credentials

    def _fetch(self):
        import google.auth
        import google.auth.transport.requests as gtr
        req = gtr.Request()
        if self.target_audience:
            from google.oauth2 import id_token
            tok = id_token.fetch_id_token(req, self.target_audience)
            return tok, time.time() + 3000
        if self._creds is None:
            self._creds, _ = google.auth.default(scopes=self.scopes)
        self._creds.refresh(req)
        exp = self._creds.expiry.timestamp() if getattr(self._creds, "expiry", None) else time.time() + 3000
        return self._creds.token, exp


class AwsSigV4Auth(AuthProvider):
    """Signs each request with AWS SigV4 (IAM), e.g. for API Gateway, Lambda
    function URLs or OpenSearch. Credentials come from the standard AWS chain."""

    def __init__(self, service: str, region: str, session=None, **_):
        if session is None:
            import boto3
            session = boto3.Session()
        self.service, self.region, self.session = service, region, session

    def __call__(self, request):
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest
        creds = self.session.get_credentials()
        if creds is None:
            raise ConfigError("aws_sigv4: no AWS credentials found (IAM role, SSO, or env vars)")
        aws_req = AWSRequest(method=request.method, url=request.url, data=request.body,
                             headers={k: v for k, v in request.headers.items() if k.lower() != "connection"})
        SigV4Auth(creds.get_frozen_credentials(), self.service, self.region).add_auth(aws_req)
        request.headers.update(dict(aws_req.headers.items()))
        return request

    def health_check(self) -> HealthStatus:
        return timed_health(lambda: "credentials found" if self.session.get_credentials() else
                            (_ for _ in ()).throw(RuntimeError("no AWS credentials")))


def build_auth(cfg: dict | None) -> AuthProvider | None:
    if not cfg:
        return None
    from .plugins import REGISTRY
    return REGISTRY.build("auth", cfg)
