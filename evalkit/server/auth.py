"""Inbound authentication and role-based access control for the API/dashboard.

server config:
  auth:
    mode: oidc                    # oidc | api_key | none (none: development only, binds localhost)
    oidc:
      issuer: https://login.microsoftonline.com/<tenant>/v2.0    # or Okta / Google / Keycloak / Auth0
      audience: api://evalkit                                    # expected 'aud' of API bearer tokens
      client_id: ${OIDC_CLIENT_ID}                               # dashboard login (authorization code + PKCE)
      client_secret: ${secret:oidc-client-secret}                # optional for public clients
      roles_claim: roles                                         # or groups
      role_map: {EvalAdmins: admin, EvalRunners: runner}         # IdP role/group -> evalkit role
      default_role: viewer                                       # signed-in users without a mapped role
    api_keys:                                                    # for CI jobs and OTLP exporters
      "${secret:evalkit-ci-key}": {name: ci, roles: [runner]}
      "${secret:otel-ingest-key}": {name: otel-collector, roles: [ingest]}

Roles: viewer (read), runner (read + start runs + datasets), ingest (send
traces only), admin (everything, incl. deleting runs and reading the audit log).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets as pysecrets
import time
from dataclasses import dataclass, field
from urllib.parse import urlencode

from ..errors import AccessDeniedError, ConfigError

PERMISSIONS = {
    "viewer": {"read"},
    "ingest": {"ingest"},
    "runner": {"read", "run", "datasets", "ingest"},
    "admin": {"read", "run", "datasets", "ingest", "delete", "audit", "admin"},
}


@dataclass
class Principal:
    name: str
    roles: set[str] = field(default_factory=set)
    via: str = "anonymous"

    def can(self, perm: str) -> bool:
        return any(perm in PERMISSIONS.get(r, set()) for r in self.roles)

    def require(self, perm: str) -> None:
        if not self.can(perm):
            raise AccessDeniedError(f"'{self.name}' (roles: {sorted(self.roles) or 'none'}) lacks '{perm}'")


class OidcVerifier:
    """Validates OIDC/OAuth2 JWT access or ID tokens: signature (JWKS), issuer,
    audience, expiry. Works with any standards-compliant IdP."""

    def __init__(self, issuer: str, audience: str | list[str] | None = None, client_id: str | None = None,
                 client_secret: str | None = None, roles_claim: str = "roles", role_map: dict | None = None,
                 default_role: str | None = None, jwks: dict | None = None, discovery: dict | None = None,
                 name_claims: tuple = ("preferred_username", "email", "upn", "sub"), leeway: int = 60, **_):
        import jwt  # PyJWT
        self._jwt = jwt
        self.issuer = issuer.rstrip("/")
        self.audience = audience or client_id
        self.client_id, self.client_secret = client_id, client_secret
        self.roles_claim, self.role_map, self.default_role = roles_claim, role_map or {}, default_role
        self.name_claims, self.leeway = name_claims, leeway
        self._discovery = discovery
        self._jwks = jwks
        self._jwks_at = time.time() if jwks else 0.0

    @property
    def discovery(self) -> dict:
        if self._discovery is None:
            import requests
            r = requests.get(f"{self.issuer}/.well-known/openid-configuration", timeout=10)
            r.raise_for_status()
            self._discovery = r.json()
        return self._discovery

    def _keys(self, refresh: bool = False) -> dict:
        if self._jwks is None or (refresh and time.time() - self._jwks_at > 60):
            import requests
            r = requests.get(self.discovery["jwks_uri"], timeout=10)
            r.raise_for_status()
            self._jwks, self._jwks_at = r.json(), time.time()
        return self._jwks

    def _key_for(self, token: str):
        kid = self._jwt.get_unverified_header(token).get("kid")
        for refresh in (False, True):            # one refresh on unknown kid (key rotation)
            for k in self._keys(refresh).get("keys", []):
                if kid is None or k.get("kid") == kid:
                    return self._jwt.PyJWK(k).key
        raise AccessDeniedError("token signed with an unknown key")

    def verify(self, token: str, audience=None) -> dict:
        try:
            return self._jwt.decode(token, self._key_for(token), algorithms=["RS256", "RS384", "RS512", "ES256", "ES384", "PS256"],
                                    audience=audience or self.audience, issuer=self.issuer, leeway=self.leeway,
                                    options={"require": ["exp", "iss"]})
        except AccessDeniedError:
            raise
        except Exception as e:
            raise AccessDeniedError(f"invalid token: {type(e).__name__}: {e}") from e

    def principal(self, claims: dict, via: str) -> Principal:
        raw = claims.get(self.roles_claim) or []
        raw = [raw] if isinstance(raw, str) else list(raw)
        roles = {self.role_map.get(r, r) for r in raw} & set(PERMISSIONS)
        if not roles and self.default_role:
            roles = {self.default_role}
        name = next((str(claims[c]) for c in self.name_claims if claims.get(c)), "unknown")
        return Principal(name=name, roles=roles, via=via)

    # ----- browser login: authorization code + PKCE -----
    def login_url(self, redirect_uri: str, session: dict) -> str:
        verifier = pysecrets.token_urlsafe(48)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
        state, nonce = pysecrets.token_urlsafe(24), pysecrets.token_urlsafe(24)
        session.update({"pkce": verifier, "state": state, "nonce": nonce})
        q = {"response_type": "code", "client_id": self.client_id, "redirect_uri": redirect_uri,
             "scope": "openid profile email", "state": state, "nonce": nonce,
             "code_challenge": challenge, "code_challenge_method": "S256"}
        return f"{self.discovery['authorization_endpoint']}?{urlencode(q)}"

    def complete_login(self, code: str, state: str, redirect_uri: str, session: dict, http=None) -> Principal:
        if not state or not hmac.compare_digest(state, session.pop("state", "") or ""):
            raise AccessDeniedError("login state mismatch (possible CSRF); start the login again")
        import requests
        http = http or requests
        data = {"grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri,
                "client_id": self.client_id, "code_verifier": session.pop("pkce", "")}
        if self.client_secret:
            data["client_secret"] = self.client_secret
        r = http.post(self.discovery["token_endpoint"], data=data, timeout=15)
        if r.status_code != 200:
            raise AccessDeniedError(f"token exchange failed: HTTP {r.status_code}")
        id_token = r.json().get("id_token")
        if not id_token:
            raise AccessDeniedError("IdP returned no id_token")
        claims = self.verify(id_token, audience=self.client_id)
        if claims.get("nonce") != session.pop("nonce", None):
            raise AccessDeniedError("login nonce mismatch")
        return self.principal(claims, via="oidc-session")


class Authenticator:
    def __init__(self, cfg: dict | None):
        from ..secrets import resolve_config
        cfg = resolve_config(cfg or {"mode": "none"})
        self.mode = cfg.get("mode", "none")
        if self.mode not in ("none", "api_key", "oidc"):
            raise ConfigError("auth.mode must be oidc, api_key or none")
        self.oidc = OidcVerifier(**cfg["oidc"]) if self.mode == "oidc" else None
        if self.mode == "oidc" and not cfg.get("oidc", {}).get("issuer"):
            raise ConfigError("auth.oidc.issuer is required")
        self.api_keys = {}
        for key, v in (cfg.get("api_keys") or {}).items():
            if not key or len(key) < 16:
                raise ConfigError("API keys must be at least 16 characters (use a secret reference)")
            v = v or {}
            self.api_keys[hashlib.sha256(key.encode()).hexdigest()] = Principal(
                v.get("name", "api-key"), set(v.get("roles", ["viewer"])), "api_key")

    def authenticate(self, headers: dict, session: dict | None) -> Principal:
        authz = headers.get("authorization", "")
        key = headers.get("x-api-key") or (authz[7:] if authz.lower().startswith("bearer ") and self.mode != "oidc" else "")
        if key:
            p = self.api_keys.get(hashlib.sha256(key.encode()).hexdigest())
            if p:
                return p
            raise AccessDeniedError("unknown API key")
        if self.mode == "none":
            return Principal("local-dev", {"admin"}, "none")
        if self.mode == "oidc" and authz.lower().startswith("bearer "):
            token = authz[7:]
            if hashlib.sha256(token.encode()).hexdigest() in self.api_keys:
                return self.api_keys[hashlib.sha256(token.encode()).hexdigest()]
            return self.oidc.principal(self.oidc.verify(token), via="oidc-bearer")
        if session and session.get("user") and session.get("exp", 0) > time.time():
            return Principal(session["user"], set(session.get("roles", [])), "oidc-session")
        raise AccessDeniedError("authentication required")
