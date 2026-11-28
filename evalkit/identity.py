"""Identity-aware evaluation: does the agent respect document permissions?

The same questions are asked as different test users. For every answer,
evalkit checks the evidence the system exposes, strongest first:

  1. retrieved documents (ids / source metadata) and tool results
  2. sources cited in the answer ([file.md], links)
  3. content: facts that exist only in documents the user may not read
     (a case's expected_contains when all its expected_sources are denied)

against an ACL source that knows who may read which document. Each finding
is a violation with the identity, the document and the evidence type.

config:
  identity:
    acl: {type: static, path: acl.yaml}          # or sharepoint | google_drive | confluence
    users:
      alice:                                     # one entry per test identity
        email: alice@corp.example
        groups: [hr]
        system: {headers: {Authorization: "Bearer ${secret:alice-token}"}}   # deep-merged into system config
      bob: {email: bob@corp.example, groups: [engineering]}
    matrix: true          # ask every non-attack case as every user (default true)
    unknown: report       # report | violation  (a document whose ACL can't be determined)

acl.yaml (static):
  documents:
    "salary_bands.md": {allow: ["group:hr"]}
    "leave_policy.md": {allow: ["everyone"]}
    "board/*":         {allow: ["user:ceo@corp.example"], deny: ["group:contractors"]}
"""
from __future__ import annotations

import base64
import copy
import fnmatch
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .errors import ConfigError
from .plugins import HealthStatus, Plugin, timed_health
from .types import RetrievedDoc, TestCase


@dataclass
class TestUser:
    name: str
    email: str | None = None
    groups: list[str] = field(default_factory=list)
    system: dict = field(default_factory=dict)        # config overrides for calling the system as this user
    attributes: dict = field(default_factory=dict)

    __test__ = False

    def principals(self) -> set[str]:
        p = {"everyone", f"user:{self.name}".lower()}
        if self.email:
            p.add(f"user:{self.email}".lower())
        p |= {f"group:{g}".lower() for g in self.groups}
        return p


class AclSource(Plugin):
    """Answers: may this user read this document? True / False / None (unknown)."""

    def allowed(self, user: TestUser, doc: RetrievedDoc) -> bool | None:  # pragma: no cover
        raise NotImplementedError

    def close(self):
        pass


def doc_keys(doc: RetrievedDoc) -> list[str]:
    """Identifiers to look a document up by: source metadata, id, id without '#chunk'."""
    keys = []
    for k in (doc.metadata.get("source"), doc.metadata.get("url"), doc.metadata.get("path"), doc.id,
              doc.id.split("#", 1)[0] if doc.id else None):
        if k and str(k) not in keys:
            keys.append(str(k))
    return keys


class StaticAclSource(AclSource):
    """ACLs from a YAML/JSON file or inline dict. Patterns are globs; deny wins.
    Principals: everyone, user:<name or email>, group:<name>."""

    def __init__(self, path: str | None = None, documents: dict | None = None, default: str = "unknown", **_):
        if path:
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
            documents = {**(data.get("documents") or {}), **(documents or {})}
        if not documents:
            raise ConfigError("static ACL source needs 'path' or 'documents'")
        if default not in ("allow", "deny", "unknown"):
            raise ConfigError("static ACL 'default' must be allow, deny or unknown")
        self.rules = [(pat, {p.lower() for p in r.get("allow", [])}, {p.lower() for p in r.get("deny", [])})
                      for pat, r in documents.items()]
        self.default = default

    @staticmethod
    def _specificity(pat: str) -> tuple:
        """Exact names beat globs; among globs, more literal characters win."""
        wild = sum(pat.count(c) for c in "*?[")
        return (wild == 0, len(pat) - wild)

    def allowed(self, user, doc):
        """A deny in ANY matching rule wins; otherwise the most specific
        matching rule decides (file order never matters)."""
        principals = user.principals()
        matches = [(pat, allow, deny) for pat, allow, deny in self.rules
                   if any(fnmatch.fnmatch(k, pat) or fnmatch.fnmatch(Path(k).name, pat) for k in doc_keys(doc))]
        if not matches:
            return {"allow": True, "deny": False}.get(self.default)
        if any(principals & deny for _, _, deny in matches):
            return False
        _, allow, _ = max(matches, key=lambda m: self._specificity(m[0]))
        return bool(principals & allow)

    def health_check(self) -> HealthStatus:
        return HealthStatus(True, f"{len(self.rules)} ACL rule(s)")


class _RestAcl(AclSource):
    """Shared plumbing for API-backed ACL sources: session, auth, caching."""

    def __init__(self, auth: dict | None = None, session=None, timeout: float = 20, doc_map: dict | None = None):
        import requests

        from .auth import build_auth
        self.session = session or requests.Session()
        if auth:
            self.session.auth = build_auth(auth)
        self.timeout = timeout
        self.doc_map = doc_map or {}      # evalkit doc key -> provider item reference
        self._cache: dict = {}
        self._lock = threading.Lock()

    def _get(self, url: str, **params) -> dict:
        r = self.session.get(url, params=params or None, timeout=self.timeout)
        if r.status_code == 404:
            return {}
        if r.status_code >= 400:
            raise RuntimeError(f"ACL lookup failed: HTTP {r.status_code} for {url}")
        return r.json()

    def _cached(self, key, fn):
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        val = fn()
        with self._lock:
            self._cache[key] = val
        return val

    def _ref(self, doc: RetrievedDoc, meta_keys: tuple[str, ...]) -> str | None:
        for k in meta_keys:
            if doc.metadata.get(k):
                return str(doc.metadata[k])
        for key in doc_keys(doc):
            if key in self.doc_map:
                return str(self.doc_map[key])
        return None


class SharePointAclSource(_RestAcl):
    """SharePoint / OneDrive permissions via Microsoft Graph.
    Item reference from doc metadata (drive_id + item_id, or web_url / url),
    or doc_map. Group membership of each test user is read from Graph
    (transitiveMemberOf) unless the user lists groups explicitly.
    Auth: an evalkit auth block for Graph, e.g. oauth2_client_credentials with
    scope https://graph.microsoft.com/.default (app needs Sites.Read.All and
    GroupMember.Read.All), or entra_id."""

    GRAPH = "https://graph.microsoft.com/v1.0"

    def __init__(self, auth: dict | None = None, graph_url: str | None = None, session=None, doc_map=None, **_):
        super().__init__(auth, session, doc_map=doc_map)
        self.graph = (graph_url or self.GRAPH).rstrip("/")

    def _item_permissions(self, doc: RetrievedDoc) -> list[dict] | None:
        drive, item = doc.metadata.get("drive_id"), doc.metadata.get("item_id")
        if drive and item:
            url = f"{self.graph}/drives/{drive}/items/{item}/permissions"
        else:
            ref = self._ref(doc, ("web_url", "url", "sharepoint_url"))
            if not ref:
                return None
            if ref.startswith("http"):
                share = "u!" + base64.urlsafe_b64encode(ref.encode()).decode().rstrip("=")
                url = f"{self.graph}/shares/{share}/driveItem/permissions"
            else:
                url = f"{self.graph}/drives/{ref.replace(':', '/items/', 1)}/permissions"
        return self._cached(("perm", url), lambda: self._get(url).get("value", []))

    def _user_principals(self, user: TestUser) -> set[str]:
        p = user.principals()
        if user.email and not user.groups:
            data = self._cached(("groups", user.email), lambda: self._get(
                f"{self.graph}/users/{user.email}/transitiveMemberOf", **{"$select": "id,displayName,mail"}))
            for g in data.get("value", []):
                for k in ("id", "displayName", "mail"):
                    if g.get(k):
                        p.add(f"group:{g[k]}".lower())
        return p

    def allowed(self, user, doc):
        perms = self._item_permissions(doc)
        if perms is None:
            return None
        principals = self._user_principals(user)
        for perm in perms:
            if perm.get("link", {}).get("scope") in ("anonymous", "organization"):
                return True
            ids = [perm.get("grantedToV2") or perm.get("grantedTo") or {}] + \
                list(perm.get("grantedToIdentitiesV2") or perm.get("grantedToIdentities") or [])
            for ident in ids:
                u = ident.get("user") or {}
                g = ident.get("group") or ident.get("siteGroup") or {}
                for v in (u.get("email"), u.get("id"), u.get("displayName")):
                    if v and f"user:{v}".lower() in principals:
                        return True
                for v in (g.get("email"), g.get("id"), g.get("displayName")):
                    if v and f"group:{v}".lower() in principals:
                        return True
        return False

    def health_check(self) -> HealthStatus:
        return timed_health(lambda: f"Graph reachable ({len(self._get(f'{self.graph}/sites', search='*').get('value', []))} site(s) visible)")


class GoogleDriveAclSource(_RestAcl):
    """Google Drive permissions API (v3). File id from doc metadata (file_id /
    drive_file_id), a drive URL, or doc_map. Group membership comes from each
    test user's 'groups' (emails) in the config.
    Auth: gcp_service_account with scope https://www.googleapis.com/auth/drive.metadata.readonly
    (domain-wide delegation if files aren't shared with the service account)."""

    API = "https://www.googleapis.com/drive/v3"

    def __init__(self, auth: dict | None = None, session=None, doc_map=None, **_):
        super().__init__(auth or {"type": "gcp_service_account",
                                  "scopes": ["https://www.googleapis.com/auth/drive.metadata.readonly"]},
                         session, doc_map=doc_map)

    def _file_id(self, doc):
        ref = self._ref(doc, ("file_id", "drive_file_id", "url", "web_url"))
        if ref and "/" in ref:
            m = re.search(r"/d/([A-Za-z0-9_-]+)|[?&]id=([A-Za-z0-9_-]+)", ref)
            ref = (m.group(1) or m.group(2)) if m else None
        return ref

    def allowed(self, user, doc):
        fid = self._file_id(doc)
        if not fid:
            return None
        data = self._cached(("perm", fid), lambda: self._get(
            f"{self.API}/files/{fid}/permissions", fields="permissions(type,role,emailAddress,domain)",
            supportsAllDrives="true"))
        if not data:
            return None
        principals = user.principals()
        domain = (user.email or "").split("@")[-1].lower()
        for p in data.get("permissions", []):
            t, email = p.get("type"), (p.get("emailAddress") or "").lower()
            if t == "anyone":
                return True
            if t == "domain" and domain and (p.get("domain") or "").lower() == domain:
                return True
            if t == "user" and email and f"user:{email}" in principals:
                return True
            if t == "group" and email and f"group:{email}" in principals:
                return True
        return False


class ConfluenceAclSource(_RestAcl):
    """Confluence Cloud page restrictions (REST v1 restriction API). A page is
    readable if neither it nor any ancestor has read restrictions, or the user
    / one of their groups is listed on every restricted level. Space-level
    permissions are not checked (set 'unrestricted: unknown' to report those
    pages as unknown instead of allowed).
    Page id from doc metadata (page_id / confluence_id), a /pages/<id>/ URL,
    or doc_map. Users are matched by Atlassian accountId (user attribute
    'account_id') or email; groups by name."""

    def __init__(self, base_url: str, auth: dict | None = None, session=None, doc_map=None,
                 unrestricted: str = "allow", **_):
        super().__init__(auth, session, doc_map=doc_map)
        self.base = base_url.rstrip("/")
        self.unrestricted = unrestricted

    def _page_id(self, doc):
        ref = self._ref(doc, ("page_id", "confluence_id", "url", "web_url"))
        if ref and "/" in ref:
            m = re.search(r"/pages/(\d+)", ref) or re.search(r"pageId=(\d+)", ref)
            ref = m.group(1) if m else None
        return ref

    def _restrictions(self, page_id: str) -> dict:
        data = self._get(f"{self.base}/wiki/rest/api/content/{page_id}/restriction/byOperation/read",
                         expand="restrictions.user,restrictions.group")
        r = data.get("restrictions") or {}
        users = {(u.get("accountId") or "").lower() for u in (r.get("user") or {}).get("results", [])} | \
                {(u.get("email") or "").lower() for u in (r.get("user") or {}).get("results", [])}
        groups = {(g.get("name") or "").lower() for g in (r.get("group") or {}).get("results", [])}
        return {"users": users - {""}, "groups": groups - {""}}

    def allowed(self, user, doc):
        pid = self._page_id(doc)
        if not pid:
            return None
        page = self._cached(("page", pid), lambda: self._get(f"{self.base}/wiki/rest/api/content/{pid}",
                                                              expand="ancestors"))
        if not page:
            return None
        chain = [pid] + [a["id"] for a in page.get("ancestors", []) if a.get("id")]
        ident = {v.lower() for v in (user.attributes.get("account_id"), user.email) if v}
        groups = {g.lower() for g in user.groups}
        restricted_any = False
        for p in chain:
            r = self._cached(("restr", p), lambda p=p: self._restrictions(p))
            if r["users"] or r["groups"]:
                restricted_any = True
                if not (ident & r["users"] or groups & r["groups"]):
                    return False
        if not restricted_any and self.unrestricted == "unknown":
            return None
        return True


# ---------- engine glue ----------
def load_users(cfg: dict) -> dict[str, TestUser]:
    users = {}
    for name, u in ((cfg or {}).get("users") or {}).items():
        u = u or {}
        users[name] = TestUser(name=name, email=u.get("email"), groups=list(u.get("groups") or []),
                               system=u.get("system") or {},
                               attributes={k: v for k, v in u.items() if k not in ("email", "groups", "system")})
    return users


def build_acl(cfg: dict | None) -> AclSource | None:
    if not cfg or not cfg.get("acl"):
        return None
    from .plugins import REGISTRY
    return REGISTRY.build("identity", cfg["acl"])


def expand_identity_cases(cfg: dict, cases: list[TestCase]) -> list[TestCase]:
    """Clone every (non-attack, identity-less) case once per test user."""
    icfg = cfg.get("identity") or {}
    users = load_users(icfg)
    if not users or not icfg.get("matrix", True):
        return cases
    out = []
    for c in cases:
        if c.identity or c.attack:
            out.append(c)
            continue
        for name in users:
            d = copy.deepcopy(c)
            d.id, d.identity = f"{c.id}@{name}", name
            d.metadata = {**d.metadata, "base_case": c.id}
            d.tags = list(dict.fromkeys(d.tags + ["identity", f"as:{name}"]))
            out.append(d)
    return out


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class IdentityRouter:
    """Holds one system adapter per test identity (built from the base system
    config deep-merged with that user's 'system' overrides), plus the default."""

    def __init__(self, cfg: dict, default_system):
        from .adapters import build_agent
        self.default = default_system
        self.by_user = {}
        self.lock = threading.Lock()    # a shared adapter switches identity: one call at a time
        for name, user in load_users(cfg.get("identity")).items():
            if user.system:
                self.by_user[name] = build_agent(_deep_merge(cfg["system"], user.system))
            elif getattr(default_system, "supports_identity", False):
                self.by_user[name] = default_system          # adapter switches identity itself
            else:
                raise ConfigError(f"identity user '{name}' has no 'system' overrides and the system type "
                                  f"cannot switch identity; add users.{name}.system (e.g. its own token)")

    def for_identity(self, name: str | None):
        """(system, lock): lock is set when the adapter is shared and must be
        held around set_identity + the call."""
        if name is None:
            return self.default, None
        if name not in self.by_user:
            raise ConfigError(f"case identity '{name}' is not in identity.users")
        s = self.by_user[name]
        return s, (self.lock if s is self.default else None)

    def close(self):
        for s in {id(x): x for x in self.by_user.values()}.values():
            if s is not self.default:
                s.close()


_CITE = re.compile(r"\[([^\[\]]+?\.(?:md|txt|pdf|docx?|pptx?|xlsx?|html?|csv|json))(?:#[^\]]*)?\]"
                   r"|\]\((https?://[^)\s]+)\)", re.I)


def cited_sources(answer: str) -> list[str]:
    out = []
    for m in _CITE.finditer(answer or ""):
        s = m.group(1) or m.group(2)
        if s and s not in out:
            out.append(s)
    return out


@dataclass
class AccessFinding:
    doc: str
    evidence: str         # retrieved | tool_result | citation | content
    allowed: bool | None

    def to_dict(self):
        return {"doc": self.doc, "evidence": self.evidence, "allowed": self.allowed}


def check_access(case: TestCase, result, acl: AclSource, users: dict[str, TestUser]) -> list[AccessFinding]:
    """Every document this answer exposed, with whether the case's identity may read it."""
    user = users.get(case.identity or "")
    if user is None:
        return []
    findings: list[AccessFinding] = []
    seen = set()

    def add(doc: RetrievedDoc, evidence: str):
        key = (doc_keys(doc)[0] if doc_keys(doc) else doc.id, evidence)
        if key in seen:
            return
        seen.add(key)
        findings.append(AccessFinding(key[0], evidence, acl.allowed(user, doc)))

    for d in result.retrieved:
        add(d, "retrieved")
    for t in result.trace:
        for name in cited_sources(t.result or "") + re.findall(r'"source"\s*:\s*"([^"]+)"', t.result or ""):
            add(RetrievedDoc(id=name, text="", metadata={"source": name}), "tool_result")
    for name in cited_sources(result.answer):
        add(RetrievedDoc(id=name, text="", metadata={"source": name}), "citation")
    # content leak: the facts this case expects live only in documents the user can't read
    if case.expected_sources and case.expected_contains:
        decisions = [acl.allowed(user, RetrievedDoc(id=s, text="", metadata={"source": s}))
                     for s in case.expected_sources]
        if decisions and all(d is False for d in decisions):
            a = (result.answer or "").lower()
            if all(s.lower() in a for s in case.expected_contains):
                findings.append(AccessFinding(",".join(case.expected_sources), "content", False))
    return findings


def access_report(scored: list[dict], cases: list[TestCase]) -> dict:
    """Security report: violations per identity and per document."""
    by_id = {c.id: c for c in cases}
    violations, unknown = [], []
    per_user: dict[str, dict] = {}
    for r in scored:
        info = r["scores"].get("_access")
        if not info:
            continue
        c = by_id[r["case_id"]]
        u = per_user.setdefault(c.identity, {"runs": 0, "violations": 0, "unknown": 0})
        u["runs"] += 1
        bad = [f for f in info if f["allowed"] is False]
        unk = [f for f in info if f["allowed"] is None]
        u["violations"] += bool(bad)
        u["unknown"] += bool(unk)
        for f in bad:
            violations.append({"case": c.metadata.get("base_case", c.id), "identity": c.identity,
                               "repeat": r["repeat"], **f})
        for f in unk:
            unknown.append({"case": c.metadata.get("base_case", c.id), "identity": c.identity, **f})
    # cross-identity comparison: same base question, different users
    compare: dict[str, dict] = {}
    for r in scored:
        c = by_id[r["case_id"]]
        if c.identity and "base_case" in c.metadata:
            compare.setdefault(c.metadata["base_case"], {})[c.identity] = {
                "violation": any(f["allowed"] is False for f in r["scores"].get("_access") or []),
                "answered": not bool(r["scores"].get("abstention")) if r["scores"].get("abstention") is not None else None}
    return {"per_identity": per_user, "violations": violations, "unknown": unknown[:200], "by_question": compare}
