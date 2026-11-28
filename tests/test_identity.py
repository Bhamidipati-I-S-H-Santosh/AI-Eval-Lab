"""Identity-aware access-control testing."""
import json

import pytest
import yaml

from evalkit import runner
from evalkit.identity import (ConfluenceAclSource, GoogleDriveAclSource, SharePointAclSource, StaticAclSource,
                              TestUser, check_access, cited_sources)
from evalkit.types import AgentResult, RetrievedDoc, TestCase, ToolCall

ACL = {"salary_bands.md": {"allow": ["group:hr"]},
       "board/*": {"allow": ["user:ceo@corp.example"], "deny": ["group:contractors"]},
       "*.md": {"allow": ["everyone"]}}
ALICE = TestUser("alice", "alice@corp.example", ["hr"])
BOB = TestUser("bob", "bob@corp.example", ["engineering"])


def doc(name):
    return RetrievedDoc(id=f"{name}#0", text="", metadata={"source": name})


def test_static_acl_allow_deny_and_globs():
    acl = StaticAclSource(documents=ACL)
    assert acl.allowed(ALICE, doc("salary_bands.md")) is True
    assert acl.allowed(BOB, doc("salary_bands.md")) is False
    assert acl.allowed(BOB, doc("leave_policy.md")) is True
    ceo = TestUser("ceo", "ceo@corp.example", ["contractors"])
    assert acl.allowed(ceo, doc("board/minutes.md")) is False          # deny wins
    assert StaticAclSource(documents={"x.md": {"allow": ["everyone"]}}).allowed(BOB, doc("y.pdf")) is None


def test_static_acl_precedence_ignores_file_order():
    """Regression: YAML dumping sorted '*.md' before 'salary_bands.md' and let everyone in."""
    generic_first = {"*.md": {"allow": ["everyone"]}, "salary_bands.md": {"allow": ["group:hr"]}}
    acl = StaticAclSource(documents=generic_first)
    assert acl.allowed(BOB, doc("salary_bands.md")) is False      # most specific rule decides
    assert acl.allowed(ALICE, doc("salary_bands.md")) is True
    deny_generic = StaticAclSource(documents={"*": {"deny": ["group:hr"]}, "salary_bands.md": {"allow": ["group:hr"]}})
    assert deny_generic.allowed(ALICE, doc("salary_bands.md")) is False   # a deny anywhere wins


def test_citations_extracted():
    assert cited_sources("See [salary_bands.md] and [leave_policy.md#2], or [doc](https://x.sharepoint.com/a.docx)") == \
        ["salary_bands.md", "leave_policy.md", "https://x.sharepoint.com/a.docx"]


def test_check_access_all_evidence_types():
    acl = StaticAclSource(documents=ACL)
    case = TestCase(id="q@bob", input="What are the salary bands?", identity="bob",
                    expected_sources=["salary_bands.md"], expected_contains=["$120k"])
    r = AgentResult(answer="Band 3 is $120k [salary_bands.md].", retrieved=[doc("salary_bands.md"), doc("leave_policy.md")],
                    trace=[ToolCall("search", {"q": "salary"}, json.dumps([{"source": "salary_bands.md"}]))])
    f = {(x.doc, x.evidence): x.allowed for x in check_access(case, r, acl, {"bob": BOB})}
    assert f[("salary_bands.md", "retrieved")] is False
    assert f[("leave_policy.md", "retrieved")] is True
    assert f[("salary_bands.md", "tool_result")] is False
    assert f[("salary_bands.md", "citation")] is False
    assert f[("salary_bands.md", "content")] is False                   # facts only in a denied doc


def test_content_leak_without_any_citation():
    acl = StaticAclSource(documents=ACL)
    case = TestCase(id="q", input="salary?", identity="bob", expected_sources=["salary_bands.md"],
                    expected_contains=["$120k"])
    findings = check_access(case, AgentResult(answer="It's $120k."), acl, {"bob": BOB})
    assert [(x.evidence, x.allowed) for x in findings] == [("content", False)]


# ---------- end to end through the runner ----------
SALARY = "Salary band 3 pays $120k [salary_bands.md]."
STATE = {"as": None}


def act_as(name):
    STATE["as"] = name


def leaky_agent(question):
    """Ignores who is asking: always reads the salary document."""
    return {"answer": SALARY, "retrieved": [{"id": "salary_bands.md#0", "text": "...", "metadata": {"source": "salary_bands.md"}}]}


def careful_agent(question):
    if STATE["as"] == "alice":
        return leaky_agent(question)
    return {"answer": "I don't know: you don't have access to that information.", "retrieved": []}


def _cfg(tmp_path, target, users):
    (tmp_path / "acl.yaml").write_text(yaml.safe_dump({"documents": ACL}), encoding="utf-8")
    (tmp_path / "cases.jsonl").write_text(json.dumps(
        {"id": "salary", "input": "What does salary band 3 pay?", "expected_contains": ["$120k"],
         "expected_sources": ["salary_bands.md"]}) + "\n", encoding="utf-8")
    cfg = {"name": "acl", "output_dir": str(tmp_path / "out"), "dataset": {"path": str(tmp_path / "cases.jsonl")},
           "system": {"type": "python", "target": target, "identity_hook": "tests.test_identity:act_as"},
           "identity": {"acl": {"type": "static", "path": str(tmp_path / "acl.yaml")}, "users": users},
           "run": {"concurrency": 4}, "gates": {"access_violation": "<=0"}}
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


USERS = {"alice": {"email": "alice@corp.example", "groups": ["hr"]},
         "bob": {"email": "bob@corp.example", "groups": ["engineering"]}}


def test_leaky_agent_is_caught_for_the_right_identity(tmp_path):
    report, run_dir = runner.run(_cfg(tmp_path, "tests.test_identity:leaky_agent", USERS), log=lambda *a: None)
    assert report["access"]["per_identity"]["alice"]["violations"] == 0
    assert report["access"]["per_identity"]["bob"]["violations"] == 1
    assert any("access_violation" in f for f in report["gate_failures"])
    acc = json.loads((run_dir / "access_report.json").read_text(encoding="utf-8"))
    assert {v["evidence"] for v in acc["violations"]} >= {"retrieved", "citation", "content"}
    assert all(v["identity"] == "bob" for v in acc["violations"])
    assert set(acc["by_question"]["salary"]) == {"alice", "bob"}       # same question, both identities
    assert "## Access control" in (run_dir / "report.md").read_text(encoding="utf-8")


def test_careful_agent_passes(tmp_path):
    report, _ = runner.run(_cfg(tmp_path, "tests.test_identity:careful_agent", USERS), log=lambda *a: None)
    assert report["access"]["violations"] == 0
    assert not [f for f in report["gate_failures"] if "access" in f]


def test_per_user_system_overrides(tmp_path):
    users = {"alice": {**USERS["alice"], "system": {"target": "tests.test_identity:leaky_agent"}},
             "bob": {**USERS["bob"], "system": {"target": "tests.test_identity:careful_agent"}}}
    cfg = _cfg(tmp_path, "tests.test_identity:leaky_agent", users)
    report, _ = runner.run(cfg, log=lambda *a: None)
    assert report["access"]["violations"] == 0       # bob's own system (careful, STATE not alice) refused


# ---------- API-backed ACL sources (fake HTTP) ----------
class FakeSession:
    def __init__(self, routes):
        self.routes, self.auth, self.calls = routes, None, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        for frag, body in self.routes.items():
            if frag in url:
                return _Resp(200, body)
        return _Resp(404, {})


class _Resp:
    def __init__(self, code, body):
        self.status_code, self._b = code, body

    def json(self):
        return self._b


def test_sharepoint_graph_permissions():
    s = FakeSession({
        "/drives/D1/items/I1/permissions": {"value": [{"grantedToV2": {"group": {"displayName": "HR Team"}}}]},
        "/drives/D1/items/I2/permissions": {"value": [{"link": {"scope": "organization"}}]},
        "/users/bob@corp.example/transitiveMemberOf": {"value": [{"displayName": "Engineering"}]},
    })
    acl = SharePointAclSource(session=s)
    hr_doc = RetrievedDoc("x", "", metadata={"drive_id": "D1", "item_id": "I1"})
    org_doc = RetrievedDoc("y", "", metadata={"drive_id": "D1", "item_id": "I2"})
    bob = TestUser("bob", "bob@corp.example")                          # groups resolved via Graph
    assert acl.allowed(bob, hr_doc) is False
    assert acl.allowed(bob, org_doc) is True
    assert acl.allowed(TestUser("h", "h@corp.example", ["HR Team"]), hr_doc) is True
    assert acl.allowed(bob, RetrievedDoc("z", "", metadata={})) is None            # no item reference -> unknown
    url_doc = RetrievedDoc("w", "", metadata={"web_url": "https://corp.sharepoint.com/sites/hr/a.docx"})
    acl.allowed(bob, url_doc)
    assert any("/shares/u!" in u for u, _ in s.calls)                    # sharing-URL lookup path


def test_google_drive_permissions():
    s = FakeSession({"/files/F1/permissions": {"permissions": [
        {"type": "group", "emailAddress": "hr@corp.example"}, {"type": "user", "emailAddress": "ceo@corp.example"}]},
        "/files/F2/permissions": {"permissions": [{"type": "domain", "domain": "corp.example"}]}})
    acl = GoogleDriveAclSource(session=s, auth=None)
    f1 = RetrievedDoc("a", "", metadata={"url": "https://docs.google.com/document/d/F1/edit"})
    assert acl.allowed(TestUser("x", "x@corp.example", ["hr@corp.example"]), f1) is True
    assert acl.allowed(BOB, f1) is False
    assert acl.allowed(BOB, RetrievedDoc("b", "", metadata={"file_id": "F2"})) is True


def test_confluence_restrictions_include_ancestors():
    s = FakeSession({
        "/content/100/restriction": {"restrictions": {"user": {"results": []}, "group": {"results": []}}},
        "/content/50/restriction": {"restrictions": {"user": {"results": []}, "group": {"results": [{"name": "finance"}]}}},
        "/content/100": {"id": "100", "ancestors": [{"id": "50"}]},
    })
    acl = ConfluenceAclSource(base_url="https://corp.atlassian.net", session=s)
    page = RetrievedDoc("p", "", metadata={"url": "https://corp.atlassian.net/wiki/spaces/FIN/pages/100/Budget"})
    assert acl.allowed(TestUser("f", "f@corp.example", ["finance"]), page) is True
    assert acl.allowed(BOB, page) is False                               # parent page restricted to finance


def test_unknown_identity_in_case_is_a_clear_error(tmp_path):
    from evalkit.errors import ConfigError
    from evalkit.identity import IdentityRouter
    from evalkit.adapters import build_agent
    cfg = {"system": {"type": "python", "target": "tests.test_identity:careful_agent",
                      "identity_hook": "tests.test_identity:act_as"}, "identity": {"users": USERS}}
    router = IdentityRouter(cfg, build_agent(cfg["system"]))
    with pytest.raises(ConfigError):
        router.for_identity("mallory")
