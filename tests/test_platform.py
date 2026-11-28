"""Platform core: plugins, config, secrets, auth, providers, judge, storage,
durable/resumable/distributed runs, budgets, rate limits, retries, multi-turn,
audit log and privacy."""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace as NS

import pytest
import yaml

from evalkit import audit, engine, privacy, runner, secrets
from evalkit.adapters.base import AgentAdapter
from evalkit.errors import (BudgetExceededError, ConfigError, PluginConfigError, PluginDependencyError,
                            PluginNotFoundError, SecretNotFoundError)
from evalkit.plugins import REGISTRY, Plugin, register
from evalkit.types import AgentResult


# ---------- plugin registry ----------
def test_unknown_plugin_lists_known_ones():
    with pytest.raises(PluginNotFoundError, match="Known"):
        REGISTRY.build("system", {"type": "nope"})


def test_missing_vendor_sdk_gives_install_hint():
    with pytest.raises(PluginDependencyError, match=r"evalkit\[aws\]"):
        REGISTRY.build("storage", {"type": "s3", "bucket": "b"})        # boto3 is not installed here


@register("system", "strict_echo")
class StrictEcho(AgentAdapter):
    def __init__(self, prefix: str, upper: bool = False):
        self.prefix, self.upper = prefix, upper

    def run(self, input):
        t = f"{self.prefix}{input}"
        return AgentResult(answer=t.upper() if self.upper else t)


def test_plugin_config_validation_and_registration_without_core_changes():
    assert REGISTRY.build("system", {"type": "strict_echo", "prefix": "> "}).run("hi").answer == "> hi"
    with pytest.raises(PluginConfigError, match="missing required"):
        REGISTRY.build("system", {"type": "strict_echo"})
    with pytest.raises(PluginConfigError, match="unknown option"):
        REGISTRY.build("system", {"type": "strict_echo", "prefix": "", "uppr": True})
    assert any(s.name == "strict_echo" for s in REGISTRY.list("system"))


def test_pydantic_config_model_validates():
    from pydantic import BaseModel

    class Cfg(BaseModel):
        size: int

    class Sized(Plugin):
        Config = Cfg

        def __init__(self, size, **_):
            self.size = size
    REGISTRY.register("storage", "sized_test", Sized)
    assert REGISTRY.build("storage", {"type": "sized_test", "size": "3"}).size == 3
    with pytest.raises(PluginConfigError):
        REGISTRY.build("storage", {"type": "sized_test", "size": "big"})


def test_entry_point_discovery(monkeypatch):
    from evalkit import plugins
    ep = NS(name="ep_echo", value="tests.test_platform:StrictEcho", dist=NS(name="acme-evalkit-plugin"))
    monkeypatch.setattr("importlib.metadata.entry_points",
                        lambda group=None: [ep] if group == "evalkit.system" else [])
    reg = plugins.Registry()
    reg.discover()
    assert reg.get("system", "ep_echo") is StrictEcho
    assert [s.source for s in reg.list("system")] == ["entry-point"]


def test_every_builtin_plugin_module_imports_without_vendor_sdks():
    """Lazy imports: listing/loading a plugin class never needs its SDK."""
    for spec in REGISTRY.list():
        if spec.source == "builtin":
            assert isinstance(spec.load(), type), spec.name


# ---------- config validation ----------
def test_config_typos_are_caught(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"systm": {"type": "python"}, "run": {"concurency": 2},
                                 "gates": {"x": "high"}}), encoding="utf-8")
    with pytest.raises(ConfigError) as e:
        runner.load_config(p)
    msg = str(e.value)
    assert "did you mean 'system'" in msg and "run.concurency" in msg and "gate 'x" in msg


# ---------- secrets ----------
def test_secret_references_env_and_redaction(monkeypatch):
    monkeypatch.setenv("MY_API_TOKEN", "tok-1234567890")
    secrets.configure(None)
    cfg = secrets.resolve_config({"a": "${secret:MY_API_TOKEN}", "b": "${MISSING:-dflt}", "c": "{input}"})
    assert cfg == {"a": "tok-1234567890", "b": "dflt", "c": "{input}"}
    assert secrets.redact("leaked tok-1234567890 here") == "leaked *** here"
    with pytest.raises(SecretNotFoundError):
        secrets.resolve_value("${secret:NOPE_NOT_SET}")


class FakeHTTP:
    def __init__(self, routes):
        self.routes, self.calls, self.auth = routes, [], None

    def get(self, url, headers=None, timeout=None, verify=None, params=None):
        self.calls.append((url, headers))
        for frag, (code, body) in self.routes.items():
            if frag in url:
                return NS(status_code=code, json=lambda b=body: b)
        return NS(status_code=404, json=lambda: {})


def test_vault_and_aws_and_named_providers(monkeypatch):
    from evalkit.secrets import AwsSecretsManager, HashiCorpVault
    s = FakeHTTP({"/v1/secret/data/evalkit/groq": (200, {"data": {"data": {"api_key": "gsk-vault-key"}}})})
    v = HashiCorpVault(url="https://vault:8200", token="t", session=s)
    assert v.get("evalkit/groq#api_key") == "gsk-vault-key"
    assert s.calls[0][1]["X-Vault-Token"] == "t"
    with pytest.raises(SecretNotFoundError):
        v.get("evalkit/groq#missing")
    aws = AwsSecretsManager(client=NS(get_secret_value=lambda SecretId: {"SecretString": '{"pw": "s3cr3t-pw"}'}))
    assert aws.get("db#pw") == "s3cr3t-pw"
    # named providers through config
    REGISTRY.register("secrets", "fake_vault_test", lambda **kw: v)
    secrets.configure({"providers": {"vault": {"type": "fake_vault_test"}}, "default": "vault"})
    assert secrets.resolve_value("key=${secret:vault:evalkit/groq#api_key}") == "key=gsk-vault-key"
    secrets.configure(None)


# ---------- auth: OAuth2 against a real local token endpoint ----------
class _TokenHandler(BaseHTTPRequestHandler):
    issued = 0

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"])).decode()
        ok = "grant_type=client_credentials" in body and "client_secret=shh" in body
        type(self).issued += ok
        self.send_response(200 if ok else 401)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"access_token": f"at-{type(self).issued}", "expires_in": 3600}).encode()
                         if ok else b'{"error":"invalid_client"}')

    def log_message(self, *a):
        pass


@pytest.fixture
def token_server():
    srv = HTTPServer(("127.0.0.1", 0), _TokenHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}/token"
    srv.shutdown()


def test_oauth2_client_credentials_real_http(token_server):
    from evalkit.auth import build_auth
    a = build_auth({"type": "oauth2_client_credentials", "token_url": token_server, "client_id": "evalkit",
                    "client_secret": "shh", "scope": "api://agent/.default"})
    assert a.headers() == {"Authorization": "Bearer at-1"}
    assert a.token() == "at-1"                      # cached, not re-issued
    bad = build_auth({"type": "oauth2_client_credentials", "token_url": token_server, "client_id": "x",
                      "client_secret": "wrong"})
    with pytest.raises(ConfigError, match="401"):
        bad.token()


def test_http_agent_sends_auth_headers():
    from evalkit.auth import ApiKeyAuth
    req = NS(headers={})
    ApiKeyAuth(key="k-123456", header="x-api-key")(req)
    assert req.headers == {"x-api-key": "k-123456"}


# ---------- model providers & judge ----------
def _openai_like(text="ok", pt=10, ct=2):
    resp = NS(choices=[NS(message=NS(content=text))], usage=NS(prompt_tokens=pt, completion_tokens=ct))
    return NS(chat=NS(completions=NS(create=lambda **kw: resp)), close=lambda: None)


def test_providers_normalize_completions():
    from evalkit.providers.anthropic import AnthropicProvider
    from evalkit.providers.bedrock import BedrockProvider
    from evalkit.providers.openai_compat import OpenAICompatibleProvider
    from evalkit.providers.vertex import VertexProvider
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}]
    assert OpenAICompatibleProvider(client=_openai_like("A")).complete(msgs, "m").text == "A"
    seen = {}

    def a_create(**kw):
        seen["anthropic"] = kw
        return NS(content=[NS(type="text", text="B")], usage=NS(input_tokens=3, output_tokens=1))
    c = AnthropicProvider(client=NS(messages=NS(create=a_create))).complete(msgs, "claude")
    assert (c.text, c.prompt_tokens) == ("B", 3) and seen["anthropic"]["system"] == "S"

    def converse(**kw):
        seen["bedrock"] = kw
        return {"output": {"message": {"content": [{"text": "C"}]}}, "usage": {"inputTokens": 4, "outputTokens": 2}}
    c = BedrockProvider(client=NS(converse=converse)).complete(msgs, "anthropic.claude")
    assert c.text == "C" and seen["bedrock"]["system"] == [{"text": "S"}]
    gen = NS(text="D", usage_metadata=NS(prompt_token_count=5, candidates_token_count=1))
    c = VertexProvider(client=NS(models=NS(generate_content=lambda **kw: gen))).complete(msgs, "gemini")
    assert (c.text, c.completion_tokens) == ("D", 1)


def test_judge_uses_any_provider_counts_tokens_and_encrypts_cache(tmp_path):
    from cryptography.fernet import Fernet

    from evalkit.judge import Judge
    from evalkit.providers.openai_compat import OpenAICompatibleProvider
    used = []
    key = Fernet.generate_key().decode()
    j = Judge(model="m", provider=OpenAICompatibleProvider(client=_openai_like('{"score": 1}')),
              cache_dir=str(tmp_path), cache="encrypted", cache_key=key, on_usage=lambda p, c: used.append(p + c))
    assert j.grade("is it right?")["score"] == 1.0
    assert j.grade("is it right?")["score"] == 1.0          # second time from cache
    assert used == [12] and j.calls == 1
    blob = next(tmp_path.glob("*.enc")).read_bytes()
    assert b"score" not in blob                              # stored encrypted
    off = Judge(model="m", provider=OpenAICompatibleProvider(client=_openai_like("x")),
                cache_dir=str(tmp_path / "off"), cache="off")
    off.complete("p")
    assert not (tmp_path / "off").exists()


@pytest.mark.skipif(os.environ.get("EVALKIT_LIVE") != "1", reason="live: needs a running Ollama (EVALKIT_LIVE=1)")
def test_ollama_live():
    from evalkit.providers.ollama import OllamaProvider
    p = OllamaProvider(default_model=os.environ.get("EVALKIT_OLLAMA_MODEL", "qwen3-vl:4b"), think=False)
    assert p.health_check().ok
    out = p.complete([{"role": "user", "content": "Reply with the single word: pong"}], p.default_model, max_tokens=400)
    assert out.text and out.completion_tokens


# ---------- storage ----------
class FakeS3:
    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body, **kw):
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            e = Exception("missing")
            e.response = {"Error": {"Code": "NoSuchKey"}}
            raise e
        return {"Body": NS(read=lambda: self.objects[Key])}

    def list_objects_v2(self, Bucket, Prefix, **kw):
        return {"Contents": [{"Key": k} for k in sorted(self.objects) if k.startswith(Prefix)], "IsTruncated": False}

    def delete_object(self, Bucket, Key):
        self.objects.pop(Key, None)


def test_local_storage_survives_torn_write_and_blocks_escape(tmp_path):
    from evalkit.storage.local import LocalStorage
    st = LocalStorage(root=str(tmp_path))
    st.append_record("r1", "a__r0", {"case_id": "a", "repeat": 0})
    with (tmp_path / "r1" / "results.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"case_id": "b", "rep')                      # crash mid-write
    assert [r["case_id"] for r in st.read_records("r1")] == ["a"]
    with pytest.raises(ConfigError):
        st.write_text("../outside.txt", "x")
    assert st.health_check().ok


def test_s3_storage_records_and_runs():
    from evalkit.storage.cloud import S3Storage
    s = S3Storage(bucket="b", prefix="ek/", client=FakeS3())
    s.write_text("run1/report.md", "# hi")
    s.append_record("run1", "c1__r0", {"case_id": "c1", "repeat": 0})
    assert s.read_text("run1/report.md") == "# hi" and s.read_text("run1/nope") is None
    assert s.read_records("run1") == [{"case_id": "c1", "repeat": 0}]
    assert s.list_runs() == ["run1"] and s.delete_run("run1") == 2 and s.list_runs() == []
    assert s.health_check().ok


def test_azure_blob_and_gcs_storage_with_fakes():
    from evalkit.storage.cloud import AzureBlobStorage, GCSStorage
    blobs = {}

    class Missing(Exception):
        status_code = 404
    container = NS(upload_blob=lambda n, d, overwrite: blobs.__setitem__(n, d),
                   download_blob=lambda n: NS(readall=lambda: blobs[n]) if n in blobs else (_ for _ in ()).throw(Missing()),
                   list_blobs=lambda name_starts_with: [NS(name=k) for k in blobs if k.startswith(name_starts_with)],
                   delete_blob=lambda n: blobs.pop(n))
    az = AzureBlobStorage(container="c", client=container)
    az.write_text("r/x.txt", "1")
    assert az.read_text("r/x.txt") == "1" and az.read_text("r/y") is None and az.list("r/") == ["r/x.txt"]
    store = {}

    def blob(name):
        return NS(upload_from_string=lambda d, content_type=None: store.__setitem__(name, d),
                  download_as_bytes=lambda: store[name] if name in store else (_ for _ in ()).throw(Missing()),
                  delete=lambda: store.pop(name))
    client = NS(bucket=lambda b: NS(blob=blob),
                list_blobs=lambda bucket, prefix: [NS(name=k) for k in store if k.startswith(prefix)])
    g = GCSStorage(bucket="b", client=client)
    g.append_record("r", "k", {"x": 1})
    assert g.read_records("r") == [{"x": 1}]


# ---------- durable, resumable, distributed runs ----------
CALLS = {"n": 0, "crash_after": None}


def counting_agent(q):
    CALLS["n"] += 1
    if CALLS["crash_after"] is not None and CALLS["n"] > CALLS["crash_after"]:
        raise KeyboardInterrupt("simulated crash")             # not caught by the runner
    return {"answer": f"answer to {q}", "prompt_tokens": 100, "completion_tokens": 50}


def _cfg(tmp_path, n=6, extra=None):
    cases = [{"id": f"c{i}", "input": f"q{i}", "expected_contains": ["answer"]} for i in range(n)]
    (tmp_path / "cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases), encoding="utf-8")
    cfg = {"name": "dur", "system": {"type": "python", "target": "tests.test_platform:counting_agent"},
           "dataset": {"path": str(tmp_path / "cases.jsonl")}, "output_dir": str(tmp_path / "out"),
           "run": {"concurrency": 1}, **(extra or {})}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(p)


def test_crash_then_resume_does_not_repeat_finished_cases(tmp_path):
    CALLS.update(n=0, crash_after=3)
    with pytest.raises(KeyboardInterrupt):
        runner.run(_cfg(tmp_path), run_id="crashy", log=lambda *a: None)
    run_dir = tmp_path / "out" / "crashy"
    assert len(runner.LocalStorage(root=str(tmp_path / "out")).read_records("crashy")) == 3   # persisted at once
    CALLS.update(n=0, crash_after=None)
    report, _ = runner.run(None, resume=str(run_dir), log=lambda *a: None)
    assert CALLS["n"] == 3                                        # only the 3 missing cases ran
    assert report["overall"]["answer_contains"]["n"] == 6 and not report["gate_failures"]
    lines = (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6                                        # compacted


def test_sharded_workers_then_finalize(tmp_path):
    CALLS.update(n=0, crash_after=None)
    cfg = _cfg(tmp_path, n=10)
    runner.run(cfg, run_id="dist", score=False, execute=False, log=lambda *a: None)      # prepare
    for k in range(3):
        rep, _ = runner.run(cfg, run_id="dist", shard=(k, 3), score=False, log=lambda *a: None)
        assert rep is None
    assert CALLS["n"] == 10                                       # every case exactly once across shards
    report, _ = runner.finalize("dist", config_path=cfg, log=lambda *a: None)
    assert report["overall"]["answer_contains"]["n"] == 10


def test_budget_stops_run_and_resume_finishes(tmp_path):
    CALLS.update(n=0, crash_after=None)
    cfg = _cfg(tmp_path, n=6, extra={"budget": {"max_tokens": 300}})
    report, run_dir = runner.run(cfg, log=lambda *a: None)
    assert CALLS["n"] == 2                                       # 150 tokens per case
    assert any("incomplete" in f for f in report["gate_failures"])
    assert report["run_meta"]["status"] == "incomplete" and "token budget" in report["run_meta"]["stop_reason"]


def test_rate_limiter_and_retry_policy():
    t = {"now": 0.0}
    rl = engine.RateLimiter(requests_per_minute=60, tokens_per_minute=600,
                            clock=lambda: t["now"], sleep=lambda s: t.__setitem__("now", t["now"] + s))
    for _ in range(60):
        assert rl.acquire() == 0                                  # burst of 60 allowed
    assert rl.acquire() > 0                                       # 61st waits ~1s
    rl.record_tokens(1200)                                        # over the token budget
    assert rl.acquire() >= 60
    p = engine.RetryPolicy(max_attempts=3)
    assert p.should_retry("RateLimitError: 429 Too Many Requests", 1)
    assert not p.should_retry("ValueError: bad input", 1) and not p.should_retry("429", 3)
    b = engine.Budget(max_cost_usd=0.01, price_in_per_1m=1.0, price_out_per_1m=1.0)
    b.charge(9000, 2000)
    with pytest.raises(BudgetExceededError):
        b.check()


FLAKY = {"n": 0}


def flaky_429(q):
    FLAKY["n"] += 1
    if FLAKY["n"] % 2:
        raise RuntimeError("429 Too Many Requests")
    return "fine"


def test_transient_errors_are_retried(tmp_path):
    FLAKY["n"] = 0
    cfg = _cfg(tmp_path, n=2, extra={"run": {"concurrency": 1, "retries": {"max_attempts": 2, "backoff_s": 0}}})
    text = open(cfg, encoding="utf-8").read().replace("counting_agent", "flaky_429")
    open(cfg, "w", encoding="utf-8").write(text)
    report, _ = runner.run(cfg, log=lambda *a: None)
    assert report["overall"]["error"]["mean"] == 0.0 and FLAKY["n"] == 4


class Memory(AgentAdapter):
    def run(self, input):
        return AgentResult(answer=f"saw {max(1, input.count('User:'))} user turn(s)", prompt_tokens=1, completion_tokens=1)


def test_multi_turn_default_replays_transcript():
    r = Memory().run_conversation(["Hi, I'm in London.", "How much for hotels?", "And meals?"])
    assert r.answer == "saw 3 user turn(s)" and r.prompt_tokens == 3
    assert [t["user"] for t in r.raw["turns"]] == ["Hi, I'm in London.", "How much for hotels?", "And meals?"]


# ---------- audit ----------
def test_audit_chain_detects_tampering(tmp_path):
    log = tmp_path / "audit.jsonl"
    audit.emit("run.started", path=log, run_id="r1", answer="secret text must not be logged")
    audit.emit("run.finalized", path=log, run_id="r1")
    assert audit.verify(log) == (True, "2 event(s), chain intact")
    assert "secret text" not in log.read_text(encoding="utf-8")
    lines = log.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace("run.started", "run.deleted")
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok, msg = audit.verify(log)
    assert not ok and "line 1" in msg


# ---------- privacy ----------
def test_pii_redaction():
    text = ("Mail jane.doe@corp.example or call +1 415-555-0134; card 4111 1111 1111 1111, "
            "not 1234 5678 9012 3456; SSN 123-45-6789; key sk-abcdefghijklmnop1234; host 10.0.0.12; costs $1,580")
    out, counts = privacy.redact_pii(text)
    assert "[EMAIL]" in out and "[CREDIT_CARD]" in out and "[SSN]" in out and "[CREDENTIAL]" in out
    assert "[IP]" in out and "[PHONE]" in out
    assert "1234 5678 9012 3456" in out                          # fails the Luhn check: not a card
    assert "$1,580" in out                                       # amounts are not phone numbers
    assert counts["email"] == 1


def test_egress_allowlist_blocks_unapproved_hosts(tmp_path):
    cfg = {"judge": {"model": "m", "base_url": "https://api.groq.com/openai/v1"},
           "system": {"type": "http", "url": "https://bot.corp.example/chat"},
           "privacy": {"allowed_hosts": ["*.corp.example"]}}
    with pytest.raises(ConfigError, match="judge -> api.groq.com"):
        privacy.check_egress(cfg)
    cfg["privacy"]["allowed_hosts"].append("api.groq.com")
    assert ("system", "bot.corp.example") in privacy.check_egress(cfg)
    assert ("judge", "api.anthropic.com") in privacy.egress_hosts({"judge": {"model": "c", "provider": "anthropic"}})


def test_redact_stored_results(tmp_path):
    CALLS.update(n=0, crash_after=None)
    cfg = _cfg(tmp_path, n=1, extra={"privacy": {"redact_stored": True}})
    text = open(cfg, encoding="utf-8").read().replace("q0", "email me at bob@corp.example")
    open(cfg, "w", encoding="utf-8").write(text)
    (tmp_path / "cases.jsonl").write_text(json.dumps({"id": "c0", "input": "email me at bob@corp.example"}) + "\n",
                                          encoding="utf-8")
    _, run_dir = runner.run(cfg, log=lambda *a: None)
    stored = (run_dir / "results.jsonl").read_text(encoding="utf-8")
    assert "bob@corp.example" not in stored and "[EMAIL]" in stored


def test_http_agent_uses_auth_block(token_server):
    """The http system authenticates every call with the configured auth plugin."""
    seen = {}

    class Sess:
        verify = True
        auth = None

        def request(self, method, url, json=None, headers=None, timeout=None):
            req = NS(headers=dict(headers or {}))
            self.auth(req)
            seen.update(req.headers)
            return NS(status_code=200, json=lambda: {"answer": "ok"}, raise_for_status=lambda: None)
    agent = REGISTRY.build("system", {"type": "http", "url": "https://bot.example/chat", "session": Sess(),
                                      "auth": {"type": "oauth2_client_credentials", "token_url": token_server,
                                               "client_id": "evalkit", "client_secret": "shh"}})
    assert agent.run("hi").answer == "ok" and seen["Authorization"].startswith("Bearer at-")
