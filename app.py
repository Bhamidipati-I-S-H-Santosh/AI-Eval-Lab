"""Web UI for the AI Eval Lab.

    streamlit run app.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from src.ingest import build_chunks
from src.retrieve import Retriever

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

st.set_page_config(page_title="AI Eval Lab", page_icon="🧪", layout="wide")


@st.cache_resource(show_spinner="Building retriever (first time may download models)...")
def get_retriever(method: str, rerank: bool) -> Retriever:
    return Retriever(build_chunks(), method=method, rerank=rerank)


# ---------- sidebar ----------
with st.sidebar:
    st.title("🧪 AI Eval Lab")
    st.caption("Northwind Labs policy assistant")
    method = st.selectbox("Retrieval method", ["bm25", "vector", "hybrid"], index=0)
    rerank = st.checkbox("Cross-encoder rerank", value=False)
    top_k = st.slider("Top-k chunks", 1, 8, 3)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
    st.divider()
    st.caption(f"LLM: `{os.getenv('LLM_MODEL', '?')}`")
    st.caption(f"Endpoint: `{os.getenv('LLM_BASE_URL', '?')}`")

tab_ask, tab_agent, tab_eval, tab_kit = st.tabs(
    ["💬 Ask (RAG)", "🤖 Agent", "📊 Eval results", "🧰 evalkit reports"])

# ---------- RAG ----------
with tab_ask:
    st.subheader("Ask a policy question")
    q = st.text_input("Question", "Can I fly business class to Singapore?", key="rag_q")
    if st.button("Answer", type="primary", key="rag_go") and q.strip():
        from src.rag import answer
        with st.spinner("Retrieving and answering..."):
            try:
                text, chunks = answer(q, get_retriever(method, rerank), top_k=top_k)
            except Exception as e:  # show API errors in the UI instead of crashing
                st.error(f"{type(e).__name__}: {e}")
                st.stop()
        st.success(text)
        st.markdown("**Retrieved chunks**")
        for i, c in enumerate(chunks, 1):
            with st.expander(f"{i}. {c.id} — {c.title}"):
                st.write(c.text)

# ---------- Agent ----------
with tab_agent:
    st.subheader("Tool-calling agent")
    aq = st.text_input("Task", "I'm going to London for 4 nights. Max I can claim for hotel and meals?",
                       key="agent_q")
    max_steps = st.slider("Max steps", 1, 10, 6)
    if st.button("Run agent", type="primary", key="agent_go") and aq.strip():
        from src.agent import run_agent
        with st.spinner("Agent working..."):
            try:
                r = run_agent(aq, max_steps=max_steps)
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
                st.stop()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("LLM calls", r.llm_calls)
        c2.metric("Tool calls", len(r.steps))
        c3.metric("Tokens", r.prompt_tokens + r.completion_tokens)
        c4.metric("Latency", f"{r.latency_s:.1f}s")
        st.markdown("**Answer**")
        st.info(r.answer or f"(no answer — stopped: {r.stopped_reason})")
        st.markdown("**Trace**")
        for i, s in enumerate(r.steps, 1):
            with st.expander(f"Step {i}: `{s.tool}`"):
                st.json(s.args)
                try:
                    st.json(json.loads(s.result))
                except Exception:
                    st.code(s.result)

# ---------- Eval results ----------
with tab_eval:
    st.subheader("RAG evaluation runs")
    c1, c2 = st.columns([1, 3])
    retrieval_only = c1.checkbox("Retrieval only (no LLM, instant)", value=True)
    if c1.button("▶ Run evaluation", type="primary"):
        cmd = [sys.executable, "-m", "src.evaluate", "--method", method, "--top-k", str(top_k)]
        if rerank:
            cmd.append("--rerank")
        if retrieval_only:
            cmd.append("--retrieval-only")
        with st.spinner(f"Running: {' '.join(cmd[2:])}"):
            p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                               env={**os.environ, "PYTHONUTF8": "1"})
        c2.code((p.stdout or "")[-3000:] + (p.stderr or "")[-2000:] if p.returncode else p.stdout[-3000:])

    hist = RESULTS / "history.csv"
    if hist.exists():
        df = pd.read_csv(hist)
        st.markdown("**History** (`results/history.csv`)")
        st.dataframe(df, width='stretch')
        metrics = [m for m in ["hit_rate", "mrr", "correctness", "faithfulness", "abstention"]
                   if m in df and df[m].notna().any()]
        if metrics and len(df) > 1:
            st.line_chart(df.set_index("timestamp")[metrics])

    runs = sorted((RESULTS / "runs").glob("*.json"), reverse=True)
    if not runs:
        st.info("No saved runs yet. Click **Run evaluation** above.")
    else:
        pick = st.selectbox("Inspect a run", runs, format_func=lambda p: p.stem)
        data = json.loads(pick.read_text(encoding="utf-8"))
        s = data["summary"]
        cols = st.columns(5)
        for col, key in zip(cols, ["hit_rate", "mrr", "correctness", "faithfulness", "abstention"]):
            col.metric(key, "—" if s.get(key) is None else s[key])
        st.caption(f"method={s['method']} · rerank={s['rerank']} · chunking={s['chunking']} · top_k={s['top_k']}")
        rows = pd.DataFrame(data["rows"])
        only_fail = st.checkbox("Show failures only")
        if only_fail:
            mask = pd.Series(False, index=rows.index)
            for k in ["hit", "correct", "faithful", "abstained"]:
                if k in rows:
                    mask |= rows[k].fillna(1) == 0
            rows = rows[mask]
        st.dataframe(rows, width='stretch', height=500)

# ---------- evalkit ----------
with tab_kit:
    st.subheader("evalkit reports")
    kit = RESULTS / "evalkit"
    reports = sorted(kit.glob("*/report.md"), reverse=True) if kit.exists() else []
    if not reports:
        st.info("No evalkit runs yet. Try: `python -m evalkit run examples/qdrant_retrieval.yaml`")
    else:
        rp = st.selectbox("Report", reports, format_func=lambda p: p.parent.name)
        st.markdown(rp.read_text(encoding="utf-8"))
