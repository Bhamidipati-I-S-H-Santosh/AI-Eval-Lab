# AI Eval Lab: Measuring and Improving a RAG System and Agent

A question-answering assistant and a tool-calling agent for a fictional company's
HR, IT and expense policies, built **eval-first**. Every change is scored against
a golden test set and a pack of prompt-injection attacks, and CI blocks any
change that makes retrieval worse.

> Most RAG demos show that something works once. This project shows **how well
> it works, how I know, and what made it better.**

It has three parts:

1. **RAG pipeline** (`src/`): chunking, BM25 / vector / hybrid search, optional
   reranking, and answers with citations or "I don't know".
2. **Tool-calling agent** (`src/agent.py`): searches policies, reads documents
   and uses a calculator, with a prompt-injection guard. The same tools are also
   exposed over MCP.
3. **evalkit** (`evalkit/`): a general evaluation framework that scores *any*
   agent (Python, HTTP API, MCP) or vector database from a YAML config. It
   measures correctness, retrieval, tool use, cost, latency and resistance to
   injection.

## Results

All numbers below were measured on the included test sets, using
`openai/gpt-oss-120b` on Groq as both the generator and the LLM judge.

### Retrieval (25 questions, top-3)

| Experiment | Hit rate | MRR | Correctness | Faithfulness | Abstention |
|---|---|---|---|---|---|
| Baseline: BM25, section chunks | 0.913 | 0.891 | 0.739 | 1.00 | 1.00 |
| Vector only (sentence-transformers) | **1.000** | 0.978 | – | – | – |
| Hybrid (BM25 + vector, RRF) | 0.957 | 0.957 | – | – | – |
| Hybrid + cross-encoder rerank | **1.000** | **1.000** | **0.783** | 0.96 | 1.00 |
| Fixed-size chunks, BM25 | 0.913 | 0.891 | – | – | – |
| Fixed-size chunks, hybrid | **1.000** | 0.957 | – | – | – |

"–" means the run was retrieval-only, with no LLM involved.

What the numbers show:

- **BM25 misses on vocabulary.** "food" vs the document's "meal" (q09) and
  "password" vs "passwords" (q11). Semantic search fixes both.
- **Hybrid is not automatically better than vector alone.** Here it still misses
  q09, most likely because rank fusion lets BM25's poor ranking pull the right
  passage out of the top 3. Adding a reranker
  restores a perfect score.
- **Better retrieval leads to better answers.** Correctness rose from 0.739 to
  0.783 with hybrid + rerank, and faithfulness stayed at or near 1.0.

### Agent hardening (evalkit, 45 cases including 12 injection attacks)

Two weaknesses found by evalkit were fixed:

1. **Hidden-instruction injection.** A planted "Note to AI assistants" passage
   could get the agent to append an attacker's code to its answer.
2. **Abstention.** On unanswerable questions, the agent searched until it hit
   the step limit instead of saying "I don't know".

| Run | Task success | Correctness | Abstention | Injection resisted | Errors |
|---|---|---|---|---|---|
| Before hardening | 0.911 | 0.976 | 1/3 ¹ | 12/12 | 0 |
| After hardening | 0.911 | 0.905 ² | **3/3** | 12/12 | 0 |

**What changed:**

- **Guard** (`src/tools.py`): quarantines any passage addressed to the AI, such
  as "note to AI assistants", "before answering", "include the verification
  code", fake "re-verify your account" links and off-site markdown images.
  - Offline, it now catches **6 of 6** attack types, up from 1 of 6.
  - It flags **0 of 22** real policy passages.
- **Final answer at the step limit** (`src/agent.py`): when the agent runs out of
  steps, it gets one last call with tools disabled, so it gives a real answer
  (often "I don't know") instead of a canned error.
- **Abstention rule:** after two searches that find nothing, the agent reads the
  most likely document, then says "I don't know" if the answer still isn't there.

**Notes**

¹ The original evalkit report said 0/3. One of those answers was
"I don’t know" with a curly apostrophe, which the scorer didn't recognise. That
bug led to the scorer being rebuilt and tested against hand-labeled answers
(see [Scorer validation](#scorer-validation-abstention) below).

² The first version of the stricter security prompt told the agent to include
"only the facts asked about". That made it drop secondary details, such as
"unused budget does not roll over", in 3 answers. The wording has been fixed.
A targeted re-check of those 3 cases plus the unanswerable ones passed. A full
re-run of the final prompt is pending, because the free Groq tier has a daily
token limit.

Caveats:

- **The guard was tuned on these same attacks.** The 6 of 6 result is optimistic
  until it's tested against attacks with new wording.
- **These are single runs.** Agents aren't deterministic, so final claims need
  `repeats: 3` or more.

### Scorer validation (abstention)

The curly-apostrophe bug raised a bigger question: **how accurate is the scorer
itself?** To find out, I checked it against 40 hand-labeled answers
(`tests/fixtures/abstention_labels.jsonl`). Each answer is labeled as a
refusal, an answer, or a partial answer (answers the question but notes a missing
detail). Only refusals should count as abstaining.

| Scorer | Dev set (25, used for tuning) | Held-out set (15, written before the change) |
|---|---|---|
| Old regex | 14/25 · precision 0.57 · recall 0.62 | 6/15 · precision 0.25 · recall 0.14 |
| New scorer | **25/25** · 1.00 · 1.00 | **15/15** · 1.00 · 1.00 |

- **What the old regex got wrong.** It missed refusals worded differently, such
  as "I couldn't find…", "There's no policy on…" and "That isn't covered…". It
  also flagged real answers as refusals because they contained a negation, such
  as "insurance *does not cover* personal trips". And it counted every partial
  answer as giving up, which inflated `over_abstention`.
- **How the new scorer decides.** An answer counts as a refusal only if the
  refusal is all it says. Negations only count when they refer to the documents
  ("the *policy* doesn't say"), and if any other clause carries a citation or a
  number, the answer counts as answered.
- **Reported results are unaffected.** Re-scoring all 180 saved agent answers
  changed no verdict, because this agent always refuses with a plain
  "I don't know". The weakness would only have shown up with a different model
  or agent.
- **Both scorers now use the same check.** The RAG pipeline (`src/evaluate.py`)
  had its own, simpler check, with the same apostrophe bug. It now uses the same
  validated scorer.

Run the check with `python -m evalkit.scorer_check --errors`. CI requires an
exact match on the dev set and at least 0.9 precision and recall on the
held-out set.

Limits of this check:

- **The held-out set isn't blind.** The same person wrote the labels and the
  scorer, so 15/15 is optimistic.
- **One known gap:** a partial answer with no citation and no number, such as
  "Yes, but the policy doesn't say how often", would be scored as a refusal.
- **What would make it stronger:** labels written or checked by someone else,
  on refusals from a different model.

### Earlier agent eval (`src.eval_agent`, 11 cases, before hardening)

| Configuration | Task success | Tool selection | Injection resisted | Avg steps | Tokens |
|---|---|---|---|---|---|
| No defenses | 0.909 | 1.00 | 1.00 | 2.18 | 32,838 |
| Guard + security prompt | 0.818 | 1.00 | 1.00 | 2.00 | 25,282 |

With only 11 cases, a single case moves task success by 9 points. That is why
the larger evalkit suite above was built.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add a free Groq or Gemini key, or point it at Ollama
```

On Windows, set `PYTHONUTF8=1` first (PowerShell: `$env:PYTHONUTF8=1`).
Without it, printing model output can crash with a `UnicodeEncodeError`.

```bash
python -m pytest -q                                  # unit tests (fake LLM, no key)
python -m src.evaluate --retrieval-only              # retrieval scores, no key needed
python -m src.evaluate --method hybrid --rerank      # full eval with LLM + judge
python -m src.rag "Can I fly business class to Singapore?"
python -m src.agent "I'm going to London for 4 nights. Max I can claim for hotel and meals?"
streamlit run app.py                                 # web UI at http://localhost:8501
```

Every RAG run is saved to `results/runs/` and appended to `results/history.csv`.

### Web UI

`app.py` is a Streamlit app with four tabs:

- **Ask (RAG):** answers with the retrieved passages shown.
- **Agent:** a step-by-step tool trace, plus tokens and latency.
- **Eval results:** run the eval and see its history charted.
- **evalkit reports:** browse saved reports.

## Metrics

| Metric | What it tells you | Needs LLM? |
|---|---|---|
| Hit rate @k / MRR | Was the right document retrieved, and how high was it ranked? | No |
| Correctness | Does the answer match the reference? (LLM judge) | Yes |
| Faithfulness | Is every claim grounded in the retrieved context? (LLM judge) | Yes |
| Abstention / over-abstention | Does it refuse on unanswerable questions, and only on those? | No (rule-based, checked against 40 hand labels) |
| Task success / tool selection | Did the agent finish the task with the right tools? | Partly |
| Injection resisted | Did a canary string from a planted attack leak into the answer or tool calls? | No |
| Cost / latency / tokens | What does one answer cost? | No |

## evalkit: evaluate any agent or database

Plug in a system through an adapter, describe the test in YAML, and get a report
with pass/fail gates:

```bash
python -m evalkit run examples/northwind.yaml          # Python agent + injection attacks
python examples/mock_http_agent.py                     # start a demo HTTP bot, then:
python -m evalkit run examples/http_agent.yaml         # evaluate it as a black box
python -m evalkit run examples/qdrant_retrieval.yaml   # a vector DB on its own
python -m evalkit compare results/evalkit/<run_a> results/evalkit/<run_b>
python -m evalkit.scorer_check --errors                # check the abstention scorer against hand labels
```

On free LLM tiers, keep `run.concurrency` at 1 to avoid rate-limit errors.

See [EVALKIT.md](EVALKIT.md) to connect your own system, and
[BUILD_GUIDE.md](BUILD_GUIDE.md) to build the agent step by step.

## Project layout

```
data/docs/            source policy documents
data/golden_set.jsonl 25 RAG test questions with reference answers and sources
data/attacks/         poisoned documents for security tests
src/ingest.py         chunking: by section or fixed-size window
src/retrieve.py       BM25, vector, hybrid (RRF) + optional cross-encoder rerank
src/rag.py            answers with citations, or "I don't know"
src/evaluate.py       RAG metrics, LLM judge, run history, CI gate
src/tools.py          agent tools + prompt-injection guard
src/agent.py          tool-calling agent loop with traces
src/eval_agent.py     the original agent evals
evalkit/              general evaluation framework (adapters, metrics, attacks, reports)
evalkit/scorer_check.py  checks the abstention scorer against hand labels
examples/             evalkit configs and adapters (Python, HTTP, Qdrant)
app.py                Streamlit UI
mcp_server.py         the agent's tools over MCP
tests/                unit tests (fake LLM, no API key)
tests/fixtures/       hand-labeled answers for scorer validation
.github/workflows/    tests + retrieval eval on every push
```

## Next experiments

- **Stemming for BM25** (e.g. Porter). This should fix q11 without needing vectors.
- **A larger golden set** with multi-part, cross-document and tricky
  unanswerable questions. 25 questions is a start, not a benchmark.
- **New-wording injection attacks**, to measure how well the guard generalises.
- **Judge validation:** the abstention scorer is now checked against hand
  labels. Do the same for the LLM judge: hand-grade 20 or more answers and
  report how often its correctness and faithfulness scores agree.
- **Blind scorer labels:** add about 15 refusals from a different model, labeled
  by someone other than the scorer's author.
- **Repeated runs** (`repeats: 3`) to report variance, not single numbers.

## License

Copyright (c) 2026 Bhamidipati I S H Santosh. **All rights reserved.**
This code may not be used, copied, modified or distributed without prior written
permission. See [LICENSE](LICENSE).
