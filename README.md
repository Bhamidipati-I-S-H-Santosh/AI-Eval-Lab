# AI Eval Lab — Measuring and Improving a RAG System

A question-answering assistant over a (fictional) company's HR, IT and expense
policies, built **eval-first**: every change is scored against a golden test
set, and CI blocks changes that reduce retrieval quality.

> Most RAG demos show that something works once. This project shows **how well
> it works, how I know, and what made it better.**

## What it measures

| Metric | What it tells you | Needs LLM? |
|---|---|---|
| Hit rate @k | Was the right document retrieved? | No |
| MRR | How high was it ranked? | No |
| Correctness | Does the answer match the reference? (LLM judge) | Yes |
| Faithfulness | Is every claim grounded in the context? (LLM judge) | Yes |
| Abstention | Does it say "I don't know" on unanswerable questions? | Yes |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add a free Groq or Gemini key, or use Ollama

python -m src.evaluate --retrieval-only              # no API key needed
python -m src.evaluate                               # full eval with LLM + judge
python -m src.evaluate --method hybrid --rerank      # try improvements
python -m src.rag "Can I fly business class to Singapore?"
```

Every run is saved to `results/runs/` and appended to `results/history.csv`.

## Project layout

```
data/docs/            source documents (swap in your own)
data/golden_set.jsonl 25 test questions with reference answers + source file
src/ingest.py         chunking: by section or fixed-size window
src/retrieve.py       BM25, vector, hybrid (RRF) + optional cross-encoder rerank
src/rag.py            answer with citations, or "I don't know"
src/evaluate.py       metrics, LLM-as-judge, run history, CI gate
src/tools.py          agent tools + prompt-injection guard
src/agent.py          tool-calling agent loop with traces
src/eval_agent.py     agent evals: tools, success, security, cost
mcp_server.py         the same tools over MCP
data/attacks/         poisoned documents for security tests
tests/                unit tests (fake LLM, no API key)
.github/workflows/    tests + retrieval eval on every push
```

## Results (fill this in as you go)

| Experiment | Hit rate | MRR | Correctness | Faithfulness | Abstention |
|---|---|---|---|---|---|
| Baseline: BM25, section chunks, k=3 | 0.913 | 0.891 | | | |
| + stemming | | | | | |
| Hybrid (BM25 + vector) | | | | | |
| Hybrid + rerank | | | | | |
| Fixed-size chunks | | | | | |

## Week 1 exercises

1. **Read the misses.** The baseline misses q09 ("food" vs the doc's "meal")
   and q11 ("password" vs "passwords"). This is the classic keyword-search failure.
2. **Fix q11 with stemming.** Add a simple stemmer to `tokenize()` (e.g.
   `nltk.stem.PorterStemmer`), re-run, and record the new score.
3. **Fix q09 with semantics.** Run `--method hybrid`. Vector search understands
   that "food" relates to "meal".
4. **Run the full eval** with a free LLM. Open `results/runs/*.json` and read
   every row where `correct` or `faithful` is 0.
5. **Check the judge.** Hand-grade 10 rows yourself. How often does the LLM judge
   agree with you? Write that percentage in this README; judges need checking too.
6. **Make it harder.** Add 15 more questions: multi-part questions, questions
   that need two documents, and tricky unanswerable ones. A 25-question set is
   a start, not a benchmark.
7. **Swap in real data.** Replace `data/docs/` with a public dataset (e.g.
   your city's government policies or an open-source project's docs) and write
   a new golden set.

## Agent layer

A tool-calling agent (`src/agent.py`) uses the same retriever as a tool, plus a
calculator and document reader. The same tools are exposed over MCP
(`mcp_server.py`) for the Claude desktop app, Cursor and other MCP clients.

```bash
python -m src.agent "I'm going to London for 4 nights. Max I can claim for hotel and meals?"
python -m src.eval_agent                  # tool choice, task success, injection resistance
python -m src.eval_agent --no-defenses    # security baseline
```

| Agent experiment | Task success | Tool selection | Injection resisted | Avg steps |
|---|---|---|---|---|
| No defenses | | | | |
| Guard + security prompt | | | | |

## evalkit: evaluate any agent or database

`evalkit/` is a general evaluation framework: plug in any agent (Python, HTTP API,
MCP-based) or any database (Qdrant, a folder, or a custom adapter), and score
correctness, retrieval, tool use, cost and prompt-injection resistance from a
YAML config. The Northwind agent is just the first system plugged in.

```bash
python -m evalkit run examples/northwind.yaml     # agent + injection attacks
python -m evalkit run examples/http_agent.yaml    # any HTTP chatbot
python -m evalkit run examples/qdrant_retrieval.yaml   # a vector DB on its own
```

**Guides:** [BUILD_GUIDE.md](BUILD_GUIDE.md) (build the agent step by step) ·
[EVALKIT.md](EVALKIT.md) (connect and evaluate any system)
