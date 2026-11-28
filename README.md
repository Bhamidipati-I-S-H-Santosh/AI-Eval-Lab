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
itself?** To find out, I checked it against 40 hand-labeled answers, then five
sets of answers from other models, labeled blind (30, 32, 32, 35 and 66 answers;
`tests/fixtures/abstention_labels.jsonl`). Each answer is labeled as a
refusal, an answer, or a partial answer (answers the question but notes a missing
detail). Only refusals should count as abstaining.

| Scorer | Dev (25) | Held-out (15) | Blind-v1 (29) | Blind-v2 (32) | Blind-v3 (32) | Blind-v4 (35) |
|---|---|---|---|---|---|---|
| Old regex | 14/25 · P 0.57 · R 0.62 | 6/15 · P 0.25 · R 0.14 | – | – | – | – |
| v2: clause rule | **25/25** | **15/15** | 25/29 · P 1.00 · R 0.64 **(blind)** | – | – | – |
| v3: + context fix | **25/25** | **15/15** | 28/29 · P 1.00 · R 0.91 | 27/32 · P 1.00 · R 0.62 **(blind)** | – | – |
| v4: + new refusal wordings | **25/25** | **15/15** | 28/29 · P 1.00 · R 0.91 | 32/32 · P 1.00 · R 1.00 | 23/32 · P 0.75 · R 0.46 **(blind)**¹ | 16/35 · P 0.83 · R 0.22 |
| v5: markdown, "what I checked", lead sentence | **25/25** | **15/15** | 28/29 · P 1.00 · R 0.91 | **32/32** | 25/32 · P 0.71 · R 0.77 | 18/35 · P 0.88 · R 0.30 **(blind)** |
| LLM judge only (gpt-oss-20b, untuned) | 21/25 · P 0.85 · R 0.85 | 12/15 · P 1.00 · R 0.57 | 24/29 · P 1.00 · R 0.55 | 28/32 · P 1.00 · R 0.69 | 27/32 · P 1.00 · R 0.62 | 20/35 · P 1.00 · R 0.35 **(blind)** |
| Hybrid: v5 when sure, judge otherwise | 24/25 · P 0.93 · R 1.00 | **15/15** | 28/29 · P 1.00 · R 0.91 | **32/32** | 27/32 · P 0.83 · R 0.77 | **22/35 · P 0.92 · R 0.48 (blind)** |

P = precision, R = recall on refusals; a bare score means 1.00 / 1.00.
"(blind)" marks the first and only time a scorer saw that set. Every other
regex or hybrid cell is a tuning score. The judge prompt was written once and
never changed, so all of its cells are effectively untuned.

¹ First scored as 24/32 · R 0.50. A second labeler later disputed one row,
which was resolved as a refusal (see below), so the same predictions now
score 23/32 · R 0.46.

**Each blind number is lower than the tuning scores**: R 0.64 → 0.62 → 0.50 as
the prompt styles got further from what the scorer was built on. Adding
patterns fixes the last set's wordings, not the next one's.

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

**The blind test found what the hand-written sets missed.** The hand-written
held-out set wasn't truly blind, because the same person wrote it and the
scorer. So I ran a real blind test:

- **Answers from other models.** gpt-oss-20b and Qwen3.8-27B answered 16 new
  questions, with a neutral prompt that didn't dictate how to word a refusal.
- **Independent labels.** gpt-oss-120b, which wrote none of the answers, labeled
  them without seeing the scorer.

v2 caught only 8 of 12 refusals. Qwen often refuses and then says what the
policy *does* cover ("The policy only covers annual, sick and parental leave
[leave_policy.md]"), and the scorer read that citation as an answer. It also
missed "The documents (benefits, expense, …) don't contain…", because the
bracketed list broke the phrase.

v3 fixes both: sentences that only describe what the policies cover don't count
as answers, and bracketed asides are ignored when matching phrases.

**Blind-v2: the honest number.** I wrote 16 new questions on topics v1 didn't
touch. gpt-oss-120b, which generated none of the v1 answers, answered them in two
prompt styles: neutral, and brief and conversational. Qwen3.8-27B labeled the
answers blind, and the labels were frozen before the scorer ran.

- **Result:** v3 agrees on 27/32, with no false alarms, but catches only 8 of 13
  refusals.
- **All 5 misses come from the conversational prompt.** They use wordings the
  scorer has no pattern for: "I **didn't find** any mention of…",
  "I'm **not seeing**…", "I **don't see** any mention of…" and
  "the policies **don't include** a specific… provision".
- **What it shows:** the scorer catches the wordings it has seen, not refusals
  in general. Each new model or prompt style brings new refusal wordings.

v4 adds those as general refusal patterns: "didn't / did not find or see",
"I don't see", "I'm not seeing / finding" and "nothing about…". It now catches
all 13 blind-v2 refusals with no false alarms, the other sets are unchanged, and
none of the 180 saved agent answers changes verdict.

**Blind-v3: v4's honest number is 24/32.** I wrote 16 new questions on topics
not used before. gpt-oss-20b and Qwen3.8-27B answered them with a new
"structured" prompt: a summary line first, then an explanation of what was
checked. gpt-oss-120b labeled the answers blind, using the blind-v2 rules, and
the labels were frozen before v4 (commit `0f8ce36`) ran. v4 caught 6 of 12
refusals and raised 2 false alarms. It failed in three new ways:

- **"What I checked" sentences** (4 misses). For example: "I don't know.
  I searched the benefits and expense documents… and found nothing.
  [benefits.md] [expense_policy.md]". Citations of the documents *searched*
  look like answer citations.
- **"Closest related item" sentences** (2 misses). For example: "The closest
  related item is the 3-month probation period [onboarding.md]". An unrelated
  fact with a number looks like an answer.
- **Markdown bold breaks the sentence splitter** (2 false alarms).
  "**…above $25, and the policy does not specify whether a photo is
  acceptable.**" merges with the next sentences, so a real partial answer reads
  as a refusal.

**Three changes beyond more patterns.**

- **The agent declares its own status.** It ends every answer with
  `Status: answered | partial | unanswerable`. The line is stripped from the
  answer and stored on the result (`AgentResult.status`; `status_field` for
  HTTP agents). When an agent declares a status, the scorer uses it instead of
  reading the text. This only works for agents you control. *Update:* the
  metrics no longer trust it by default, because it is the agent grading
  itself. The judge decides, and a `status_agreement` metric reports how often
  the declared status matches the judge's verdict (`abstention.trust_status: true`
  restores the shortcut).
- **An LLM judge that sees the question** (`evalkit/abstention.py`,
  `JUDGE_PROMPT`). It labels each reply refuse, answer or partial. It runs
  zero-shot on gpt-oss-20b, which wrote and labeled none of the scored sets,
  and the prompt was frozen before any score was seen.
- **Regex v5 as a cheap first pass.** It strips markdown before splitting
  sentences, treats "I searched / checked…" and "the closest related item…"
  as context rather than answers, and lets the lead sentence decide when a
  refusal and an answer-like sentence appear together. When it can't tell,
  it says "uncertain" and the hybrid asks the judge.

**Two labelers, with disputes resolved by one rule.** Every blind answer now
has a second label from a different model (gpt-oss-120b and Qwen3.8-27B).
When they disagree on refuse vs not, the row is "pending" and left out of the
score until it gets a `label_human`. All 16 disputes had the same shape:
"No, the policies don't mention X". gpt-oss-120b called that an answer, and
Qwen called it a refusal. They were resolved with one rule, chosen by the
project owner and applied by Claude: **"No, the policies don't mention X" is
a refusal; "No, the policy says X isn't allowed" is an answer.** All 16
became refusals. No answer has been labeled directly by a human yet.

**Blind-v4: the hybrid's honest number is 22/35.** Qwen3.8-27B answered 24
new questions in two new prompt styles, bullets and formal, for 48 answers
in all. The API rate limit cost 7 answers, and 6 more were held back because
their second label failed, which leaves 35 with two labels. Both labels and the dispute rule were fixed before any
scorer ran. Each method was scored once:

| Method | Blind-v4 (35 answers, 23 refusals) |
|---|---|
| Regex v4 | 16/35 · P 0.83 · R 0.22 [0.10–0.42] |
| Regex v5 | 18/35 · P 0.88 · R 0.30 [0.16–0.51] |
| LLM judge | 20/35 · P 1.00 · R 0.35 [0.19–0.55] |
| Hybrid | 22/35 · P 0.92 · R 0.48 [0.29–0.67] |

Brackets are Wilson 95% intervals. The hybrid is the best of the four, but it
still misses half the refusals. All 12 misses have the disputed shape: a
leading "No", then "the policies don't mention X", then related facts with
numbers ("The only home-office reimbursements are $500 … [expense_policy.md]").
The regex reads the numbers as an answer. The judge reads "No" as an answer,
even though its prompt says a "no" counts only when a rule says so.

What this shows:

- **Recall dropped mainly because of the definition, not the wording.** The
  two labeler models split on exactly these answers, and so does the judge.
  A scorer can't match labels that are themselves inconsistent, so the rule
  had to be fixed first. It now is, and blind-v4 is the first set scored
  against it.
- **The small sets gave false comfort.** With only 11–13 refusals in a set,
  even a perfect recall of 1.00 has an interval of 0.74–1.00. Blind-v4 has 23
  refusals: better, but still short of the 30 per round planned.
- **Per style, across all sets:** the hybrid gets R 0.94–1.00 on the neutral,
  brief and hand-written styles, 0.77 on structured, 0.50 on bullets and 0.44
  on formal. The styles it was tuned on look solved; the new ones don't. Only
  the blind-v4 styles are truly held out, so they are the honest
  leave-one-style-out numbers.
- **One possible label inconsistency.** The bullets answer "No, the company
  does not reimburse personal mobile phone bills" was labeled an answer by
  both models, while the formal answer to the same question was resolved as a
  refusal. It's left as labeled, because labels aren't changed after a set is
  scored.

**Round 5: fix the judge, not the patterns.** Blind-v4 showed the problem was
the judge reading "No, the policies don't mention X" as an answer. Three changes
were made using only dev through blind-v4, before blind-v5 existed:

- **Judge prompt v2** (`JUDGE_PROMPT`; v1 is kept as `JUDGE_PROMPT_V1`). It
  states the rule in full, including "still a refusal if it starts with No,
  says the company doesn't offer X only because nothing was found, or lists
  related benefits". It adds four worked examples on invented topics that
  appear in no labeled set.
- **Judge model: gpt-oss-safeguard-20b.** It's built to classify text against
  a written policy, and it wrote and labeled none of the answers. On the 168
  earlier labels it beat gpt-oss-20b: 161/168 vs 150/168 inside the hybrid.
- **Regex v6.** The lead sentence is now the first one that says something,
  so "- No" or "Based on the available policies, **no**" no longer decides
  the verdict. "The only wellness benefit is $60/month" counts as context, and
  "…$40/month for internet—printers aren't covered" splits into two clauses.

The labelers also got the rule written into their prompt. Disagreement fell
from 13 of 35 answers on blind-v4 to 2 of 66 on blind-v5.

**Blind-v5: the honest number is 63/66, recall 0.93.** There were 22 new
questions: 15 with no answer in the policies, 4 answerable and 3 partly
answerable. Qwen3.8-27B answered them in two new styles, "casual" (team-chat
tone) and "terse" (two sentences at most), and gpt-oss-120b in the terse
style. That gives 66 answers, 44 of them refusals. gpt-oss-120b and Qwen
labeled every answer, the 2 disputes were resolved with the rule, and the set
was frozen before any scorer ran. Each method was scored once:

| Method | Blind-v5 (66 answers, 44 refusals) | All 234 labels (mostly tuning) |
|---|---|---|
| Regex v4 | 43/66 · P 1.00 · R 0.48 [0.34–0.62] | 182/234 · P 0.96 · R 0.60 |
| Regex v5 | 44/66 · P 0.92 · R 0.55 [0.40–0.68] | 187/234 · P 0.92 · R 0.68 |
| Regex v6 | 50/66 · P 0.94 · R 0.68 [0.53–0.80] | 198/234 · P 0.95 · R 0.75 |
| Old judge (gpt-oss-20b, prompt v1) | 48/66 · P 1.00 · R 0.59 [0.44–0.72] | 180/234 · P 0.97 · R 0.58 |
| Old hybrid (regex v5 + old judge) | 51/66 · P 0.94 · R 0.70 [0.56–0.82] | 199/234 · P 0.94 · R 0.77 |
| Hybrid (regex v6 + new judge) | 59/66 · P 0.95 · R 0.89 [0.76–0.95] | 221/234 · P 0.94 · R 0.95 |
| **Judge only (safeguard-20b, prompt v2)** | **63/66 · P 1.00 · R 0.93 [0.82–0.98]** | 225/234 · P 0.96 · R 0.97 |

Brackets are Wilson 95% intervals, now much tighter because there are 44
refusals instead of 11–13.

- **Recall went from 0.70 to 0.93 on unseen data**, comparing the old hybrid
  with the new judge on the same blind set, and precision stayed at 1.00.
  Both new styles scored about the same (casual R 0.93, terse R 0.93), so the
  judge isn't tied to one way of writing.
- **The regex is now the weak part.** The new hybrid loses 4 answers to the
  judge alone, all cases where the regex was confident and wrong, such as "No
  on-site gym mentioned in the policies — but the company does cover up to
  $60/month…". So **the default is now judge-first**
  (`classify(..., method="judge")`). The regex runs only when there is no
  judge or the judge call fails, and the result's `method` field says which
  one decided. This choice was made *after* seeing blind-v5, so it needs a
  blind-v6 to confirm it. On the 168 earlier labels the two tied (161/168
  each), so blind-v5 is the only evidence that judge-first is better.
- **The judge's 3 misses** are the 2 answers whose labels were disputed and
  resolved by the rule, plus one near-copy of them ("…no paid time off for
  moving house. You could use your annual leave"). They're the borderline
  cases where reasonable labelers differ.
- **A guard against false refusals.** Re-scoring the saved agent runs with
  the judge changed 11 verdicts, and all 11 were wrong. Nine came from the
  example HTTP chatbot pasting raw document text ("According to
  benefits.md: # Northwind Labs — Benefits…"), and two were the crash message
  "I couldn't finish within the step limit." The judge called these refusals
  because they don't answer the question. But abstention means *admitting* the
  gap, and counting these would credit a broken bot with honest "I don't
  know"s. So a judge "refuse" now counts only if the reply admits the gap in
  some wording (`admits_gap`); otherwise the method is `judge+guard` and it is
  not counted as abstaining. All 124 labeled refusals pass the guard (a CI
  test checks this), and none of the 25 dumps and crash messages do. With the
  guard, none of the 102 saved agent answers changes verdict, so the reported
  agent results stand, and the dev set's one judge error is fixed. Blind-v5 is
  unchanged, because the guard never blocks a real refusal. It was added after
  blind-v5 was scored.
- **Cost:** judge-first means one judge call per scored answer (cached on
  disk). Set `abstention: {method: hybrid}` to call the judge only on the
  ~20% of answers the regex is unsure about, or `method: regex` for no API
  calls.

Configure it in the eval YAML:

```yaml
abstention:
  method: judge                                # judge (default) | hybrid | regex
  judge_model: openai/gpt-oss-safeguard-20b    # same endpoint as `judge`, validated model
```

Run the check with
`python -m evalkit.scorer_check --methods regex-v6,judge,hybrid --by-style --errors`
(`--judge-prompt v1` and `--judge-model` reproduce the older judges;
`--pending FILE.csv` writes any unresolved disputes out for a human). CI runs
without an API key, so it checks the regex fallback (an exact match on dev and
blind-v2, at least 0.9 precision and recall on held-out and blind-v1, and
blind-v3 at its measured level: precision ≥ 0.75, recall ≥ 0.45), the guard
(every labeled refusal must pass it), and the judge-first and fallback logic
with a fake judge.

Limits of this check:

- **Blind-v1 is no longer blind.** It was used to build v3, so 28/29 is a
  tuning score.
- **The blind-v1 labels were reviewed.** gpt-oss-120b labeled 4 refusals as
  answers. Each one got a second opinion from a model that didn't write it:
  - 1 is confirmed as a refusal.
  - 1 is now "ambiguous" and left out of the score. It says "Northwind does
    **not** offer pet insurance" as a fact, while also saying there's "no
    mention" of it.
  - 2 are kept as refusals but marked as disputed. Both other models read
    "the policy doesn't list parking" on a yes/no question as an answer ("No").
  - This yes/no rule is now explicit in the blind-v2 labeling prompt.
  - Every decision is recorded in `label_by` in the fixture. No labels have been
    reviewed by a human yet.
- **Known gaps:**
  - The remaining miss is a refusal followed by an unrelated fact containing a
    number ("…the buddy program runs for the first 90 days").
  - A partial answer with no citation and no number ("Yes, but the policy
    doesn't say how often") is scored as a refusal.
  - An answer that only lists what a policy covers, followed by "the policy
    doesn't mention X", could also be scored as a refusal.

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
| Abstention / over-abstention | Does it refuse on unanswerable questions, and only on those? | No (rule-based, checked against 134 labeled answers) |
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
- **Abstention scoring, round 6.** Blind-v5 is now tuning data. Next:
  confirm the judge-first default on a blind-v6 from a model family not used
  yet; check the agent's declared `Status:` line on a live run; and have a
  human label a sample of the 18 rule-resolved disputes, since those are
  exactly the cases where the scorer and the labelers disagree.
- **Repeated runs** (`repeats: 3`) to report variance, not single numbers.

## License

Copyright (c) 2026 Bhamidipati I S H Santosh. **All rights reserved.**
This code may not be used, copied, modified or distributed without prior written
permission. See [LICENSE](LICENSE).
