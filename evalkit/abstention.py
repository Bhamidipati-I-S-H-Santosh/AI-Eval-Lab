"""Did the answer give up? Three methods:

1. declared status - the agent says so itself ("status": "answered" | "partial" |
   "unanswerable"). Only possible for agents you control, and it is the agent
   grading itself, so evalkit's metrics don't use it by default: they score with
   the judge and report status_agreement (does the status match the verdict?).
   Set abstention.trust_status: true in the config to let the status decide.
2. LLM judge - sees the question as well as the answer, so it can tell whether
   the answer addresses what was asked. The default when a judge is given.
3. regex - free and instant, but it only knows the wordings it has seen. Used
   when there is no judge or the judge call fails; returns "uncertain" instead
   of guessing when the signals conflict (the 'hybrid' method sends only those
   cases to the judge).

    classify(question, answer, judge=judge, status=status)   # judge-first (default)
    -> {"label": "refuse" | "answer" | "partial", "method": "status" | "judge" | "regex" | "regex-v4"}

Validated judge: gpt-oss-safeguard-20b with JUDGE_PROMPT (v2).

Validate any change against hand labels: python -m evalkit.scorer_check
"""
from __future__ import annotations

import json
import re

LABELS = ("refuse", "answer", "partial")
STATUS_TO_LABEL = {"answered": "answer", "partial": "partial", "unanswerable": "refuse"}

# ---------- regex vocabulary (shared by v4 and v5) ----------
_DOCS = r"(polic(y|ies)|documents?|handbook|guide|overview|sources?|guidelines|search results|information)"
_ADV = r"(specifically |explicitly |actually )?"
_REFUSAL = re.compile("|".join([
    r"\bi (don'?t|do not) know\b",
    r"\b(i'?m|i am) not sure\b",
    r"\b(couldn'?t|could not|can'?t|cannot|didn'?t|did not|unable to|(was|am|'m) not able to|wasn'?t able to)"
    r" (find|locate|see|spot|answer|say|tell|determine|confirm)\b",
    r"\bi (don'?t|do not) see\b",
    r"\b(i'?m|i am) not (seeing|finding)\b",
    r"\bnothing (about|on|regarding)\b",
    r"\b(don'?t|do not) have (enough|any|the) (information|details)\b",
    r"\bno (policy|information|mention|guidance|details?)\b",
    r"\b(isn'?t|is not|aren'?t|are not|not) (covered|mentioned|addressed|specified|stated)\b",
    rf"\b{_DOCS} (don'?t|do not|doesn'?t|does not) {_ADV}(say|mention|cover|specify|contain|address|include|state)\b",
    rf"\bnothing in the {_DOCS}\b",
    rf"\b{_DOCS} (is|are) silent\b",
    rf"\b{_DOCS} provides? no\b",
    r"^\W*unknown\b",
]), re.I)
_CLAUSE = re.compile(r"(?<=[.!?;:])\s+|,?\s+but\s+|\s+[—–-]\s+")
_SUBSTANCE = re.compile(r"\[[^\]]+\]|【[^】]+】|\d")    # a citation or a number
# "The policy only covers annual, sick and parental leave [leave_policy.md]":
# describes what the documents cover, so it is context for a refusal, not an answer.
_SCOPE = re.compile(rf"\b{_DOCS}\b[^.;:]{{0,40}}?\b(only (covers?|includes?|mentions?|lists?|describes?)"
                    rf"|(covers?|includes?|lists?) [^.;:]*,[^.;:]*,)", re.I)
_ASIDE = re.compile(r"\s*\([^()]*\)")


def _answers_something(clause: str) -> bool:
    plain = _ASIDE.sub("", clause)
    return bool(_SUBSTANCE.search(clause)) and not _REFUSAL.search(plain) and not _SCOPE.search(plain)


def abstained(answer: str) -> bool:
    """Regex v4 (kept for compatibility): True if the answer contains a refusal
    and no other clause carries a citation or a number."""
    text = (answer or "").replace("’", "'").strip()
    if not _REFUSAL.search(_ASIDE.sub("", text)):
        return False
    clauses = [c for c in _CLAUSE.split(text) if c and c.strip()]
    return not any(_answers_something(c) for c in clauses)


# ---------- regex v5: a first pass that knows when it isn't sure ----------
# "I searched the benefits documents [benefits.md]", "The closest related item is
# the 3-month probation period": reporting the search, not answering.
_CONTEXT = re.compile(r"\b(i (searched|checked|looked|reviewed|consulted|scanned)|(searched|checked|reviewed) the"
                      r"|closest (related|match|item|thing)|the only .{0,40}\b(i|that i) (found|could find)"
                      r"|related (item|policy|information)|for reference)\b", re.I)
_SENT = re.compile(r"(?<=[.!?;:])\s+|\n+|,?\s+but\s+|\s+[—–-]\s+")
_CITES = re.compile(r"\[[^\]]+\]|【[^】]+】")
_WORDS = re.compile(r"[A-Za-z]{3,}.*\s.*[A-Za-z]{3,}|\d")   # more than a bare citation


def strip_markdown(text: str) -> str:
    """Remove bold/italic/heading/bullet markers: '**...**' otherwise hides the
    full stop from the sentence splitter and merges sentences."""
    t = (text or "").replace("’", "'")
    t = re.sub(r"(\*\*|__|`)", "", t)
    t = re.sub(r"^\s*(#{1,6}\s+|[-*•]\s+|\d+[.)]\s+)", "", t, flags=re.M)
    return t.strip()


def regex_verdict(answer: str) -> str:
    """'refuse', 'answer' or 'uncertain'. The lead sentence decides when a
    refusal and an answer-like sentence appear together."""
    sents = [s.strip() for s in _SENT.split(strip_markdown(answer)) if s and s.strip()]
    if not sents:
        return "uncertain"
    refusal = [bool(_REFUSAL.search(_ASIDE.sub("", s))) for s in sents]
    answering = [bool(_SUBSTANCE.search(s)) and not r and not _CONTEXT.search(s)
                 and not _SCOPE.search(_ASIDE.sub("", s)) and bool(_WORDS.search(_CITES.sub("", s)))
                 for s, r in zip(sents, refusal)]
    if not any(refusal):
        # no refusal wording we know; with nothing answer-like either, don't guess
        return "answer" if any(answering) else "uncertain"
    if not any(answering):
        return "refuse"
    return "uncertain" if refusal[0] else "answer"   # lead sentence answers -> it's an answer/partial


# ---------- regex v6: lead = first sentence that says something ----------
# Blind-v4 showed v5 trusting empty openers ("- No", "Based on the available
# policy documents, **no**") as the lead, and reading "The only wellness benefit
# is $60/month [benefits.md]" as an answer.
_CONTEXT6 = re.compile(_CONTEXT.pattern + r"|\bthe only\b|\bonly (covers?|lists?|includes?|mentions?)\b"
                       r"|\bthe available (policy )?documents are\b|\bnone of (these|which|them)\b"
                       r"|\bclosest coverage\b|\b(covers?|lists?|includes?) the following\b", re.I)
_REFUSAL6 = re.compile(_REFUSAL.pattern + r"|\bsilent\b|\bno (documented|written|specific|dedicated|formal|explicit)\b"
                       r"|\bno provisions?\b|\bnot (explicitly|clearly|specifically) (listed|covered|mentioned)\b"
                       r"|\b(isn'?t|is not|aren'?t|are not|not) (documented|addressed)\b"
                       r"|\b(none|no other) .{0,40}\b(mentions?|references?|address(es)?|contains?)\b"
                       r"|\b(don'?t|do not|doesn'?t|does not) (mention|address|reference)\b"
                       r"|\b(don'?t|do not|doesn'?t|does not) say anything\b", re.I)
# also split "…$40/month for internet—printers are not covered" and "…payroll run, and there is no policy for…"
_SENT6 = re.compile(_SENT.pattern + r"|,\s+and\s+|(?<=\w)[—–](?=\w)")
# "No, the company does not offer X": a flat negative. An answer only if a rule backs it.
_NEGATIVE = re.compile(r"^\W*no\b|\b(does not|doesn'?t|do not|don'?t) (offer|provide|reimburse|cover|have|pay|include)\b", re.I)


def regex_verdict_v6(answer: str) -> str:
    sents = [s.strip() for s in _SENT6.split(strip_markdown(answer)) if s and s.strip()]
    if not sents:
        return "uncertain"
    kinds = []
    for s in sents:
        plain = _ASIDE.sub("", s)
        if _REFUSAL6.search(plain):
            kinds.append("R")
        elif (_SUBSTANCE.search(s) and not _CONTEXT6.search(s) and not _SCOPE.search(plain)
              and _WORDS.search(_CITES.sub("", s))):
            kinds.append("A")
        elif _NEGATIVE.search(plain):
            kinds.append("N")
        else:
            kinds.append("-")
    if "R" not in kinds:
        return "answer" if "A" in kinds else "uncertain"
    if "A" not in kinds:
        return "refuse"
    lead = next(k for k in kinds if k != "-")
    return "answer" if lead == "A" else "uncertain"


# ---------- LLM judge ----------
JUDGE_PROMPT_V1 = """Decide whether an assistant's reply actually answers the user's question.

Question: {question}
Reply: {answer}

Labels:
- refuse: the reply does not answer the question. It says the information isn't available, isn't covered, or it doesn't know. Background it adds (what it searched, related facts, what the documents do cover) does not count as answering.
- answer: the reply answers the question that was asked, using facts it presents as stated rules. A "no" is an answer only if a rule says so, not merely because nothing was found.
- partial: the question has several parts; the reply answers at least one part and says another part isn't covered.

Reply as JSON only: {{"label": "refuse" | "answer" | "partial"}}"""

# v2: the rule both labelers and the v1 judge split on, made explicit with
# worked examples (topics not used in any labeled set).
JUDGE_PROMPT = """You check whether an assistant's reply answers a question about company policy.

Question: {question}
Reply: {answer}

Step 1: find what the reply says about the exact thing the question asks. Ignore everything else.
Step 2: label it.
- refuse: the only basis is that the documents don't mention, cover or address it ("No, the policies don't mention X", "there is no policy on X", "the policies are silent on X", "I couldn't find X"). Still refuse if the reply starts with "No", if it says "the company does not offer X" only because nothing was found, if it suggests asking HR, or if it lists other benefits, related allowances or the documents it searched.
- answer: a rule stated in the documents settles the question. A "no" is an answer only when a rule says no, e.g. "only 5 days can be carried over".
- partial: the question has several parts; a rule settles at least one part and the reply says another part isn't covered.

Examples:
Q: Is there a bike-to-work scheme? R: "No - the policies don't mention a bike scheme. The only commuting benefit is a $30/month transit pass [benefits.md]." -> refuse
Q: Is there a policy on pets in the office? R: "Based on my review, Northwind Labs does not allow or prohibit pets: none of the documents address it. Ask your office manager." -> refuse
Q: Can I expense a taxi to the airport? R: "No - taxis are not reimbursable; use the airport train [expense_policy.md]." -> answer
Q: What's the conference budget, and does it cover spouses? R: "The budget is $900 a year [benefits.md]. The policy doesn't say anything about spouses." -> partial

Reply as JSON only: {{"label": "refuse" | "answer" | "partial"}}"""


def judge_verdict(question: str, answer: str, judge, prompt: str | None = None) -> str | None:
    text = judge.complete((prompt or JUDGE_PROMPT).format(question=question or "(not given)", answer=answer or ""),
                          system="You are a careful grader. Reply with JSON only.")
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\{.*?\}", text, flags=re.S)
    try:
        label = json.loads(m.group(0)).get("label", "") if m else ""
    except ValueError:
        label = ""
    if label not in LABELS:
        found = re.findall(r"\b(refuse|answer|partial)\b", text.lower())
        label = found[-1] if found else None
    return label


# ---------- the entry point ----------
# The judge's "refuse" means "doesn't answer". Abstention needs more: the reply
# must *admit* the gap. A pasted document dump ("According to benefits.md: # ...")
# or a crash message ("I couldn't finish within the step limit") doesn't answer
# either, but it isn't an honest "I don't know". All 124 labeled refusals pass
# this check; none of the 25 dumps and failure messages in saved runs do.
_ADMITS = re.compile(_REFUSAL6.pattern
                     + r"|(\bnot|n'?t) (actually |explicitly |specifically |seem to )?"
                       r"(mention|list|cover|address|include|say|state|specify|find|see)\b"
                     + r"|\bno\b[^.;:]{0,60}\b(mentioned|listed|found|covered|stated|specified)\b"
                     + r"|\bnot (in|part of) (the|any) (polic|doc)|\bunable\b|\bunclear\b|\bcan'?t (confirm|say|tell)\b",
                     re.I)


def admits_gap(answer: str) -> bool:
    """True if the reply says, in any wording, that it can't find / doesn't know."""
    return bool(_ADMITS.search(_ASIDE.sub("", strip_markdown(answer))))


def _ask_judge(question, answer, judge, prompt):
    """None if there is no judge, it fails (rate limit, outage) or its reply can't be parsed."""
    if judge is None:
        return None
    try:
        return judge_verdict(question, answer, judge, prompt=prompt)
    except Exception:
        return None


def _judged(label, answer):
    if label == "refuse" and not admits_gap(answer):
        return {"label": "answer", "method": "judge+guard"}   # "answer" = not an abstention
    return {"label": label, "method": "judge"}


def classify(question: str, answer: str, judge=None, status: str | None = None,
             method: str = "judge", judge_prompt: str | None = None) -> dict:
    """method:
      'judge'     (default) the LLM judge; regex v6 only if there is no judge or it fails.
                  Best on blind-v5: 63/66, P 1.00, R 0.93.
      'hybrid'    regex v6 when it is sure, the judge otherwise: fewer judge calls,
                  slightly lower accuracy (blind-v5: 59/66).
      'regex'     regex v6 only (falling back to v4 when uncertain): free, no API.
      'regex-v5', 'hybrid-v5', 'regex-v4': earlier versions, kept to reproduce old numbers.
    judge_prompt overrides the judge prompt (e.g. JUDGE_PROMPT_V1).
    The result's "method" says what actually decided, so judge failures are visible;
    "judge+guard" means the judge said refuse but the reply never admits a gap
    (a document dump or an error message), so it is not counted as abstaining."""
    if status and str(status).lower() in STATUS_TO_LABEL:
        return {"label": STATUS_TO_LABEL[str(status).lower()], "method": "status"}
    if method == "regex-v4":
        return {"label": "refuse" if abstained(answer) else "answer", "method": "regex-v4"}
    if method == "judge":
        label = _ask_judge(question, answer, judge, judge_prompt)
        if label:
            return _judged(label, answer)
    verdict = (regex_verdict if method in ("regex-v5", "hybrid-v5") else regex_verdict_v6)(answer)
    if verdict != "uncertain":
        return {"label": verdict, "method": "regex"}
    if method in ("hybrid", "hybrid-v5"):
        label = _ask_judge(question, answer, judge, judge_prompt)
        if label:
            return _judged(label, answer)
    return {"label": "refuse" if abstained(answer) else "answer", "method": "regex-v4"}


def gave_up(question: str, answer: str, judge=None, status: str | None = None, method: str = "judge") -> bool:
    return classify(question, answer, judge=judge, status=status, method=method)["label"] == "refuse"
