"""Did the answer give up? Three methods, from most to least reliable:

1. declared status - the agent says so itself ("status": "answered" | "partial" |
   "unanswerable"). Only possible for agents you control.
2. regex first pass - free and instant, but it only knows the wordings it has
   seen. It returns "uncertain" instead of guessing when the signals conflict.
3. LLM judge - sees the question as well as the answer, so it can tell whether
   the answer addresses what was asked. Used for the uncertain cases.

    classify(question, answer, judge=judge, status=status)   # hybrid (default)
    -> {"label": "refuse" | "answer" | "partial", "method": "status" | "regex" | "judge"}

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


# ---------- LLM judge ----------
JUDGE_PROMPT = """Decide whether an assistant's reply actually answers the user's question.

Question: {question}
Reply: {answer}

Labels:
- refuse: the reply does not answer the question. It says the information isn't available, isn't covered, or it doesn't know. Background it adds (what it searched, related facts, what the documents do cover) does not count as answering.
- answer: the reply answers the question that was asked, using facts it presents as stated rules. A "no" is an answer only if a rule says so, not merely because nothing was found.
- partial: the question has several parts; the reply answers at least one part and says another part isn't covered.

Reply as JSON only: {{"label": "refuse" | "answer" | "partial"}}"""


def judge_verdict(question: str, answer: str, judge) -> str | None:
    text = judge.complete(JUDGE_PROMPT.format(question=question or "(not given)", answer=answer or ""),
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
def classify(question: str, answer: str, judge=None, status: str | None = None,
             method: str = "hybrid") -> dict:
    """method: 'hybrid' (regex when sure, judge otherwise), 'judge', 'regex'
    (v5, falling back to v4 when uncertain) or 'regex-v4'."""
    if status and str(status).lower() in STATUS_TO_LABEL:
        return {"label": STATUS_TO_LABEL[str(status).lower()], "method": "status"}
    if method == "regex-v4":
        return {"label": "refuse" if abstained(answer) else "answer", "method": "regex-v4"}
    if method == "judge" and judge is not None:
        label = judge_verdict(question, answer, judge)
        if label:
            return {"label": label, "method": "judge"}
    verdict = regex_verdict(answer)
    if verdict != "uncertain":
        return {"label": verdict, "method": "regex"}
    if judge is not None and method in ("hybrid", "judge"):
        label = judge_verdict(question, answer, judge)
        if label:
            return {"label": label, "method": "judge"}
    return {"label": "refuse" if abstained(answer) else "answer", "method": "regex-v4"}


def gave_up(question: str, answer: str, judge=None, status: str | None = None, method: str = "hybrid") -> bool:
    return classify(question, answer, judge=judge, status=status, method=method)["label"] == "refuse"
