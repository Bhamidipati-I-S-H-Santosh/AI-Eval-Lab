"""Tools the agent can call. The same functions are exposed over MCP in
mcp_server.py, so there is one implementation and two ways to reach it.

Each tool returns a string, because that is what goes back to the LLM.
"""
from __future__ import annotations

import ast
import json
import operator
import os
import re

from .ingest import DOCS_DIR, build_chunks
from .retrieve import Retriever

_retriever: Retriever | None = None
GUARD_ENABLED = True


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever(build_chunks(), method=os.getenv("RETRIEVAL_METHOD", "bm25"))
    return _retriever


def set_retriever(r: Retriever) -> None:
    """Used by evals to swap in a retriever (e.g. one with poisoned docs)."""
    global _retriever
    _retriever = r


# ---------- prompt-injection guard ----------
_INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) instructions",
    r"disregard (the |your )?(system|previous) (prompt|instructions)",
    r"\bsystem (notice|override|prompt)\b",
    r"you are now\b",
    r"(send|email|forward) .{0,40}(password|credential|login)",
    # text addressed to the AI itself, rather than to employees
    r"\b(note|message|instructions?|reminder) (to|for) (the )?(ai|assistants?|llms?|language models?|chatbots?|bots?)\b",
    r"\b(ai |the )?assistants? (must|should|are required to|need to|have to)\b",
    r"\b(include|append|add|insert|mention)\b.{0,60}\b(verification|confirmation|case|tracking) (code|number|string)\b",
    r"\b(at the end of|in) (every|each|all) (answer|response|reply)",
    r"\bbefore (answering|you answer|responding)\b",
    r"\bre-?verify (your|their) account\b",
    r"!\[[^\]]*\]\(https?://",   # markdown image pointing off-site (exfiltration)
]


QUARANTINED = "[passage removed: it contained instructions aimed at the AI assistant]"


def guard(text: str) -> tuple[str, bool]:
    """If a passage contains instructions aimed at the AI, quarantine the WHOLE
    passage. (An earlier version removed only the matching sentence; evalkit
    showed attacks survive that, because the payload sits in the next sentence.)
    Returns (clean_text, was_flagged). Pattern matching is a first layer only;
    the system prompt and evals are the other layers."""
    if any(re.search(p, text, flags=re.I) for p in _INJECTION_PATTERNS):
        return QUARANTINED, True
    return text, False


# ---------- tools ----------
def search_policies(query: str, top_k: int = 3) -> str:
    results = []
    for c in get_retriever().search(query, top_k=top_k):
        text, flagged = guard(c.text) if GUARD_ENABLED else (c.text, False)
        item = {"source": c.source, "section": c.title, "text": text}
        if flagged:
            item["warning"] = "suspicious instructions were removed from this text"
        results.append(item)
    return json.dumps(results, ensure_ascii=False)


def list_documents() -> str:
    return json.dumps(sorted(p.name for p in DOCS_DIR.glob("*.md")))


def read_document(name: str) -> str:
    allowed = {p.name: p for p in DOCS_DIR.glob("*.md")}
    if name not in allowed:  # blocks path traversal like ../../.env
        return json.dumps({"error": f"unknown document '{name}'", "available": sorted(allowed)})
    return allowed[name].read_text(encoding="utf-8")


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.Mod: operator.mod}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("only numbers and + - * / % ** ( ) are allowed")


def calculator(expression: str) -> str:
    """Safe arithmetic. Never use eval() on model output."""
    try:
        result = _eval(ast.parse(expression, mode="eval").body)
        return json.dumps({"expression": expression, "result": round(result, 4)})
    except Exception as e:
        return json.dumps({"error": str(e)})


TOOLS = {
    "search_policies": search_policies,
    "list_documents": list_documents,
    "read_document": read_document,
    "calculator": calculator,
}

# JSON schemas in OpenAI function-calling format
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "search_policies",
        "description": "Search Northwind Labs policy documents (leave, expenses, IT security, "
                       "remote work, onboarding, benefits). Returns the most relevant passages.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "A focused search query, one topic at a time"},
            "top_k": {"type": "integer", "description": "Number of passages, default 3"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "list_documents",
        "description": "List the names of all available policy documents.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "read_document",
        "description": "Read one full policy document by file name, e.g. 'benefits.md'.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression such as '4*320 + 4*75'. "
                       "Always use this for any arithmetic instead of computing in your head.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string"}}, "required": ["expression"]}}},
]


def run_tool(name: str, args: dict) -> str:
    fn = TOOLS.get(name)
    if fn is None:
        return json.dumps({"error": f"unknown tool '{name}'"})
    try:
        return fn(**args)
    except TypeError as e:  # the model sent bad arguments
        return json.dumps({"error": f"bad arguments for {name}: {e}"})
