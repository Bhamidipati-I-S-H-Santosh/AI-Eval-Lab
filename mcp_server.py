"""MCP server: exposes the policy tools to any MCP-capable AI app
(Claude desktop app, Cursor, VS Code, your own agent).

Test it:      npx @modelcontextprotocol/inspector python mcp_server.py
Claude app:   see "Connect to the Claude desktop app" in BUILD_GUIDE.md

It talks over stdio, so never print() to stdout here; it would corrupt the protocol.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `src` imports work from any cwd

try:  # MCP Python SDK 2.x
    from mcp.server.mcpserver import MCPServer as Server  # noqa: E402
except ImportError:  # SDK 1.x used the name FastMCP
    from mcp.server.fastmcp import FastMCP as Server  # noqa: E402

from src import tools  # noqa: E402

mcp = Server("northwind-policies")


@mcp.tool()
def search_policies(query: str, top_k: int = 3) -> str:
    """Search Northwind Labs policy documents (leave, expenses, IT security,
    remote work, onboarding, benefits). Returns the most relevant passages
    with their source file. Search one topic at a time."""
    return tools.search_policies(query, top_k)


@mcp.tool()
def list_documents() -> str:
    """List the names of all available policy documents."""
    return tools.list_documents()


@mcp.tool()
def read_document(name: str) -> str:
    """Read one full policy document by file name, e.g. 'benefits.md'."""
    return tools.read_document(name)


@mcp.tool()
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as '4*320 + 4*75'."""
    return tools.calculator(expression)


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
