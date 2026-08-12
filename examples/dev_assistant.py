"""Developer assistant routing demo.

Ten tools, fifteen queries covering the common ChatOps surface: tests,
deploys, code search, PRs, docs, formatting, dependency lookup, secrets.

    python examples/dev_assistant.py
    python examples/dev_assistant.py --openai-embeddings
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_tool_router import Router, Tool  # noqa: E402
from agent_tool_router.embedding_prefilter import make_encoder  # noqa: E402


TOOLS = [
    Tool(name="run_tests", description="Run the project's pytest suite (or chosen subset) and return results.",
         args_schema={"path": "string?"}),
    Tool(name="deploy", description="Trigger a deployment of the current branch to the named environment.",
         args_schema={"env": "staging|prod", "ref": "string?"}),
    Tool(name="search_code", description="Ripgrep-style search across the repository for a symbol or string.",
         args_schema={"pattern": "string", "glob": "string?"}),
    Tool(name="open_pr", description="Open a GitHub pull request from the current branch with a title and body.",
         args_schema={"title": "string", "body": "string"}),
    Tool(name="search_docs", description="Search internal engineering documentation (architecture, runbooks, ADRs).",
         args_schema={"query": "string"}),
    Tool(name="format_code", description="Run the project's formatter (ruff/black/prettier) on changed files.",
         args_schema={"path": "string?"}),
    Tool(name="lookup_package", description="Look up an external package version, license, and last release date on PyPI/npm.",
         args_schema={"name": "string", "registry": "pypi|npm"}),
    Tool(name="rotate_secret", description="Rotate a named secret in the secret manager. Requires confirmation.",
         args_schema={"name": "string"}),
    Tool(name="show_logs", description="Tail or query application logs from the observability backend.",
         args_schema={"service": "string", "since": "duration"}),
    Tool(name="create_issue", description="File a tracker issue with a title, description, and labels.",
         args_schema={"title": "string", "body": "string", "labels": "list[string]"}),
]


QUERIES = [
    "Run the tests in tests/integration",
    "Deploy main to staging",
    "Where is the function `parse_invoice` defined?",
    "Open a PR titled 'Fix off-by-one in pagination'",
    "What's the on-call runbook for the billing service?",
    "Format the files I just changed",
    "What version of pydantic are we pinned to?",
    "Rotate the GITHUB_TOKEN secret",
    "Show me errors from the orders service in the last 15 minutes",
    "File a bug — checkout button is misaligned on Safari",
    "Push to prod",
    "grep for TODO comments in src/",
    "Look up react-router on npm",
    "What did we decide about retry policy in the ADRs?",
    "Tell me about Picasso.",  # out of scope
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openai-embeddings", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    encoder = make_encoder("openai" if args.openai_embeddings else "sentence-transformers")
    router = Router(TOOLS, encoder=encoder, top_k=args.top_k)
    print(f"Routing {len(QUERIES)} queries against {len(router)} tools using {encoder.model_name}\n")

    routed = refused = 0
    for q in QUERIES:
        d = router.route(q)
        if d.is_route:
            routed += 1
            print(f"  ROUTE  -> {d.tool:<18} conf={d.confidence:.2f}  | {q}")
        else:
            refused += 1
            reason = d.refusal.reason.value if d.refusal else "unknown"
            print(f"  REFUSE -> {reason:<18} conf={d.confidence:.2f}  | {q}")
    print(f"\nSummary: routed={routed}  refused={refused}  total={len(QUERIES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
