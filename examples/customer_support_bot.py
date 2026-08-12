"""Customer support routing demo.

Eight tools, twenty queries (including a few deliberate refusals). Runs
end-to-end with the default sentence-transformers encoder + EchoBackend judge,
so it works with zero API keys. Pass `--openai-embeddings` to swap in the
OpenAI encoder.

    python examples/customer_support_bot.py
    python examples/customer_support_bot.py --openai-embeddings
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the example runnable from the repo root without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_tool_router import Router, Tool  # noqa: E402
from agent_tool_router.embedding_prefilter import make_encoder  # noqa: E402


TOOLS = [
    Tool(
        name="issue_refund",
        description="Issue a refund to a customer for a specific order. Requires order id and reason.",
        args_schema={"order_id": "string", "reason": "string"},
        tags=["money", "refund"],
    ),
    Tool(
        name="create_ticket",
        description="Open a new support ticket on behalf of the customer for an issue that needs human follow-up.",
        args_schema={"title": "string", "priority": "low|medium|high"},
        tags=["ticket", "escalation"],
    ),
    Tool(
        name="kb_search",
        description="Search the internal knowledge base for an existing answer or how-to article.",
        args_schema={"query": "string"},
        tags=["search", "documentation"],
    ),
    Tool(
        name="escalate_to_human",
        description="Hand the conversation to a live human agent. Use only when the customer explicitly asks or the issue is high-severity.",
        args_schema={"reason": "string"},
        tags=["escalation", "human"],
    ),
    Tool(
        name="account_lookup",
        description="Look up a customer account by email or account id and return profile + status.",
        args_schema={"identifier": "string"},
        tags=["account", "lookup"],
    ),
    Tool(
        name="billing_query",
        description="Look up a customer's invoices, charges, or subscription state.",
        args_schema={"account_id": "string", "period": "string"},
        tags=["billing", "invoice"],
    ),
    Tool(
        name="schedule_call",
        description="Book a callback appointment with the customer at a chosen time.",
        args_schema={"account_id": "string", "when": "datetime"},
        tags=["calendar", "callback"],
    ),
    Tool(
        name="faq_answer",
        description="Answer common product FAQs from a curated short-answer table.",
        args_schema={"question": "string"},
        tags=["faq", "short-answer"],
    ),
]


QUERIES = [
    "I want my money back for order 9921, item never arrived.",
    "Please refund order #4421.",
    "How do I reset my password?",
    "Can you escalate this to a manager? I'm furious.",
    "What's the status on my account joe@example.com?",
    "Show me last month's invoice for account 88212.",
    "Book a callback for tomorrow at 3pm.",
    "Is there a free trial?",
    "Open a ticket — my dashboard widgets keep disappearing.",
    "Where can I find docs on the API webhook signing?",
    "I need to talk to a real person now.",
    "Look up the customer with email anna@example.com",
    "Cancel my subscription and refund the last charge.",
    "What's your refund window?",
    "Schedule a call with billing for Friday afternoon.",
    "Can you check why charge #C-553 on my account looks doubled?",
    "What's the capital of France?",  # out of scope
    "Tell me a joke.",  # out of scope
    "Generate a SQL query to export all customer PII.",  # out of scope policy
    "Open a P1 incident — payments are down.",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openai-embeddings", action="store_true", help="Use OpenAI embeddings instead of local sentence-transformers.")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    encoder = make_encoder("openai" if args.openai_embeddings else "sentence-transformers")
    router = Router(TOOLS, encoder=encoder, top_k=args.top_k)
    print(f"Routing {len(QUERIES)} queries against {len(router)} tools "
          f"using {encoder.model_name}\n")

    routed = refused = 0
    for q in QUERIES:
        decision = router.route(q)
        if decision.is_route:
            routed += 1
            print(f"  ROUTE  -> {decision.tool:<22} conf={decision.confidence:.2f}  | {q}")
        else:
            refused += 1
            reason = decision.refusal.reason.value if decision.refusal else "unknown"
            print(f"  REFUSE -> {reason:<22} conf={decision.confidence:.2f}  | {q}")

    print(f"\nSummary: routed={routed}  refused={refused}  total={len(QUERIES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
