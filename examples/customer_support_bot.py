"""Customer support routing demo.

Eight tools, twenty queries (including deliberate refusals). Runs end to end OFFLINE by default:
the lexical `HashingEncoder` and the `EchoBackend` stub judge, with an optional rule-based policy
for requests the agent must never route. No API key, no model download.

    python examples/customer_support_bot.py
    python examples/customer_support_bot.py --no-policy
    python examples/customer_support_bot.py --encoder st       # needs sentence-transformers
    python examples/customer_support_bot.py --encoder openai   # needs OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the example runnable from the repo root without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_tool_router import PatternPolicy, Router, Tool  # noqa: E402
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


#: What this deployment forbids the agent to route. Regular expressions, case-insensitive. They only
#: catch the phrasings they name; see agent_tool_router/policy.py. Every gap between words is
#: bounded (`.{0,80}`): with an open-ended `.*` twice in one rule, 1,300 characters of "export all "
#: took 1.3 s to scan and 5,000 never finished.
POLICY_RULES = {
    "bulk-export-or-pii": (
        r"\b(export|dump|download|extract)\b.{0,80}\b(all|every|entire|full)\b.{0,80}"
        r"\b(customers?|users?|accounts?|emails?|table|records?)\b|\bpii\b|\bcard numbers?\b"
    ),
    "credentials": r"\bpasswords? for\b|\bcredentials\b|\bapi keys?\b|someone else'?s account",
    "bulk-delete": r"\b(delete|remove|wipe|purge)\b.{0,80}\b(every|all)\b",
    "regulated-advice": (
        r"\bshould i (invest|put my savings|buy)\b|\bmedication\b|\blegally binding\b"
        r"|\btax(es)? should i\b|\bwhat tax\b"
    ),
    "control-bypass": (
        r"\b(override|bypass|get around|turn off|disable)\b.{0,80}"
        r"\b(limit|check|verification|audit|sign-off|approval)\b"
    ),
}

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=["hashing", "st", "openai"], default="hashing")
    parser.add_argument("--no-policy", action="store_true", help="Do not apply POLICY_RULES.")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)

    encoder = make_encoder(args.encoder)
    policy = None if args.no_policy else PatternPolicy(POLICY_RULES)
    router = Router(TOOLS, encoder=encoder, top_k=args.top_k, policy=policy)
    print(f"Routing {len(QUERIES)} queries against {len(router)} tools "
          f"(encoder {encoder.model_name}, policy {'off' if policy is None else 'on'})\n")

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
