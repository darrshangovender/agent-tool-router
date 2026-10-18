"""A deterministic pre-check that refuses blocked requests before any model runs.

Similarity scores cannot tell "an allowed topic" from "a forbidden request" (an `account_lookup`
tool is the nearest neighbour of "dump every customer's account record"), and an LLM judge can be
argued round. Rules you can read are the dependable backstop for what must never be routed:

    policy = PatternPolicy({
        "bulk-export": r"\\b(export|dump)\\b.{0,80}\\b(all|every)\\b.{0,80}\\bcustomers?\\b",
        "credentials": r"\\bpassword for\\b",
    })
    router = Router(tools, policy=policy)

A rule only fires on the phrasings it names. A request worded around the rules is not caught: this
reduces risk, it does not remove the need for the authorisation checks on the tools themselves.

Bound every gap between words (`.{0,80}`), never `.*`. Two open-ended gaps in one rule make the
scan take time cubic in the query length: 1,300 characters of "export all " took 1.3 s and 5,000
never finished, so one pasted paragraph could hold a worker. `Router(max_query_chars=...)` caps the
input as a second line of defence.
"""

from __future__ import annotations

import re
from collections.abc import Mapping


class PatternPolicy:
    """`policy(query) -> message | None`, from named case-insensitive regular expressions."""

    def __init__(self, rules: Mapping[str, str]) -> None:
        if not rules:
            raise ValueError("a policy needs at least one rule")
        self._rules: list[tuple[str, re.Pattern[str]]] = []
        for name, pattern in rules.items():
            if not name or not isinstance(name, str):
                raise ValueError("rule names must be non-empty strings")
            try:
                self._rules.append((name, re.compile(pattern, re.IGNORECASE | re.DOTALL)))
            except re.error as e:
                raise ValueError(f"rule {name!r} is not a valid regular expression: {e}") from e

    @property
    def rule_names(self) -> list[str]:
        return [n for n, _ in self._rules]

    def matches(self, query: str) -> list[str]:
        """Names of every rule that fires on `query`."""
        return [n for n, rx in self._rules if rx.search(query)]

    def __call__(self, query: str) -> str | None:
        hit = self.matches(query)
        if not hit:
            return None
        return (f"Blocked by policy ({', '.join(hit)}): "
                "this request is not something the agent may do.")
