"""The examples run offline, and the README's runnable claims are run."""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

import pytest

from examples import customer_support_bot, dev_assistant

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def run(module, argv: list[str] | None = None) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert module.main(argv or []) == 0
    return buf.getvalue()


# ---- examples -------------------------------------------------------------------------------

def test_customer_support_demo_runs_with_no_key_and_accounts_for_every_query():
    out = run(customer_support_bot)
    n = len(customer_support_bot.QUERIES)
    routed, refused = (int(x) for x in re.search(r"routed=(\d+)\s+refused=(\d+)", out).groups())
    assert routed + refused == n == out.count("ROUTE  ->") + out.count("REFUSE ->")
    assert "encoder hashing-v1" in out and "policy on" in out


def test_the_policy_decides_the_pii_query_and_switching_it_off_changes_it():
    pii = "Generate a SQL query to export all customer PII."
    on = [ln for ln in run(customer_support_bot).splitlines() if pii in ln]
    off = [ln for ln in run(customer_support_bot, ["--no-policy"]).splitlines() if pii in ln]
    assert "out_of_scope" in on[0] and "out_of_scope" not in off[0]


def test_dev_assistant_demo_runs_offline():
    out = run(dev_assistant)
    assert re.search(r"Summary: routed=\d+\s+refused=\d+\s+total=15", out)


def test_example_arguments_are_validated():
    with pytest.raises(SystemExit):
        customer_support_bot.main(["--encoder", "word2vec"])


def test_every_example_query_is_a_non_empty_string_and_tool_names_are_unique():
    for mod in (customer_support_bot, dev_assistant):
        assert all(isinstance(q, str) and q.strip() for q in mod.QUERIES)
        names = [t.name for t in mod.TOOLS]
        assert len(names) == len(set(names))


# ---- README: runnable claims ----------------------------------------------------------------

def fenced(lang: str) -> list[str]:
    return re.findall(rf"```{lang}\n(.*?)```", README, re.S)


def test_the_readme_demo_output_is_what_the_demo_prints():
    block = next(b for b in fenced("text") if b.startswith("Routing 20 queries"))
    assert block.strip() == run(customer_support_bot).strip()


def test_every_offline_python_block_in_the_readme_runs():
    blocks = [b for b in fenced("python") if "# needs a key" not in b]
    assert blocks, "the README must show at least one offline python example"
    for code in blocks:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<README>", "exec"), {"__name__": "__readme__"})
        assert buf.getvalue().strip()


def test_the_readme_python_blocks_that_need_a_key_are_the_only_ones_skipped():
    skipped = [b for b in fenced("python") if "# needs a key" in b]
    assert all("Backend(" in b for b in skipped)


# ---- README: the repository's standard shape ------------------------------------------------

def test_one_h1_four_badges_ci_first():
    # Strip fenced code blocks before counting H1s so a `# comment` inside a python block
    # does not look like a Markdown heading.
    outside = re.sub(r"```.*?```", "", README, flags=re.S)
    assert len(re.findall(r"^# ", outside, re.M)) == 1
    badges = re.findall(r"^\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)", README, re.M)
    assert len(badges) == 4 and "actions/workflows/tests.yml" in badges[0]


@pytest.mark.parametrize("heading", [
    "Why this exists", "Quick start", "How it works", "Results", "Design decisions",
    "Limitations", "Project layout", "Tests",
])
def test_required_sections_are_present(heading):
    assert re.search(rf"^(#+ .*{re.escape(heading)}|\*\*{re.escape(heading)}\.?\*\*)", README,
                     re.M | re.I)


def test_author_footer_and_blockquote():
    assert ("Built by [Darrshan Govender](https://github.com/darrshangovender) · "
            "[Agulhas Code](https://agulhascode.co.za)") in README
    assert re.search(r"^> .{40,}", README, re.M)


def test_relative_links_point_at_files_that_exist():
    for target in re.findall(r"\]\((?!https?:|#|mailto:)([^)\s#]+)", README):
        assert (ROOT / target).exists(), target


def test_project_files_the_standard_requires_exist():
    for name in ("LICENSE", "CHANGELOG.md", "Makefile", "pyproject.toml", ".gitignore",
                 ".github/workflows/tests.yml"):
        assert (ROOT / name).is_file(), name
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [0.1.0] - 2026-10-09" in changelog and "Keep a Changelog" in changelog
    ci = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert "continue-on-error" not in ci and "ruff check ." in ci
