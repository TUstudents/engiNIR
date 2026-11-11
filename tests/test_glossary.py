"""The glossary covers every term of the terminology contract (terminology contract, Enforcement)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def contract_terms() -> set[str]:
    """The bold term names of the contract's Terms section."""
    text = (ROOT / "contracts" / "terminology_v3.md").read_text(encoding="utf-8")
    section = text[text.index("## Terms"):text.index("## Internal names")]
    return {match.lower() for line in section.splitlines() if not line.startswith(":")
            for match in re.findall(r"\*\*([^*]+)\*\*", line)}


def glossary_terms() -> set[str]:
    """The terms of the ``.. glossary::`` blocks (lines indented by three spaces)."""
    text = (ROOT / "docs" / "glossary.rst").read_text(encoding="utf-8")
    return {line.strip().lower() for line in text.splitlines()
            if line.startswith("   ") and not line.startswith("    ") and line.strip()}


def test_the_glossary_has_an_entry_for_every_term_of_the_contract():
    missing = contract_terms() - glossary_terms()
    assert not missing, f"terms of contracts/terminology_v3.md without a glossary entry: {missing}"
