"""Public documentation stays user-facing and internally consistent."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]

FORBIDDEN = re.compile(
    r"\bm0\b|\bm1\b|\bwaitlist\b|\bphase [0-9]+\b|catalog proposal|"
    r"on purpose|curated library|curated catalog",
    re.IGNORECASE,
)


def test_public_docs_do_not_expose_internal_project_language():
    failures = []
    for path in PUBLIC_FILES:
        text = path.read_text(encoding="utf-8")
        for match in FORBIDDEN.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            failures.append(f"{path.relative_to(ROOT)}:{line}: {match.group(0)}")
    assert not failures, "\n".join(failures)


def test_readme_uses_model_aliases_for_rest_quantization():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "documented local extension" not in readme
    assert "id@variant" in readme
    assert "no `variant`" in readme or "There is\nno `variant`" in readme


def test_docs_home_links_to_the_entry_point_guides():
    index = (ROOT / "docs/index.md").read_text(encoding="utf-8")
    for target in (
        "QUICKSTART.md",
        "INSTALLATION.md",
        "MODELS.md",
        "RECIPES.md",
        "TROUBLESHOOTING.md",
        "FAQ.md",
        "GLOSSARY.md",
    ):
        assert target in index
