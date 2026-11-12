"""The repository carries no trace of the technical identifiers of the original repository.

Scope: code, configurations, scripts, tests, documentation and the table of results. Excluded on purpose: ``docs/design/`` (original documents, copied literally),
``docs/arm_names.csv`` (the translation of the old names), ``scripts/verify_equivalence.py`` (it drives the original code) and the data files of ``results/`` and ``tests/data/``.
"""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
PATTERNS = {
    "fix tag": r"\bFIX-\d+|\bNEW-\d+|\bfix\d|fix_autoreset|\bfix\b",
    "experiment number": r"\bexp\d+[a-z]?\b|exp\d+_",
    "old arm prefix": r"\bpaper_(?!ppo)\w+|__flkw|\bhybrid_fig4|\bhybrid_OR|\bhybrid_DR|\bclassical_OR|\boversized_mlp",
    "absolute path": r"[A-Za-z]:\\\\Users|/content/drive|Mi unidad|tfm_qrl|/home/\w+",
}
SKIP_DIRS = {"design", "data", "__pycache__", ".git", "results", "tables", "figures"}
# verify_equivalence.py runs the original code, whose arm labels, module and parameter names it has to use.
SKIP_FILES = {"arm_names.csv", "test_no_legacy_names.py", "verify_equivalence.py"}
SUFFIXES = {".py", ".md", ".toml", ".txt", ".cff", ".lock", ".csv"}


def _files():
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.suffix not in SUFFIXES or p.name in SKIP_FILES:
            continue
        rel = p.relative_to(ROOT)
        if set(rel.parts) & SKIP_DIRS:
            continue
        if rel.parts[0] == "third_party" and p.name != "VENDORED.md":
            continue                                   # the library itself is unmodified upstream code
        yield p


def test_no_legacy_vocabulary():
    offenders = []
    for p in _files():
        text = p.read_text(encoding="utf-8", errors="replace")
        for name, pat in PATTERNS.items():
            for m in re.finditer(pat, text):
                line = text.count("\n", 0, m.start()) + 1
                offenders.append(f"{p.relative_to(ROOT)}:{line}: [{name}] {m.group(0)}")
    assert not offenders, "\n".join(offenders[:60])
