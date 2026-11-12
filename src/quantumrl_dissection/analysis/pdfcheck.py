"""Compare the numbers of a generated table with the printed table of a PDF (text layer, via ``pdftotext``).

The check is deliberately simple and strict in one direction: every number of the data columns of a generated table must
appear in the text of the page that carries the caption of the table (or the next one). A number that does not appear is
reported; nothing is adjusted. It is a check of presence, not of position: the tables of the PDF are laid out in columns that a text
extraction does not keep in order.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from typing import Dict, List, Optional, Sequence, Tuple

NUM = re.compile(r"\d+(?:\.\d+)?")


def page_texts(pdf: str) -> List[str]:
    exe = shutil.which("pdftotext")
    if exe is None:
        raise RuntimeError("pdftotext (poppler) is needed to compare with the PDF")
    out = subprocess.run([exe, "-layout", pdf, "-"], capture_output=True, check=True).stdout.decode("utf-8", errors="replace")
    return out.split("\f")


def numbers(text: str) -> List[str]:
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)             # 10,934 -> 10934
    text = text.replace("{,}", "")
    return NUM.findall(text)


def caption_page(pages: Sequence[str], caption: str) -> Optional[int]:
    pat = re.compile(r"^\s*" + re.escape(caption) + r":", re.M)
    for i, p in enumerate(pages):
        if pat.search(p):
            return i
    return None


def check_table(pages: Sequence[str], caption: str, rows: Sequence[Sequence[str]], label_cols: int = 1,
                span: int = 2, extra: Sequence[Tuple[str, int, int]] = ()) -> Dict[str, object]:
    """``extra``: further places of the PDF whose text may carry the numbers, as (caption, pages before, pages after)."""
    i = caption_page(pages, caption)
    if i is None:
        return {"caption": caption, "found": False, "checked": 0, "missing": []}
    sel = list(pages[i:i + span])
    for cap, before, after in extra:
        j = caption_page(pages, cap)
        if j is not None:
            sel += list(pages[max(0, j - before): j + after + 1])
    text = "
".join(sel)
    have = set(numbers(text))
    lines = [set(numbers(ln)) for ln in text.splitlines() if ln.strip()]
    checked, missing, one_line = 0, [], 0
    for r_i, row in enumerate(rows):
        toks = []
        for c_i, c in enumerate(row):
            if c_i < label_cols:
                continue
            for tok in numbers(str(c)):
                toks.append(tok)
                checked += 1
                if tok not in have:
                    missing.append((r_i, c_i, str(c), tok))
        if toks and any(set(toks) <= ln for ln in lines):
            one_line += 1
    return {"caption": caption, "found": True, "page": i + 1, "checked": checked, "missing": missing,
            "rows": len(rows), "rows_on_one_line": one_line}
