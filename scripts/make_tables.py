"""Compute every result table of the study from ``results/`` and, given the PDF of the thesis, compare each with the printed table.

    python scripts/make_tables.py --out tables                         # write tables/*.csv and tables/*.md
    python scripts/make_tables.py --out tables --pdf thesis.pdf        # and report every number that the PDF does not contain

Nothing is adjusted to make a number agree: a difference is reported as it is.
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "third_party"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from quantumrl_dissection.analysis import pdfcheck, tables, tables_extra  # noqa: E402,F401
from quantumrl_dissection.analysis.data import Results  # noqa: E402

EXTRA = {"4.3": [("Table E.1", 3, 3)]}      # the numbers quoted by table 4.3 are also in the text of appendix E.1
LABEL_COLS = {"D.4": 2, "D.5": 3, "D.6": 5, "A.2": 2, "4.6": 2, "E.1": 2, "E.3": 2, "E.4": 2, "4.5": 1, "4.7": 1, "E.2": 1}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "tables"))
    ap.add_argument("--pdf", default=None, help="PDF of the thesis to compare with")
    ap.add_argument("--only", nargs="+", default=None, help="table ids, e.g. D.3 4.6")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R = Results(pathlib.Path(a.results))
    pages = pdfcheck.page_texts(a.pdf) if a.pdf else None
    report = []
    for tid, fn in tables.ALL.items():
        if a.only and tid not in a.only:
            continue
        t = fn(R)
        t.frame().to_csv(out / f"table_{tid}.csv", index=False)
        md = [f"# Table {t.id}: {t.title}", "", "| " + " | ".join(t.header) + " |", "|" + "---|" * len(t.header)]
        md += ["| " + " | ".join(r) + " |" for r in t.rows]
        (out / f"table_{tid}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        line = f"Table {tid}: {len(t.rows)} rows"
        if pages is not None:
            res = pdfcheck.check_table(pages, f"Table {tid}", t.rows, LABEL_COLS.get(tid, 1), extra=EXTRA.get(tid, ()))
            if not res["found"]:
                line += "  | caption not found in the PDF"
            else:
                line += (f"  | page {res['page']}: {res['checked']} numbers checked, {len(res['missing'])} not in the text; "
                         f"{res['rows_on_one_line']} of {res['rows']} rows with all their numbers on one line of the PDF")
                for r_i, c_i, c, tok in res["missing"]:
                    line += f"\n      row {r_i} col {c_i} {c!r}: {tok} not found"
        report.append(line)
        print(line, flush=True)
    (out / "comparison_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
