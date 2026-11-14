"""Recompute the tables and figures of the study from results/.

    python experiments/reproduce.py --out out
    python experiments/reproduce.py --out out --only capacity_ladder capacity

Tables are written as CSV and Markdown, figures as PDF. Set SOURCE_DATE_EPOCH to make the PDF files identical from one run to the next.
"""
import argparse
import pathlib

from quantumrl_dissection.analysis import TABLES, Results, write_tables
from quantumrl_dissection.appendix_tables import TABLES as APPENDIX_TABLES
from quantumrl_dissection.figures import write_figures

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default="out")
    ap.add_argument("--only", nargs="+", help="names of the tables and figures to produce")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    results = Results(pathlib.Path(a.results))
    write_tables(results, out / "tables", {**TABLES, **APPENDIX_TABLES}, a.only)
    write_figures(results, out / "figures", a.only)


if __name__ == "__main__":
    main()
