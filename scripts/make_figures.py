"""Draw the result figures of the study (4.1 to 4.7) from ``results/``.

    python scripts/make_figures.py --out figures                # all
    python scripts/make_figures.py --out figures --only 4.5 4.7
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "third_party"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from quantumrl_dissection.analysis import figures  # noqa: E402
from quantumrl_dissection.analysis.data import Results  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "figures"))
    ap.add_argument("--only", nargs="+", default=None, help="figure numbers, e.g. 4.5 4.7")
    ap.add_argument("--format", default="pdf", choices=["pdf", "png"])
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R = Results(pathlib.Path(a.results))
    for fid, fn in figures.FIGURES.items():
        if a.only and fid not in a.only:
            continue
        path = out / f"fig_{fid.replace('.', '_')}.{a.format}"
        fn(R, str(path))
        print(f"Figure {fid} -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
