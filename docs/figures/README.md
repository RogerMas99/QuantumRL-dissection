# Figures 4.5 and 4.7

Final vector versions of Figures 4.5 (depth sweep) and 4.7 (key contrasts), as drawn by `src/quantumrl_dissection/analysis/figures.py` from `results/`. They are the figures used in the thesis.

```bash
SOURCE_DATE_EPOCH=1767225600 python scripts/make_figures.py --out figures --only 4.5 4.7
```

With `SOURCE_DATE_EPOCH` set the PDF has no run-dependent metadata, so the files regenerate byte for byte (same library versions, `requirements.lock`):

```
3ed38948a34cc34e33e27e57a9d72c1e03910c90f9b62c14b1584312dff0ef07  fig_4_5.pdf
ee42cd68e043111982b34c228e48ca68499ef27678c050748ba32e86a35f0de4  fig_4_7.pdf
```

`/figures/` (the default output of `make_figures.py`) is ignored by git; this directory holds the committed copies.
