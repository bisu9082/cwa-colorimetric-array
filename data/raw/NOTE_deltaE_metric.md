# ΔE metric in this directory

All `DeltaE` columns are **CIEDE2000** (Luo, Cui & Rigg, *Color Res. Appl.* 26 (2001) 340–350),
computed for each sample against its own matched 0 µM control in the same illumination
condition, dye and agent group. Parametric factors k_L = k_C = k_H = 1.

Two files were regenerated on 2026-08-24 to enforce this consistently:

- `d1.rgb_deltaE_data.csv` — 1,375 of 1,740 rows previously carried the Euclidean CIE76
  distance ΔE*ab instead of CIEDE2000. CIE76 runs about 1.4× larger than CIEDE2000 on this
  data, so the affected column was not comparable with the d2 and d3 files.
- `d3.rgb_deltaE_data.csv` — 1 row corrected.

`d2.rgb_deltaE_data.csv` required no change.

**This did not affect any reported result.** The analysis pipeline reads the pivot tables
(`*_deltaE_pivot.csv` → `data/processed/integrated_17compounds.csv`), which were already
CIEDE2000 throughout: they reproduce the CIEDE2000 values of the d2 and d3 RGB files to
machine precision, and for d1 they agree in form (Pearson r = 0.98 against CIEDE2000 versus
0.95 against CIE76) with the residual attributable to the pivots averaging five sampling
points per vial where these RGB files record one. The inconsistency was confined to the
`DeltaE` convenience column of the deposited RGB file and has been removed.

## Sampling structure

Each dye-agent-concentration condition was measured three times, and five regions of
interest were sampled within each measurement, so every value in the pivot tables is the
mean of fifteen readings. The RGB files in this directory record one sampling point and
therefore do not by themselves regenerate the pivot tables. The individual measurement
values were not retained in the working dataset, so no repeatability estimate can be
computed from what is deposited here.
