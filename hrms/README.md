# LC–HRMS — files still to be deposited

**This directory is currently empty of data, and the manuscript's Data Availability statement
promises the files listed below.** They must be added before the statement is accurate.

The manuscript says: *"Colorimetric response data, fold-level prediction tables, LC–HRMS peak
tables and search outputs, DFT input/output files, machine-learning code and analysis scripts
are available at …"*. Everything except the LC–HRMS items is present in this repository. The
LC–HRMS outputs live with the analyst, under `Color sensors\실측분석_결과\`.

## What to add

Per the analysis handover record (`인수인계_LCHRMS_실측분석_v2.md`, §14), these exist:

| File | Contents |
|---|---|
| `all_ions_master_v4.csv` | 656 predicted species, 547 cations searched per file (§8-1) |
| `실측_판정요약.csv` | targeted-search verdict per species |
| `실측_파일별요약.csv` | per-file summary |
| `실측_원물질검출확인.csv` | Δppm, RT and S/N of starting materials in each reaction file |
| `실측_차등후보수.csv` | untargeted differential candidate counts |
| `실측_전조합탐색_결과.csv` | the 18 hits from the 21,758-species enumeration, with per-item verdicts |
| `분석스크립트.tar.gz` | 7 Python scripts (stage1, score, diff, interp, …) |

Suggested layout:

```
hrms/
  peak_tables/     실측_*.csv
  master_list/     all_ions_master_v4.csv
  scripts/         contents of 분석스크립트.tar.gz
  README.md        this file, updated once the above are present
```

## Two additions worth making at the same time

Both were requested by referees and both are reprocessing of files already held — no new
acquisition:

1. **Extracted-ion chromatogram for *m/z* 574.3517**, overlaid with protonated quinine and
   with 556.3411, on a common time axis, plus the retention time and peak width. This is the
   cheapest test of whether the species is a solution-phase associate or an in-source
   proton-bound cluster. The manuscript currently lists the artefact hypothesis as unexcluded.

2. **Mass errors under a phthalate-only recalibration.** The in-file correction presently uses
   the dye and agent [M+H]⁺ ions together with the two phthalate background ions, which makes
   the reported +0.27 ppm for 574.3517 partly circular. Re-deriving the calibration from the
   background ions alone, and reporting the residual errors, would make that figure
   independent.

## Not to be deposited

Raw `.raw` files total roughly 4.5 GB and are unsuitable for a git repository. If a raw
deposit is required, use MassIVE, PRIDE or a Zenodo record and cite the accession here.
