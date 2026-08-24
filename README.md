# Colorimetric CWA array — data, code and computational record

Supporting data for:

> **A 29-Dye Colorimetric Array for Chemical-Family Triage of Chemical Warfare Agents:
> LC–HRMS Evidence Against a Covalent Chemodosimeter Mechanism**
> J. Kim, J. Yoo, M. Shin, S. Kim, K. Kang, M.-K. Kim, D.-H. Lee
> Submitted to *Journal of Hazardous Materials*.

---

## Headline results

All values below are the **leak-free** figures reported in the manuscript. Standardization is
fitted inside each cross-validation fold on the training subset only.

| Quantity | Value |
|---|---|
| Chemical-family assignment, leave-one-**sample**-out | **98.5%** (67/68), 95% cluster-bootstrap 95.6–100% |
| Chemical-family assignment, leave-one-**compound**-out | **92.6%** (63/68), 95% cluster-bootstrap 83.8–100% |
| — compounds correct at all four concentrations | 14/17 (82.4%), exact 95% 56.6–96.2% |
| — compounds that fail | A-242 (2/4), VX (2/4), Lewisite (3/4) |
| Majority-family baseline | 41.2% |
| Compound-level (flat 17-class) LOO | **61.8%** (42/68), κ = 0.594, macro-F1 = 0.596 |
| Hierarchical, both tiers correct | 57.4% (39/68) |
| Minimal dye subset reproducing the family result | 4 dyes (Anthracene, Pyrene, Methyl Orange, Bromophenol Blue) |
| Tier-1 permutation null, **compound-block** | 74.1 ± 6.0% (observed 98.5%, p = 0.003) |
| Tier-1 permutation null, sample-level (anti-conservative) | 34.2 ± 7.2% |
| Leave-one-**compound**-out permutation null | 29.9 ± 10.7% (observed 92.6%, p = 0.002, z = 5.9) |

> **Correction notice.** An earlier version of this repository and of the manuscript reported
> a flat 17-class accuracy of **63.2% (43/68)**. That value came from standardizing the full
> feature matrix once *before* cross-validation, which leaks held-out information. The
> leak-free value is **61.8% (42/68)** and every derived quantity — confusion matrix,
> per-compound metrics, ablations, hierarchical totals, dye-subset results — was recomputed.
> Do not use any 63.2% figure from earlier snapshots.

---

## Colour-difference metric

ΔE is **CIEDE2000** throughout, each sample referenced to its own matched 0 µM control.
See `data/raw/NOTE_deltaE_metric.md` — two deposited RGB files carried a CIE76 column and
were regenerated on 2026-08-24. No reported result changed; the analysis pipeline reads the
pivot tables, which were already CIEDE2000.

## Layout

```
data/
  raw/                9 ΔE and RGB tables (d1/d2/d3 × normal/UV/rgb)
  processed/          integrated_17compounds.csv — 85 rows × 58 ΔE channels
results/
  classification/     authoritative outputs backing every number in the paper
  legacy/             superseded step4_* outputs — see warning below
dft/
  inputs/             17 ORCA input files
  outputs/            17 ORCA output files (converged, zero imaginary modes)
  coordinates/        17 optimised geometries (.xyz) + _summary.json
  dft_converged.json  descriptors parsed from the FINAL wavefunctions
structures/
  agents_structures.csv   SMILES, InChI, InChIKey, formula, exact mass for all 17 targets
  chem_corrected.json     RDKit descriptors + Tanimoto on the corrected structures
code/
  analysis/           re-runs every reported statistic
  figures/            regenerates Figures 1, 2, 5, 6, 7, 8
hrms/                 see hrms/README.md — files to be added
```

### ⚠ `results/legacy/` contains superseded values

`step4_*.json` and `step4_summary.csv` are the original analysis outputs, and
`results/legacy/NOTICE_SUPERSEDED.md` repeats this warning in place. They are retained for provenance and are **not** the values in
the paper. Two known defects:

1. **`step4_full_results.json`** carries the pre-correction 63.2% accuracy (global scaling).
2. **`step4_cheminformatics.json`** used incorrect A-series SMILES — an O-*tert*-butyl amidine
   for A-242 (MW 252.27), which contains no guanidine, and O-ethyl / O-isopropyl homologues
   for A-232 / A-234. The correct structures are in `structures/agents_structures.csv` and the
   recomputed descriptors in `structures/chem_corrected.json`. Three descriptor–response
   correlations derived from the wrong structures were withdrawn in the manuscript.

The dye registry inside `step4_cheminformatics.json['dye_info']` **is** correct and is the
authoritative source for the 29 dye identities in Table S1.

---

## Reproducing the numbers

```bash
pip install numpy pandas scikit-learn scipy rdkit

python code/analysis/full.py     # accuracy, error structure, hierarchy, LOCO, per-concentration
python code/analysis/subs.py     # nested dye subsets, per-compound OvR AUC
python code/analysis/chem.py     # RDKit descriptors + Tanimoto, corrected structures
python code/analysis/perm3.py 300 t1    # compound-block permutation, Tier-1
python code/analysis/samp.py            # sample-level permutation, 17-class
```

Figure scripts write PNGs and print their own layout-alignment checks:

| script | figure |
|---|---|
| `fig1.py` | Fig 1 — array fingerprints |
| `fig2.py` | Fig 2 — chemical-family assignment |
| `hrmsfig.py` | Fig 5 — LC–HRMS; panel (d) is a descriptor record, converged values only |
| `fig6.py` | Fig 6 — compound-level performance |
| `fig7panel.py` | Fig 7 — measured readout + workflow |
| `fig7.py` | Fig 8 — robustness checks |

---

## Computational details

B3LYP-D3BJ/def2-TZVP, ORCA 6.1.1, RIJCOSX with def2/J, DEFGRID2, gas phase, no implicit
solvation. 474–640 contracted basis functions. All 17 structures converged with **zero
imaginary frequencies**; energies and mode counts are in `dft/coordinates/_summary.json`.

Descriptors in `dft/dft_converged.json` are parsed from the **final converged** wavefunction.
An earlier parser read the first population block written during the optimisation, i.e. the
input geometry, and gave Mayer P–F bond orders of 1.019–1.101 instead of 1.043–1.086. Those
values are superseded.

Natural population analysis was requested in the input files but the NBO executable was
unavailable at runtime, so no NPA/NBO data exist. Mulliken and Löwdin charges are in the
outputs; both are strongly basis-set dependent at def2-TZVP and nearly invert each other.

**The Mayer P–F bond order does not discriminate.** Converged G-agent values (GD 1.051,
GB 1.054, GF 1.055) fall inside the A-series range (1.043–1.086). Of five descriptors
obtainable from the same wavefunctions, only one ranks with hydrolysate abundance:

| descriptor | Spearman ρ vs hydrolysate abundance (n = 4) |
|---|---|
| Mayer P–F | −1.00 |
| ω (Koopmans) | −0.80 |
| Mayer P–N | **+0.40** |
| q(P) Mulliken | 0.00 |
| q(P) Löwdin | **+0.40** |

The DFT section is a supporting record. No conclusion in the manuscript depends on a DFT
correlation.

---

## Structures — read before use

The A-series structural assignments are **literature hypotheses**, not experimentally
confirmed assignments in the open literature. They follow Mirzayanov and the subsequent
review literature and are consistent with the LC–HRMS exact-mass arithmetic reported in the
paper, but all d3 conclusions are conditional on them. The certificate of analysis for the
A-242 material records a nominal mass near 252 Da whereas the guanidine structure has a
monoisotopic mass of 251.156; this discrepancy is unresolved and is stated as a limitation.

This repository contains structures, optical response data and computed descriptors only. It
contains no synthesis route, purification condition, scale-up procedure, dispersal information
or handling instruction.

---

## Scope of the data

Measurements were made in the dye solvents of Table S1 at 10–500 µM with no control of pH,
ionic strength, temperature or matrix. That range is three to six orders of magnitude above
environmental occurrence. These are not environmental measurements and should not be used as
such.

## Licence and citation

Please cite the manuscript above. Raw instrument files (LC–HRMS `.raw` / mzML) are not in this
repository owing to size; see `hrms/README.md`.
