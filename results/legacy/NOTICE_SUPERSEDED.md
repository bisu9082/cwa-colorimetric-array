# ⚠ These files are superseded — do not cite values from them

They are kept for provenance only. Every number in the manuscript comes from
`../classification/` and from the scripts in `../../code/analysis/`.

## Known defects

**`step4_full_results.json`, `step4_summary.csv`**
Flat 17-class accuracy is recorded as **0.6324 (43/68)**. That figure was produced by
standardizing the full 58-channel matrix once *before* cross-validation, which leaks
held-out information into every fold. The leak-free value is **0.6176 (42/68)**.
Everything derived from it is affected: the confusion matrix, per-compound precision /
recall / F1, the ablation deltas, the Tier-2 counts (13/24, 19/28, 7/16 rather than
15/24, 19/28, 8/16) and the hierarchical total (39/68 rather than 42/68).

**`step4_cheminformatics.json` — `agent_descriptors`, `agent_similarity`**
Computed on incorrect A-series structures:

| tag | used here (wrong) | correct (paper, Table S2) |
|---|---|---|
| A-232 | 224.22, O-ethyl amidine | 210.19, O-methyl amidine |
| A-234 | 238.24, O-isopropyl amidine | 224.22, O-ethyl amidine |
| A-242 | 252.27, O-*tert*-butyl amidine (**no guanidine**) | 251.29, P-methyl guanidine |

A-230 is correct. Three descriptor–response correlations derived from these structures
were withdrawn in the manuscript after recomputation (XLogP↔PAH channels ρ = +0.38,
p = 0.14; TPSA↔sulfonephthalein ρ = +0.27, p = 0.29; TPSA↔thiol ρ = +0.07, p = 0.79 —
none distinguishable from zero at n = 17).

Correct structures: `../../structures/agents_structures.csv`
Recomputed descriptors: `../../structures/chem_corrected.json`

## One part of this file IS authoritative

`step4_cheminformatics.json['dye_info']` is the dye registry that generated every reported
result, and it is the source for the 29 dye identities in Table S1 of the paper. An earlier
version of Table S1 listed different compounds at nine positions (Dye3, 13, 17, 19, 23, 24,
25, 27, 28) and different solvents at several others; those entries were in error and were
corrected against this registry.
