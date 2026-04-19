# Colorimetric Sensor Array for CWA Detection and Classification

**A 29-dye dual-illumination colorimetric sensor array with hierarchical machine learning for simultaneous detection and discrimination of 17 chemical warfare agents (CWAs), including four Novichok A-series compounds.**

> Ku Kang, Jin Yoo, Jeongyun Kim, Myeongsik Shin, Soohwan Kim, Min-Kun Kim, Doo-Hee Lee  


---

## Overview

This repository provides all data, analysis scripts, and manuscript source files for the study:

- **Array**: 29 fluorescent/chromogenic dyes imaged under Normal (white LED) and UV (365 nm) illumination → **58-channel ΔE\*ab fingerprint** per sample
- **Analytes**: 17 CWAs across 3 structural series
  - d1: OP nerve agents (DMMP, GA, GB, GD, GF, VX)
  - d2: Blood/blister/choking agents (AC, CG, CK, HD, HN, L, PS)
  - d3: Novichok A-series (A-230, A-232, A-234, A-242)
- **ML pipeline**: One-vs-Rest SVM (RBF, C=10) + Leave-One-Out CV (n=68)
  - Flat 17-class accuracy: **63.2%** (F1 = 0.615)
  - Hierarchical Tier 1 (series): **98.5%** LOO accuracy
- **Interpretability**: SHAP TreeExplainer, RDKit molecular descriptors, functional-group mechanism proposals

---

## Repository Structure

```
cwa-colorimetric-array/
├── data/
│   ├── raw/                        # Per-series raw ΔE pivot tables
│   │   ├── d1.normal_deltaE_pivot.csv
│   │   ├── d1.uv_deltaE_pivot.csv
│   │   ├── d1.rgb_deltaE_data.csv
│   │   ├── d2.normal_deltaE_pivot.csv
│   │   ├── d2.uv_deltaE_pivot.csv
│   │   ├── d2.rgb_deltaE_data.csv
│   │   ├── d3.normal_deltaE_pivot.csv
│   │   ├── d3.uv_deltaE_pivot.csv
│   │   └── d3.rgb_deltaE_data.csv
│   └── processed/
│       └── integrated_17compounds.csv  # Combined 68-sample dataset (all series)
│
├── results/                        # Pre-computed analysis outputs
│   ├── step4_summary.csv           # Per-agent classification summary
│   ├── step4_full_results.json     # Full ML result dict (accuracy, F1, confusion)
│   ├── step4_advanced_results.json # Hierarchical + ablation results
│   ├── step4_cheminformatics.json  # RDKit descriptor + SHAP correlation results
│   ├── selectivity_matrix.csv      # Selectivity Index (SI) matrix (17 agents × 29 dyes)
│   ├── agent_descriptors.csv       # RDKit descriptors for 17 CWAs
│   ├── dye_descriptors.csv         # RDKit descriptors for 29 dyes
│   └── descriptor_dye_correlation.csv  # Agent–dye descriptor correlation table
│

│
├── scripts/
│   ├── step4_figures.py            # [MAIN] Figure generation (Figs 1–8)
│   ├── step4_ml_analysis.py        # Core ML pipeline (OvR SVM, LOO-CV, baselines)
│   ├── step4_advanced_analysis.py  # Hierarchical 2-tier + ablation + SHAP
│   ├── step4_rdkit_analysis.py     # RDKit cheminformatics + descriptor correlation
│   └── gen_SI_confusion.py         # SI confusion matrix figure (FigS1)
│

├── .gitignore
└── README.md
```

---

## Quick Start

### Requirements

```bash
pip install numpy pandas scikit-learn matplotlib seaborn shap rdkit-pypi
```

### Run ML pipeline (core classifier)

```bash
cd scripts/
python step4_ml_analysis.py
# → reads: ../data/processed/integrated_17compounds.csv
# → outputs: ../results/step4_full_results.json, step4_summary.csv
```

```

### Run hierarchical classification + SHAP + ablation

```bash
python step4_advanced_analysis.py
# → outputs: ../results/step4_advanced_results.json
```

### Run RDKit cheminformatics analysis

```bash
python step4_rdkit_analysis.py
# → outputs: ../results/step4_cheminformatics.json, agent_descriptors.csv,
#            dye_descriptors.csv, descriptor_dye_correlation.csv, selectivity_matrix.csv
```

### Regenerate SI confusion matrix (FigS1)

```bash
python gen_SI_confusion.py
# → outputs: ../figures/confusion_matrix_17agents.png + .pdf
```

---

## Key Results

| Metric | Value |
|---|---|
| Analyte panel | 17 CWAs (d1: 6, d2: 7, d3: 4) |
| Total samples | 68 (4 concentrations × 17 agents) |
| Feature dimensions | 58 (29 dyes × Normal + UV) |
| Flat LOO-CV accuracy | **63.2%** (43/68), F1 = 0.615 |
| Hierarchical Tier 1 (series) | **98.5%** (67/68) |
| Hierarchical Tier 2 d1 | 62.5% |
| Hierarchical Tier 2 d2 | 67.9% |
| Hierarchical Tier 2 d3 | 50.0% |
| Bootstrap 95% CI | [0.559, 0.608] |
| ANOVA (9 model variants) | F = 45.30, p < 10⁻¹⁸ |
| Random baseline | 5.9% (1/17) |
| Highest SI pair | Dye11–A-242, SI = 29.99 |
| Most specific dye | L-Glutathione (A-242, SHAP = 0.0226) |

---

## Data Description

### `data/processed/integrated_17compounds.csv`

The primary analysis dataset. Columns:

| Column | Description |
|---|---|
| `agent` | CWA identifier (e.g., GA, GB, A-242) |
| `series` | Series label (d1 / d2 / d3) |
| `concentration` | Concentration in μM (10, 50, 100, 500) |
| `Dye01_N` … `Dye29_N` | ΔE\*ab under Normal illumination (29 channels) |
| `Dye01_U` … `Dye29_U` | ΔE\*ab under UV illumination (29 channels) |

### `results/selectivity_matrix.csv`

Selectivity Index matrix (SI_ij = ΔE_ij / mean_k≠i(ΔE_kj)) at 500 μM, Normal illumination. SI > 5 indicates high specificity; SI < 3 may be susceptible to environmental matrix interference.

---

## Dye Array Composition

29 dyes spanning 12 chemical classes:

| Class | Dyes |
|---|---|
| PAH | Anthracene, Pyrene |
| Xanthene | Fluorescein, Rhodamine B, Rhodamine 6G, Eosin Y |
| Azo | Methyl Orange, Congo Red |
| Triarylmethane | Methyl Blue, Crystal Violet, Cresol Red |
| Thiazine | Toluidine Blue O |
| Oxazine | Nile Blue A |
| Phenazine | Safranin O |
| Sulfonephthalein | Phenol Red, Bromophenol Blue |
| Quinoline | Quinoline Yellow |
| Thiol/Amine nucleophiles | L-Glutathione, Cysteine, 2,5-Diaminobenzenedithiol, 3,4-Diaminobenzamide |
| Biphenol | 4,4′-Dihydroxybiphenyl, Biphenyl-4,4′-diol |
| Others | Acridine Orange, Eriochrome Black T, Nile Red, Quinine |

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{Kang2026cwa,
  author  = {Kang, Ku and Yoo, Jin and Kim, Jeongyun and Shin, Myeongsik
             and Kim, Soohwan and Kim, Min-Kun and Lee, Doo-Hee},
  title   = {Machine learning-guided colorimetric sensor array for simultaneous
             detection and discrimination of 17 chemical warfare agents
             including Novichok A-series},
  journal = {Journal of Hazardous Materials},
  year    = {2026},
  note    = {under review}
}
```

---

## License

This repository is released for academic use. For commercial applications or redistribution, contact the corresponding authors.

---

## Contact

**Doo-Hee Lee** (PI) — Korea University  
**Ku Kang** — [bisu9082@gmail.com](mailto:bisu9082@gmail.com)
