# LC–HRMS

## What is here

`acquisition_manifest.csv` — the complete acquisition inventory: 43 Thermo `.raw` files,
6.2 GB uncompressed, with file name, type, dye, agent, size and CRC32 for each.

| type | n | note |
|---|---|---|
| reaction (dye × agent) | 17 | 16 unique mixtures; `Quinine_A230` was re-acquired after a truncated run |
| dye-only control (dye × ACN) | 6 | Quinine, Rhodamine B, Rhodamine 6G, HBT, 2,4-DNPH, anthracene |
| agent-only control (ACN × agent) | 6 | GB, GA, A-230, A-232, A-234, A-242 |
| solvent blank | 14 | 2 processed (Blank_02, Blank_05) |

The 16 unique reaction mixtures correspond one-to-one with Table S14 of the manuscript.
Instrument metadata read from the file headers: **Thermo Q Exactive Plus – Orbitrap MS**,
Exactive Series instrument #04666L, instrument control 2.11-211101 / firmware 2.11.0.3006,
acquisition method `20min_Full MS_ddMS_PO_130-750.meth`, raw format version 66.

Acquisition parameters, read from the scan headers rather than the method document:

| | |
|---|---|
| polarity / mode | positive, profile (centroided on conversion) |
| MS1 range | *m/z* 130–750 |
| MS2 | data-dependent, Top 5, HCD, fixed first mass *m/z* 50 |
| collision energy | NCE 35, single value — **the project method document specifies stepped 20/35/50; the files do not show stepped activation** |
| isolation window | 1.5 *m/z* (±0.75) |
| charge states | 1–2 |
| run time | 20.00 min |
| injection | 10 µL, undiluted |
| duty cycle | 0.688 s median MS1-to-MS1 |
| scans per file | MS1 1,718–1,794; MS2 8,590–8,967 |
| ion injection time | MS1 median 15.1 ms (0.2–200); MS2 median 37.4 ms (0.1–50) |
| resolution | not written to the scan filter; 70,000 / 17,500 inferred from duty cycle |

Not recoverable from the `.raw` files: LC mobile phase, gradient, flow, column and
temperature; ESI source voltages and gas settings; AGC targets; and the reaction
conditions (dye and agent concentrations, solvent, contact time, temperature).

Detection limits in the manuscript are quoted at S/N = 10, the acceptance criterion of the
search pipeline, rather than at a conventional 3σ. A species present at 3σ would have been
rejected by that criterion and never reported, so the 3σ figure would overstate the power of
this screen by a factor of 10/3.

## What is not here, and why

The `.raw` files themselves (6.2 GB) are unsuitable for a git repository. They should be
deposited in a mass-spectrometry repository that issues an accession:

- **MassIVE** — https://massive.ucsd.edu (free, accepts Thermo `.raw`, no size limit in practice)
- **PRIDE** — https://www.ebi.ac.uk/pride (ProteomeXchange accession)
- **Zenodo** — https://zenodo.org (50 GB per record on request)

Once deposited, replace this paragraph with the accession and update the Data Availability
statement in the manuscript.

The processed peak tables and search outputs from the targeted, untargeted, exhaustive and
±2 Da searches, together with the analysis scripts, are available from the corresponding
authors on request and will be added to `hrms/peak_tables/` and `hrms/scripts/` on deposit.

## Two analyses to run against these files

Both are reprocessing of the files in this manifest — no new acquisition is needed. Both can
be done with ThermoRawFileParser (raw → mzML) plus the existing pyteomics pipeline, or
directly in Xcalibur / FreeStyle.

1. **Extracted-ion chromatogram for *m/z* 574.3517**, from
   `260804_Quinine_A242_FullMS_PO_130-750.raw`, overlaid with protonated quinine
   (*m/z* 325.1911) and with the covalent-ester mass 556.3411, on a common time axis.
   Report retention time and peak width. The species is currently reported as spanning
   148 scans, which at the stated duty cycle is 1–2 minutes — too broad for a
   chromatographic peak and consistent with either a persistent background or an
   in-source cluster. The manuscript lists the in-source-artefact hypothesis as
   unexcluded; this EIC is the cheapest test of it. Compare against
   `260804_Quinine_ACN_FullMS_PO_130-750.raw` (dye-only control).

2. **Mass errors under a phthalate-only recalibration.** The in-file correction currently
   uses the dye and agent [M+H]⁺ ions together with the two phthalate background ions
   (dibutyl phthalate 279.1591, DEHP 391.2843), which makes the reported +0.27 ppm for
   574.3517 partly circular — quinine contributes 325 of those 574 Da and is itself a
   calibrant. Re-deriving from the background ions alone and reporting the residuals would
   make that figure independent.
