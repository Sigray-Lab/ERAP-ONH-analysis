# ERAP ONH FDG-PET Quantification Pipeline

Automated extraction of [18F]FDG-PET uptake metrics from the optic nerve head (ONH) in the **ERAP clinical trial**
(pilot study of rapamycin in early Alzheimer's disease).

**State:** revised after an adversarial technical review (2026-09-06); tag `v2-review-response` (the reviewed state is
tag `v1-as-reviewed`). Response to the review: `REVIEW_RESPONSE.md`.

## Background

The ONH (optic disc) is where retinal ganglion cell axons leave the eye. FDG-PET measures local glucose metabolism.
The anatomical ONH (~1.5-2 mm) is far smaller than the PET resolution (~5 mm FWHM at the ONH location), so the visible
hotspot is dominated by partial-volume effects and mask means would be confounded by delineation size. The pipeline
therefore reports fixed-count hottest-voxel statistics (Top-150), a single-voxel maximum and a small fixed-sphere peak.

## Pipeline

| Script | Role |
|---|---|
| `00_install_injection_corrected_pet.py` | One-off: installs injection-referenced PET images into `RawData/` (archives the frame-start-referenced originals; writes `RawData/pet_manifest.csv`) |
| `extract_onh_metrics.py` | Discovers sessions, validates sidecars (decay reference, units, timing), loads PET and masks, assigns anatomical laterality, computes all metrics, structured QC, run manifest |
| `utils.py` | I/O, validation, SUV/SUVR/TPR/FUR calculation, QC helpers |
| `qc_visualizations.py` | One PNG per session (radiological display, R/L from the affine) |
| `statistical_analysis.py` | Paired tests (t and Wilcoxon), Tables 1-2, ancillary numbers, rapamycin correlation, `README_outputs.md` |
| `sensitivity_analysis.py` | Supplementary Table S1 (dose source, sub-110 follow-up sample times, ±1 mm mask change, IF bridging) |

Run order: `00_…` (once) → `extract_onh_metrics.py` → `statistical_analysis.py` → `sensitivity_analysis.py`.

## Metrics

Five intensity measures (max, 2 mm-radius peak sphere, Top-150 mean / median / p90) x four normalisations. SUV and FUR
are the reported endpoints; SUVR and TPR are computed and available in the outputs.

| | max | peak (2 mm) | top150 mean | top150 median | top150 p90 |
|---|:---:|:---:|:---:|:---:|:---:|
| **SUV** (reported) | x | x | x | x | x |
| **FUR** (reported) | x | x | x | x | x |
| SUVR (cerebellum) | x | x | x | x | x |
| TPR (plasma) | x | x | x | x | x |

| Metric | Formula |
|---|---|
| SUV | C[Bq/mL] x weight[kg] / (dose[MBq] x 1000) |
| SUVR | C / time-weighted cerebellum mean over the scan |
| TPR | (C / 1000) / mean plasma [kBq/mL] over the scan |
| FUR | C / AUC(0 to scan midpoint) of the input function x 60 [min⁻¹] |

**Decay reference.** PET voxels (sidecar-verified), aorta IDIF and cerebellum TAC (from the injection-referenced dynamic
series) are referenced to injection; manual blood samples are taken as injection-referenced per the laboratory protocol
(written laboratory record pending). The pipeline aborts if a PET sidecar declares another reference.

**Laterality.** The mask filenames carry the delineator's display-side label; the CSV column `eye` is the anatomical eye
derived from the mask centroid in world coordinates (`mask_label_in_filename` keeps the file label).

**Top-150.** 150 voxels ≈ two resolution elements at the ONH (FWHM ~5.2 mm → 74 mm³ each). Hottest-N statistics depend
on the delineation (a superset can only add hotter voxels; the size of the effect is empirical). Measured here: a
1-voxel dilation changes the Top-150 mean by 0-4.6 % (median 0.1 %); a 1-voxel erosion leaves 26/52 masks with fewer
than 150 voxels, where the statistic falls back to all remaining voxels (Supplementary Table S1).

**Peak sphere.** 2 mm radius (33 voxels) centred on the max voxel, chosen for the ONH size. It is not the PERCIST SULpeak.

## Quick start

```bash
pip install nibabel numpy pandas scipy matplotlib
cd Scripts/
python extract_onh_metrics.py
python statistical_analysis.py
python sensitivity_analysis.py
```

## Data requirements

Raw data are not part of this repository (patient data). See `RawData_Requirements.md`. The expected layout is
`../RawData/` and `../BlindKey/` relative to this folder.

## Development

Developed with Claude Code (Anthropic); maintained by https://github.com/Sigray-Lab, Department of Clinical
Neuroscience, Karolinska Institutet.

## References

1. Wahl RL, et al. *J Nucl Med*. 2009;50 Suppl 1:122S-150S.
2. Patlak CS, Blasberg RG. *J Cereb Blood Flow Metab*. 1985;5(4):584-90.
