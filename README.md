# ERAP ONH FDG-PET Quantification Pipeline

Automated extraction of [18F]FDG-PET uptake metrics from the optic nerve head (ONH) and retina in the **ERAP clinical trial**. 

## Background

The ONH (optic disc) is the anatomical location where retinal ganglion cell axons exit the eye. FDG-PET measures local glucose metabolism, and changes in ONH uptake may reflect treatment effects on retinal health.

**The core challenge** is the mismatch between anatomical ONH size (~1.5–2 mm diameter) and PET scanner resolution (~5 mm FWHM at the ONH location). The visible PET "hotspot" is dominated by partial volume effects, and variable mask sizes would confound simple mean calculations. This pipeline therefore uses **resolution-robust metrics** that are independent of mask volume.

## Pipeline Overview

| Script | Description |
|--------|-------------|
| `extract_onh_metrics.py` | Main pipeline: discovers subjects, loads PET/masks, calculates all metrics, runs QC |
| `utils.py` | Helper functions for file I/O, SUV/SUVR/TPR/FUR calculation, QC flag generation |
| `qc_visualizations.py` | Generates visual QC images with mask contours, peak spheres, and max voxel markers |
| `statistical_analysis.py` | Paired t-tests (Baseline vs Follow-up) with Cohen's dz effect sizes |

## Quantitative Metrics

|               | max | peak (2 mm) | top150 mean | top150 median | top150 p90 |
|---------------|:---:|:-----------:|:-----------:|:-------------:|:----------:|
| **SUV**       |  x  |      x      |      x      |       x       |     x      |
| **FUR**       |  x  |      x      |      x      |       x       |     x      |

**Metric definitions:**

| Metric | Formula | Description |
|--------|---------|-------------|
| SUV | PET × (weight / dose) | Standardized uptake value (body-weight normalized) |
| FUR | tissue / AUC(input function) × 60 | Fractional uptake rate (min⁻¹) |

### Top-150 Rationale

The Top-150 metric extracts the mean, median, and 90th percentile from the 150 highest-intensity voxels within each mask. The number 150 is derived from scanner resolution:

- **Scanner**: GE Discovery MI 5 PET/CT
- **FWHM at ONH** (~75 mm from FOV center): ~5.2 mm
- **1 resolution element**: (4/3)π(5.2/2)³ ≈ 74 mm³ ≈ 74 voxels (1 mm³ isotropic)
- **2 resolution elements** ≈ 148 voxels → rounded to **150**

Since all masks contain ≥ 230 voxels, Top-150 is unbiased by mask size.

## Quick Start

### Prerequisites

```bash
pip install nibabel numpy pandas scipy matplotlib
```

### Running the Pipeline

```bash
cd Scripts/
python extract_onh_metrics.py       # Extracts all metrics
python statistical_analysis.py      # Runs pre-post statistics
```

## Data Requirements

Raw imaging data are **not included** in this repository due to patient privacy regulations. The pipeline expects the following BIDS-like directory structure at the sibling level:

```
../RawData/
├── sub-XXX/
│   └── ses-XXXXX/
│       └── pet/
│           ├── *_pet.nii                   # FDG-PET image (Bq/mL)
│           ├── *_left_ONH_mask.nii.gz      # Left eye mask (binary)
│           └── *_right_ONH_mask.nii.gz     # Right eye mask (binary)
├── eCRF_data/                              # Body weight, injected dose
├── Cerebellum_tacs/                        # Reference region TACs
├── BloodPlasma/                            # Manual plasma samples
├── InputFunctions/                         # IDIF + plasma input functions
└── json_side_cars_updated/                 # Corrected PET timing metadata

../BlindKey/
└── Blinding_key.csv                        # Session blinding key
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Top-150 voxels (not SUVmean) | Mask sizes vary 3.5×; Top-150 is independent of mask volume |
| 150 = 2 resolution elements | Matches ~2× PET FWHM at ONH location for noise robustness |
| 2 mm SUVpeak sphere | PERCIST-recommended fixed-size ROI, centered on max voxel |
| FUR with midpoint AUC | Approximates metabolic rate without kinetic modeling; midpoint = ScanStart + Duration/2 |
| Blinded delineation | Masks drawn on blinded PET images to avoid bias |

## Development

This pipeline was developed using [Claude Code](https://claude.ai/claude-code) (Anthropic) at the [Sigray Lab](https://ki.se/en/research/research-areas-centres-and-networks/research-groups/sigray-lab-pet-methodology), Department of Clinical Neuroscience, Karolinska Institutet.

## References

1. Wahl RL, et al. From RECIST to PERCIST: Evolving Considerations for PET Response Criteria in Solid Tumors. *J Nucl Med*. 2009;50 Suppl 1:122S-150S.
2. Patlak CS, Blasberg RG. Graphical evaluation of blood-to-brain transfer constants from multiple-time uptake data. *J Cereb Blood Flow Metab*. 1985;5(4):584-90.

## License

This project is part of the ERAP clinical trial. Raw imaging data are not included in this repository. 
