# Data Requirements

Raw imaging and clinical data are **not included** in this repository due to patient privacy regulations. The pipeline expects the following files at sibling-level directories (`../RawData/` and `../BlindKey/` relative to the repo root).

## Directory Overview

| Directory | Contents | Per session | Format |
|-----------|----------|-------------|--------|
| `RawData/sub-*/ses-*/pet/` | PET images and ONH masks | 4 files | NIfTI |
| `RawData/eCRF_data/` | Clinical trial data (weight, dose) | 1 file total | CSV |
| `RawData/Cerebellum_tacs/` | Cerebellum time-activity curves | 1 file | TSV |
| `RawData/BloodPlasma/` | Plasma radioactivity samples | 1 file | TSV |
| `RawData/InputFunctions/` | Combined IDIF + plasma input functions | 1 file | TSV |
| `RawData/json_side_cars_updated/` | Corrected PET timing metadata | 1 file | JSON |
| `BlindKey/` | Session blinding key | 1 file total | CSV |

---

## 1. PET Images and ONH Masks

**Location**: `RawData/sub-{ID}/ses-{code}/pet/`

Each session directory contains four files:

| File | Description |
|------|-------------|
| `sub-{ID}_ses-{code}_chunk-brain_rec-StaticMoCo_trc-18FFDG_pet.nii` | Static FDG-PET image |
| `sub-{ID}_ses-{code}_chunk-brain_rec-StaticMoCo_trc-18FFDG_pet.json` | BIDS JSON sidecar |
| `sub-{ID}_ses-{code}_..._pet_left_ONH_mask.nii.gz` | Left eye ONH mask (binary) |
| `sub-{ID}_ses-{code}_..._pet_right_ONH_mask.nii.gz` | Right eye ONH mask (binary) |

**PET image specifications**:

| Parameter | Value |
|-----------|-------|
| Matrix | 384 x 384 x 249 |
| Voxel size | 1 x 1 x 1 mm (isotropic) |
| Units | Bq/mL |
| Scanner | GE Discovery MI 5 PET/CT |
| Reconstruction | Motion-corrected static (MoCo) |

**Masks**: Binary NIfTI volumes manually delineated on blinded FDG-PET images. Typical volume range: 230-800 voxels.

---

## 2. eCRF Data

**Location**: `RawData/eCRF_data/`

| File | Description |
|------|-------------|
| `K8ERAPKIH22001_DATA_*.csv` | REDCap export (timestamped filename) |

**Columns used by pipeline**: `subject_id`, `weight_kg` (body weight at each visit), injected FDG dose (MBq). The full eCRF contains 500+ fields; the pipeline reads only SUV-relevant parameters.

---

## 3. Cerebellum Time-Activity Curves

**Location**: `RawData/Cerebellum_tacs/`

| File | Description |
|------|-------------|
| `sub-{ID}_ses-{Timepoint}_label-cerebellum_tacs.tsv` | Cerebellum TAC (1 per session) |

**Columns**:

| Column | Description |
|--------|-------------|
| `Frame` | Frame index |
| `ROI` | Region label (`cerebellum`) |
| `Mean(Bq/mL)` | Mean activity in cerebellum ROI |
| `Median(Bq/mL)` | Median activity |
| `Std(Bq/mL)` | Standard deviation |
| `Volume(voxels)` | ROI volume |
| `FrameStart(s)` | Frame start time (seconds post-injection) |
| `FrameDuration(s)` | Frame duration (seconds) |
| `FrameCenter(s)` | Frame center time |

Used for **SUVR normalization** (cerebellum mean during the static scan window).

---

## 4. Blood Plasma Samples

**Location**: `RawData/BloodPlasma/`

| File | Description |
|------|-------------|
| `sub-{ID}_ses-{Timepoint}_recording-manual_blood.tsv` | Manual blood samples (1 per session) |

**Columns**:

| Column | Unit | Description |
|--------|------|-------------|
| `time` | seconds | Time post-injection |
| `whole_blood_radioactivity` | kBq/mL | Whole blood activity |
| `plasma_radioactivity` | kBq/mL | Plasma activity |

Typically 4-5 samples per session drawn during the PET scan window. Used for **TPR calculation** (mean plasma activity during scan).

---

## 5. Input Functions

**Location**: `RawData/InputFunctions/`

| File | Description |
|------|-------------|
| `sub-{ID}_ses-{Timepoint}_desc-IF_tacs.tsv` | Combined input function (1 per session) |

**Columns**:

| Column | Unit | Description |
|--------|------|-------------|
| `Time(s)` | seconds | Time post-injection |
| `ROI` | — | Source: `aorta` (IDIF), `wbl` (whole blood), `plasma` |
| `Radioactivity(Bq/mL)` | Bq/mL | Measured radioactivity |

The input function combines an image-derived input function (IDIF) from the descending aorta (early phase, ~27 time points) with manual plasma samples (late phase, 4-5 samples). Used for **FUR calculation** (AUC from 0 to scan midpoint).

---

## 6. PET JSON Sidecars (Updated)

**Location**: `RawData/json_side_cars_updated/`

| File | Description |
|------|-------------|
| `sub-{ID}_ses-{Timepoint}_trc-18FFDG_rec-StaticMoCo_chunk-1_pet.json` | Corrected BIDS sidecar |

These are updated versions of the original PET JSON sidecars with corrected timing values. The pipeline preferentially loads these over the originals.

**Fields used by pipeline**:

| Field | Unit | Description |
|-------|------|-------------|
| `ScanStart` | seconds | Scan start time post-injection (~1800-2520 s) |
| `FrameDuration` | milliseconds | Scan frame duration (typically 1800000 ms = 30 min) |

---

## 7. Blinding Key

**Location**: `BlindKey/`

| File | Description |
|------|-------------|
| `Blinding_key.csv` | Maps blinded session codes to timepoints |

**Columns**:

| Column | Description |
|--------|-------------|
| `participant_id` | Subject identifier (e.g., `sub-101`) |
| `Session` | Timepoint: `Baseline` or `Followup` |
| `Blind.code` | Randomized 5-character session code |

The pipeline uses this file to map blinded directory names (`ses-{code}`) to their actual timepoints for paired statistical analysis.

---

## Expected Directory Tree

```
ERAP_FDG_ONH_periodontium_analysis/
│
├── RawData/
│   ├── sub-101/
│   │   ├── ses-xxxxx/
│   │   │   └── pet/
│   │   │       ├── sub-101_ses-xxxxx_..._pet.nii
│   │   │       ├── sub-101_ses-xxxxx_..._pet.json
│   │   │       ├── sub-101_ses-xxxxx_..._pet_left_ONH_mask.nii.gz
│   │   │       └── sub-101_ses-xxxxx_..._pet_right_ONH_mask.nii.gz
│   │   └── ses-yyyyy/
│   │       └── pet/
│   │           └── (same structure)
│   ├── sub-102/
│   │   └── ...
│   │
│   ├── eCRF_data/
│   │   └── K8ERAPKIH22001_DATA_*.csv
│   ├── Cerebellum_tacs/
│   │   └── sub-{ID}_ses-{Timepoint}_label-cerebellum_tacs.tsv
│   ├── BloodPlasma/
│   │   └── sub-{ID}_ses-{Timepoint}_recording-manual_blood.tsv
│   ├── InputFunctions/
│   │   └── sub-{ID}_ses-{Timepoint}_desc-IF_tacs.tsv
│   └── json_side_cars_updated/
│       └── sub-{ID}_ses-{Timepoint}_trc-18FFDG_rec-StaticMoCo_chunk-1_pet.json
│
├── BlindKey/
│   └── Blinding_key.csv
│
└── ONH_Analysis/          ← this repository
    ├── Scripts/
    ├── README.md
    └── Data_Requirements.md
```
