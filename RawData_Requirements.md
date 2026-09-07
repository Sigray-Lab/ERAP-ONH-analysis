# Data Requirements

Raw imaging and clinical data are **not included** in the repository. The pipeline expects the following files at
`../RawData/` and `../BlindKey/` relative to the repository root (`ONH_Analysis/`).

| Directory | Contents | Per session | Format |
|---|---|---|---|
| `RawData/sub-*/ses-*/pet/` | PET image + sidecar, two ONH masks | 3 NIfTI + 1 JSON (plus ignorable `*ScalarVolume*` files) | NIfTI, JSON |
| `RawData/pet_manifest.csv` | provenance of the installed PET images (source, SHA-256, decay reference) | 1 file total | CSV |
| `RawData/eCRF_data/` | REDCap export | exactly one `K8ERAPKIH22001_DATA_*.csv` | CSV |
| `RawData/Cerebellum_tacs/` | cerebellum TAC over the static frame | 1 file | TSV |
| `RawData/BloodPlasma/` | manual blood samples (+ JSON) | 1 TSV (JSON optional) | TSV, JSON |
| `RawData/InputFunctions/` | aorta IDIF + plasma input function | 1 file | TSV |
| `RawData/json_side_cars_updated/` | corrected sidecars (consistency check only) | 1 file | JSON |
| `BlindKey/Blinding_key.csv` | session blinding key | 1 file total | CSV |

## 1. PET image, sidecar and masks

`RawData/sub-{ID}/ses-{code}/pet/`

| File | Description |
|---|---|
| `sub-{ID}_ses-{code}_chunk-brain_rec-StaticMoCo_trc-18FFDG_pet.nii` | static 30-min FDG-PET, 384x384x249, 1 mm isotropic, Bq/mL, float32 |
| `sub-{ID}_ses-{code}_chunk-brain_rec-StaticMoCo_trc-18FFDG_pet.json` | BIDS sidecar **adjacent to the image**; must have `"DecayCorrection": "INJECTION"`, `"ImageDecayCorrected": true`, `"Units": "Bq/mL"`, `ScanStart` (s post-injection, 600-7200) and `FrameDuration` `[1800000]` (ms, local convention) |
| `…_pet_left_ONH_mask.nii.gz`, `…_pet_right_ONH_mask.nii.gz` | binary masks on the PET grid (same shape and affine). The `left`/`right` in the filename is the delineator's display label; anatomical side is derived by the pipeline |

The images are the injection-referenced BIDS export (`BIDS_20260205/raw/.../rec-StaticMoCo_chunk-1_pet.nii.gz`)
installed under the blinded name by `Scripts/00_install_injection_corrected_pet.py`. Frame-start-referenced originals are
archived in `RawData/_archive_START_corrected_pet/` and are rejected by the pipeline.

Masks: manually delineated on blinded PET, 230-797 voxels in the current data set.

## 2. eCRF data

`RawData/eCRF_data/K8ERAPKIH22001_DATA_*.csv` (REDCap export, 447 columns). Columns used: `subject_id`,
`weight_kg_pet_1`, `injected_mbq_pet_1` (Baseline), `weight_kg_pet_2`, `injected_mbq_pet_2` (Followup). Decimal commas
are accepted. Exactly one export must be present (or set `CONFIG["ecrf_filename"]`).

## 3. Cerebellum TAC

`sub-{ID}_ses-{Timepoint}_label-cerebellum_tacs.tsv` with columns `Frame, ROI, Mean(Bq/mL), Median(Bq/mL), Std(Bq/mL),
Volume(voxels), FrameStart(s), FrameDuration(s), FrameCenter(s)`. Frames must be contiguous, start at `ScanStart` and
cover the static frame (checked). Referenced to injection.

## 4. Blood samples

`sub-{ID}_ses-{Timepoint}_recording-manual_blood.tsv`: `time` (s post-injection), `whole_blood_radioactivity`,
`plasma_radioactivity` (kBq/mL; assumed decay-corrected to injection per the laboratory protocol - written record
pending). 5 samples per session from 16 to 101 min; some precede and some follow the static frame. Empty plasma cells are skipped and flagged. Times must be unique and finite.

## 5. Input functions

`sub-{ID}_ses-{Timepoint}_desc-IF_tacs.tsv`: `Time(s)`, `ROI` (`aorta` = image-derived whole-blood curve, 27 samples
to ~10 min; `wbl`; `plasma`), `Radioactivity(Bq/mL)`. The pipeline uses `aorta` + `plasma`; the plasma rows equal the
blood TSV x 1000. Non-finite or negative values and duplicate times are rejected.

## 6. Updated sidecars

`RawData/json_side_cars_updated/sub-{ID}_ses-{Timepoint}_trc-18FFDG_rec-StaticMoCo_chunk-1_pet.json` — used only to
cross-check `ScanStart`, `FrameDuration` and `DecayCorrection` of the adjacent sidecar; a mismatch aborts the run.

## 7. Blinding key

`BlindKey/Blinding_key.csv`: `participant_id`, `Session` (`Baseline`/`Followup`), `Blind.code` (5-character session
code; the session folder is `ses-{Blind.code}`).

## Expected tree

```
ERAP_FDG_ONH_periodontium_analysis/
├── RawData/
│   ├── pet_manifest.csv
│   ├── sub-101/ses-xxxxx/pet/{pet.nii, pet.json, left mask, right mask}
│   ├── …
│   ├── eCRF_data/K8ERAPKIH22001_DATA_*.csv
│   ├── Cerebellum_tacs/, BloodPlasma/, InputFunctions/, json_side_cars_updated/
│   └── _archive_START_corrected_pet/          (rejected by the pipeline)
├── BlindKey/Blinding_key.csv
└── ONH_Analysis/                               ← this repository
    ├── Scripts/, README.md, RawData_Requirements.md, REVIEW_RESPONSE.md
```
