# Response to the adversarial technical review of 2026-09-06

Review: `Review_adversarial/REVIEW_REPORT.md` (20 findings). Reviewed state: tag `v1-as-reviewed`. This response: tag
`v2-review-response` (2026-09-07). Independent re-check and decisions: see the revision plan (Section 0 of
`REVISION_PLAN.md` if present). Numbers below refer to the regenerated outputs.

| ID | Finding | Decision | What changed | Where |
|---|---|---|---|---|
| F01 | Static PET referenced to frame start, inputs to injection | **Accepted, S1** | Injection-referenced BIDS images installed into RawData (originals archived, `pet_manifest.csv`); adjacent sidecar validated, pipeline aborts on any other reference; all outputs regenerated | `Scripts/00_install_injection_corrected_pet.py`, `utils.load_pet_json`, `RawData/pet_manifest.csv` |
| F02 | File "left" = anatomical right | **Accepted, S1** | `eye` = anatomical side from the mask centroid (world x); file label kept in `mask_label_in_filename`; opposite-side and 40-75 mm assertions; QC images and all tables relabelled; manuscript laterality swapped | `utils.mask_physical_eye`, `extract_onh_metrics.py`, `qc_visualizations.py` |
| F03 | Blood value/time conflicts | **Partly accepted** | sub-110 Baseline: TSV confirmed correct by the PI (eCRF referenced one hour late) - locked, no sensitivity. sub-110 Followup times: sensitivity row, binder check pending (B2). sub-103 BL: sample 5 is imputed from whole blood and lies outside the analysed window; sample 4 time pending (B3); no numeric effect | `Outputs/sensitivity_table.md`, CLAUDE.md §7 |
| F04 | eCRF vs scanner dose (4 sessions) | **Accepted as open provenance, not a code defect** | eCRF stays the declared source; INFO QC flag per discordant session; scanner-dose sensitivity; manuscript states SUV depends on the source, FUR does not; radiopharmacy check pending (B1) | `QC_flags_report.csv` (dose_source), sensitivity table |
| F05 | Lab decay reference undocumented (S1) | **Pushed back to S2** | Continuity of plasma with the injection-referenced IDIF across 25 sessions shows one reference; documented in RawData_Requirements; blood-JSON statement pending lab confirmation (B5) | `RawData_Requirements.md` §4 |
| F06 | "Independent of mask size" false | **Accepted (wording)** | Claim removed everywhere; monotonicity stated; ±1 mm erosion/dilation sensitivity added. The reviewer's registered-intersection variant is not adopted (resampling) | manuscript, README, CLAUDE.md, `sensitivity_analysis.py` |
| F07 | 7 volume-change flags missing | **Accepted** | Second-pass QC over the finished table; 15 flags | `utils.volume_change_flags` |
| F08 | Final mask version / blinding record | **Open (team)** | Masks used as-is; islands noted; `mask_manifest.csv` and delineator statement pending (B4). SHA-256 of PET recorded per row | CLAUDE.md §7 |
| F09 | PERCIST misattribution | **Accepted** | "Study-specific 2 mm-radius sphere, not PERCIST SULpeak" | manuscript, README |
| F10 | Stale counts / normalisation scope | **Accepted** | Counts generated (`ancillary_numbers.json`); four normalisations computed, two reported; stale report and docx retired; CLAUDE.md rewritten | `Outputs/ancillary_numbers.json`, `old/` |
| F11 | Correlation code uncommitted | **Accepted** | Committed as used (tag v1-as-reviewed); rapa file from project RawData copy; duplicates must agree; exclusions written to file | `statistical_analysis.py` |
| F12 | Exclusion rationale | **Accepted** | Dose-period rationale in code and `rapa_correlation_exclusions.csv` | `statistical_analysis.py` |
| F13 | Timing/unit guessing | **Accepted with a change** | ScanStart required (600-7200 s), FrameDuration asserted ~1 800 000 ms (local convention kept), units asserted | `utils.load_pet_json` |
| F14 | First-match globs, timepoint fall-through | **Accepted** | Unique-match discovery, enumerated timepoints | `utils._unique`, `TIMEPOINTS` |
| F15 | `import re` UnboundLocalError | **Accepted** | Structured flags; no regex parsing | `extract_onh_metrics.py` |
| F16 | Unsorted blood, stale state | **Accepted** | Sorted/validated samples; per-session reset | `utils.load_blood_data`, `calculate_plasma_auc` |
| F17 | Missing geometry/TAC guards | **Accepted** | Affine/binary/finite checks; TAC coverage check; non-finite IF raises | `utils.validate_mask`, `load_cerebellum_tac`, `load_input_function` |
| F18 | IF cache | **Accepted** | Always regenerated with endpoints and AUC header | `utils.save_processed_input_function` |
| F19 | Extrapolation, np.trapz, hard-coded eCRF, silent skips | **Accepted** | `np.trapezoid`; eCRF via config; skipped tests logged; duplicate concentrations must agree | `utils.py`, `statistical_analysis.py` |
| F20 | Unused QC function label | **Accepted** | Function removed | `qc_visualizations.py` |

## Effect on the results

| Endpoint (bilateral) | v1 (reviewed) | v2 (corrected) |
|---|---|---|
| SUV top150 mean p / dz | 0.036 / 0.66 | 0.042 / 0.63 |
| SUV top150 median p / dz | 0.035 / 0.66 | 0.040 / 0.64 |
| FUR top150 mean p / dz | 0.042 / 0.63 | 0.036 / 0.66 |
| FUR top150 median p / dz | 0.041 / 0.63 | 0.036 / 0.66 |

Unilateral significance is in the anatomical **right** eye (v1 reported "left", which was the file label). Absolute SUV
values are 21-30 % higher than in v1. Sensitivities: `Outputs/sensitivity_table.md`.

## Verification performed

* 52 rows; anatomical-eye mapping identical to the reviewer's (`metrics_recomputed.csv`); every metric matches the
  reviewer's decay-corrected recomputation to ≤ 1.5e-7 relative; mask volumes identical.
* Paired statistics equal the reviewer's `confirmed_joint` version.
* Pipeline aborts on an archived frame-start-referenced sidecar; two consecutive runs give byte-identical CSVs.
* 15 volume-change flags (reviewer's expected count).
