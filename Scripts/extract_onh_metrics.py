#!/usr/bin/env python3
"""
ONH FDG-PET Metric Extraction

Extracts SUVmax, SUVpeak (2 mm sphere), Top-150 statistics and their SUV / SUVR / TPR / FUR normalisations
from manually delineated optic-nerve-head masks on static [18F]FDG-PET images (ERAP trial).

Revised 2026-09-07 after the adversarial review (see ../Review_adversarial/REVIEW_REPORT.md and
../REVIEW_RESPONSE.md):
  * PET images must be decay-corrected to injection; the adjacent sidecar is validated (F01, F13).
  * The `eye` column is the ANATOMICAL eye derived from the mask centroid in world coordinates; the
    filename label is kept in `mask_label_in_filename` (F02).
  * Volume-change QC is computed in a second pass over all sessions (F07); QC flags are structured (F15).
  * Input discovery is unique-match only; blood/IF data are validated; processed IFs are always regenerated.
  * A run manifest (git commit, package versions, input hashes) is written next to the outputs.

Usage:
    cd ONH_Analysis/Scripts
    python extract_onh_metrics.py

Outputs (all under ONH_Analysis/):
    Outputs/ONH_FDG_metrics.csv, Outputs/run_manifest.json
    DerivedData/session_scaling_factors.csv, DerivedData/input_functions/*.csv
    QC/QC_flags_report.csv, QC/QC_summary_report.txt, QC/SUVpeak_visualizations/<sub>/*.png
    LogNotes/extraction_log_YYYYMMDD_HHMMSS.txt
"""
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

from utils import (  # noqa: E402
    F18_HALF_LIFE_S, InputError, calculate_fur, calculate_input_function_auc, calculate_metrics,
    calculate_plasma_auc, calculate_plasma_total_auc, calculate_suv_scaler, calculate_suvr, calculate_tpr,
    check_updated_sidecar, convert_to_suv, discover_sessions, discover_subjects, find_blood_file,
    find_cerebellum_tac, find_input_function_file, find_mask_file, find_pet_file, find_pet_json,
    generate_qc_flags, get_suv_parameters, get_voxel_dimensions, load_blinding_key, load_blood_data,
    load_cerebellum_tac, load_ecrf_data, load_input_function, load_nifti_with_scaling, load_pet_json,
    mask_physical_eye, save_processed_input_function, sha256_file, validate_mask, volume_change_flags,
)

# ----------------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------------
CONFIG = {
    "ecrf_filename": None,          # None = the single K8ERAPKIH22001_DATA_*.csv in RawData/eCRF_data
    "sphere_radius_mm": 2.0,        # fixed peak sphere centred on the max voxel (study-specific, not PERCIST)
    "top_n_voxels": 150,            # ~2 resolution elements at the ONH (see README)
    "eye_separation_mm": (40.0, 75.0),  # plausible inter-ONH distance; assertion on the two masks of a session
    "asymmetry_warn_pct": 50.0,
    "volume_change_warn_pct": 25.0,
}

INTENSITY_KEYS = ["max", "peak", "top150_mean", "top150_median", "top150_p90"]
INTENSITY_SOURCE = {"max": "intensity_max", "peak": "intensity_peak", "top150_mean": "intensity_top150_mean",
                    "top150_median": "intensity_top150_median", "top150_p90": "intensity_top150_p90"}
COLUMN_NAME = {"max": "max", "peak": "peak_2mm", "top150_mean": "top150_mean",
               "top150_median": "top150_median", "top150_p90": "top150_p90"}


def git_commit(path: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--", "Scripts"], capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def main():
    analysis_dir = script_dir.parent
    project_root = analysis_dir.parent
    rawdata_dir = project_root / "RawData"
    outputs_dir, derived_dir, lognotes_dir, qc_dir = (analysis_dir / d for d in ("Outputs", "DerivedData", "LogNotes", "QC"))
    for d in (outputs_dir, derived_dir, lognotes_dir, qc_dir):
        d.mkdir(parents=True, exist_ok=True)
    input_func_dir = derived_dir / "input_functions"
    input_func_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = lognotes_dir / f"extraction_log_{timestamp}.txt"

    def log(msg: str):
        with open(log_file, "a") as f:
            f.write(f"{datetime.now().strftime('%H:%M:%S')} - {msg}\n")
        print(msg)

    log("=" * 60)
    log("ONH FDG-PET Metric Extraction (SUV, SUVR, TPR, FUR) - revised pipeline")
    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Config: {CONFIG}")
    log("=" * 60)

    blinding_map = load_blinding_key(project_root)
    log(f"Loaded blinding key with {len(blinding_map)} entries")
    ecrf_df = load_ecrf_data(rawdata_dir, CONFIG["ecrf_filename"])
    log(f"Loaded eCRF data ({ecrf_df.attrs['source_path']}) with {len(ecrf_df)} rows")
    subjects = discover_subjects(rawdata_dir)
    log(f"Discovered {len(subjects)} subjects: {', '.join(subjects)}")

    # Pre-load input functions
    log("\nPre-loading input functions for FUR calculation...")
    input_function_cache, missing_input_functions = {}, []
    for subject_id in subjects:
        for timepoint in ("Baseline", "Followup"):
            if_file = find_input_function_file(rawdata_dir, subject_id, timepoint)
            if if_file is None:
                log(f"  {subject_id}/{timepoint}: No input function file found")
                missing_input_functions.append(f"{subject_id}/{timepoint}")
                continue
            if_data = load_input_function(if_file)   # raises on NaN
            input_function_cache[(subject_id, timepoint)] = if_data
            log(f"  {subject_id}/{timepoint}: Loaded {if_data['n_idif_samples']} IDIF + {if_data['n_plasma_samples']} plasma samples")
            for w in if_data["warnings"]:
                log(f"    {w}")
    log(f"Loaded {len(input_function_cache)} input functions")

    results, derived_data, qc_flags = [], [], []
    missing = {"masks": [], "pet": [], "suv_params": [], "cerebellum": [], "blood": []}
    session_flag = lambda subj, tp, cat, sev, desc, value="", threshold="", rec="": qc_flags.append(  # noqa: E731
        {"subject_id": subj, "session": tp, "eye": "both", "mask_label_in_filename": "", "flag_category": cat,
         "severity": sev, "flag_description": desc, "value": value, "threshold": threshold, "recommendation": rec})

    # Persist per-session data for the QC images (loaded once, reused after the loop)
    viz_jobs = []

    for subject_id in subjects:
        subject_dir = rawdata_dir / subject_id
        sessions = discover_sessions(subject_dir)
        log(f"\n{subject_id}: Found {len(sessions)} sessions: {', '.join(sessions)}")

        for session_id in sessions:
            pet_dir = subject_dir / session_id / "pet"
            if not pet_dir.exists():
                log(f"  {session_id}: No pet/ directory found")
                continue
            timepoint = blinding_map.get((subject_id, session_id))
            if timepoint is None:
                log(f"  WARNING: {subject_id}/{session_id} not in blinding key - skipped")
                continue

            pet_file = find_pet_file(pet_dir, subject_id, session_id)
            if pet_file is None:
                log(f"  {session_id} ({timepoint}): PET file not found")
                missing["pet"].append(f"{subject_id}/{session_id}")
                continue
            pet_json_file = find_pet_json(pet_file)
            if pet_json_file is None:
                raise InputError(f"{pet_file}: no adjacent JSON sidecar; cannot establish timing/decay reference")
            pet_timing = load_pet_json(pet_json_file)          # validates decay reference, units, timing
            mism = check_updated_sidecar(pet_timing, rawdata_dir, subject_id, timepoint)
            if mism:
                raise InputError(f"{subject_id}/{session_id}: adjacent sidecar disagrees with json_side_cars_updated: {mism}")

            pet_data, pet_img = load_nifti_with_scaling(pet_file)
            voxel_dims = get_voxel_dimensions(pet_img)
            pet_sha = sha256_file(pet_file)
            log(f"  {session_id} ({timepoint}): PET {pet_data.shape}, voxel {voxel_dims.tolist()} mm, "
                f"decay ref {pet_timing['decay_correction']}, ScanStart {pet_timing['scan_start_s']:.0f} s, "
                f"duration {pet_timing['scan_duration_s']:.0f} s, sha256 {pet_sha[:12]}")
            scan_start = pet_timing["scan_start_s"]
            scan_end = scan_start + pet_timing["scan_duration_s"]
            scan_midpoint_s = scan_start + pet_timing["scan_duration_s"] / 2.0

            # SUV parameters
            suv_params = get_suv_parameters(ecrf_df, subject_id, timepoint)
            weight_kg, injected_mbq = suv_params.get("weight_kg"), suv_params.get("injected_mbq")
            suv_scaler = None
            if weight_kg and injected_mbq:
                suv_scaler = calculate_suv_scaler(weight_kg, injected_mbq)
                log(f"    SUV params: weight={weight_kg} kg, dose={injected_mbq} MBq (eCRF), scaler={suv_scaler:.6e}")
            else:
                log(f"    WARNING: Missing SUV parameters (weight={weight_kg}, dose={injected_mbq})")
                missing["suv_params"].append(f"{subject_id}/{session_id}")
            scanner_dose_mbq = (pet_timing.get("radionuclide_total_dose_bq") or np.nan) / 1e6
            if injected_mbq and np.isfinite(scanner_dose_mbq) and abs(scanner_dose_mbq / injected_mbq - 1) > 0.01:
                msg = f"eCRF dose {injected_mbq} MBq vs scanner sidecar {scanner_dose_mbq:.1f} MBq ({(scanner_dose_mbq / injected_mbq - 1) * 100:+.1f}%)"
                log(f"    INFO: {msg}")
                session_flag(subject_id, timepoint, "dose_source", "INFO", msg, f"{scanner_dose_mbq:.1f}", ">1%",
                             "Confirm assay time / net activity with radiopharmacy (see sensitivity table)")

            # Cerebellum
            cerebellum_data = None
            tac_file = find_cerebellum_tac(rawdata_dir, subject_id, timepoint)
            if tac_file:
                cerebellum_data = load_cerebellum_tac(tac_file, scan_start, pet_timing["scan_duration_s"])
                log(f"    Cerebellum: mean={cerebellum_data['cerebellum_mean_bq_ml']:.2f} Bq/mL over {cerebellum_data['cerebellum_n_frames']} frames")
            else:
                log("    WARNING: Cerebellum TAC not found")
                missing["cerebellum"].append(f"{subject_id}/{session_id}")

            # Blood (state reset per session)
            blood_data, plasma_auc_result = None, None
            blood_tsv, blood_json = find_blood_file(rawdata_dir, subject_id, timepoint)
            if blood_tsv:
                blood_data = load_blood_data(blood_tsv, blood_json)
                plasma_auc_result = calculate_plasma_auc(blood_data, scan_start, scan_end)
                log(f"    Blood: {blood_data['n_samples']} valid samples; plasma AUC {plasma_auc_result['plasma_auc_kbq_s_ml']:.2f} kBq*s/mL, "
                    f"mean {plasma_auc_result['plasma_mean_kbq_ml']:.3f} kBq/mL")
                for w in plasma_auc_result["warnings"]:
                    log(f"      {w}")
                    sev = "INFO" if w.startswith("INFO") else "WARNING"
                    session_flag(subject_id, timepoint, "plasma_sampling" if "sample" in w and "NA" not in w else "plasma_data",
                                 sev, w.split(": ", 1)[-1], "", "", "Review blood sampling record")
            else:
                log("    WARNING: Blood data not found")
                missing["blood"].append(f"{subject_id}/{session_id}")

            # Input function AUC (once per session)
            fur_auc = np.nan
            if_key = (subject_id, timepoint)
            if if_key in input_function_cache:
                auc_result = calculate_input_function_auc(input_function_cache[if_key], scan_midpoint_s)
                fur_auc = auc_result["auc_0_to_midpoint_Bq_s_mL"]
                save_processed_input_function(input_function_cache[if_key], auc_result,
                                              input_func_dir / f"{subject_id}_ses-{timepoint}_if_processed.csv")
                log(f"    IF AUC(0 -> {scan_midpoint_s:.0f} s) = {fur_auc:.4e} Bq*s/mL")
                for w in auc_result["warnings"]:
                    if "ERROR" in w or "WARN" in w:
                        log(f"    FUR: {w}")

            derived_data.append({
                "subject_id": subject_id, "session_blinded": session_id, "session_unblinded": timepoint,
                "injected_MBq": injected_mbq, "body_weight_kg": weight_kg, "SUV_scaler": suv_scaler,
                "scanner_RadionuclideTotalDose_MBq": scanner_dose_mbq,
                "scan_start_s": scan_start, "scan_duration_s": pet_timing["scan_duration_s"],
                "decay_reference": pet_timing["decay_correction"],
                "CER_mean_Bq_mL": cerebellum_data["cerebellum_mean_bq_ml"] if cerebellum_data else np.nan,
                "CER_AUC": cerebellum_data["cerebellum_auc_bq_s_ml"] if cerebellum_data else np.nan,
                "plasma_brain_chunk_AUC": plasma_auc_result["plasma_auc_kbq_s_ml"] if plasma_auc_result else np.nan,
                "plasma_total_session_AUC": calculate_plasma_total_auc(blood_data) if blood_data else np.nan,
                "IF_AUC_0_to_midpoint_Bq_s_mL": fur_auc,
                "pet_sha256": pet_sha,
            })

            # ---- masks: load both, establish anatomical side, then compute ----
            loaded = {}
            for label in ("left", "right"):
                mask_file = find_mask_file(pet_dir, subject_id, session_id, label)
                if mask_file is None:
                    log(f"    file-{label}: Mask not found")
                    missing["masks"].append(f"{subject_id}/{session_id}/file-{label}")
                    continue
                if "OHN" in mask_file.name:
                    log(f"    file-{label}: Note - filename contains 'OHN' typo")
                mask_data, mask_img = load_nifti_with_scaling(mask_file)
                validate_mask(mask_data, mask_img, pet_data, pet_img, f"{subject_id}/{session_id} {mask_file.name}")
                side, centroid = mask_physical_eye(mask_data, mask_img.affine)
                loaded[label] = {"file": mask_file, "data": mask_data, "side": side, "centroid": centroid}
                log(f"    file-{label}: centroid x={centroid[0]:+.1f} mm -> anatomical {side} eye")

            if len(loaded) == 2:
                sides = {v["side"] for v in loaded.values()}
                dist = float(np.linalg.norm(loaded["left"]["centroid"] - loaded["right"]["centroid"]))
                lo, hi = CONFIG["eye_separation_mm"]
                if len(sides) != 2:
                    raise InputError(f"{subject_id}/{session_id}: both masks fall on the same anatomical side ({sides})")
                if not (lo <= dist <= hi):
                    raise InputError(f"{subject_id}/{session_id}: mask centroids {dist:.1f} mm apart, outside {lo}-{hi} mm")

            eye_rows = {}
            for label, m in loaded.items():
                eye = m["side"]
                metrics = calculate_metrics(pet_data, m["data"], voxel_dims, CONFIG["sphere_radius_mm"], CONFIG["top_n_voxels"])
                if "error" in metrics:
                    log(f"    {eye}: ERROR - {metrics['error']}")
                    continue
                inten = {k: metrics[INTENSITY_SOURCE[k]] for k in INTENSITY_KEYS}
                row = {"subject_id": subject_id, "session_blinded": session_id, "session_unblinded": timepoint,
                       "eye": eye, "mask_label_in_filename": label}
                cer_mean = cerebellum_data["cerebellum_mean_bq_ml"] if cerebellum_data else None
                plasma_mean = plasma_auc_result["plasma_mean_kbq_ml"] if plasma_auc_result else np.nan
                # Explicit SUV column names (kept identical to v1 for downstream compatibility)
                row["SUVmax"] = convert_to_suv(inten["max"], suv_scaler) if suv_scaler else np.nan
                row["SUVpeak_2mm"] = convert_to_suv(inten["peak"], suv_scaler) if suv_scaler else np.nan
                row["SUVtop150_mean"] = convert_to_suv(inten["top150_mean"], suv_scaler) if suv_scaler else np.nan
                row["SUVtop150_median"] = convert_to_suv(inten["top150_median"], suv_scaler) if suv_scaler else np.nan
                row["SUVtop150_p90"] = convert_to_suv(inten["top150_p90"], suv_scaler) if suv_scaler else np.nan
                for k in INTENSITY_KEYS:
                    row[f"SUVR_{COLUMN_NAME[k]}"] = calculate_suvr(inten[k], cer_mean) if cer_mean else np.nan
                for k in INTENSITY_KEYS:
                    row[f"TPR_{COLUMN_NAME[k]}"] = calculate_tpr(inten[k], plasma_mean)
                for k in INTENSITY_KEYS:
                    row[f"FUR_{COLUMN_NAME[k]}"] = calculate_fur(inten[k], fur_auc)
                for k in INTENSITY_KEYS:
                    row[f"intensity_{k}_Bq_ml"] = inten[k]
                row.update({
                    "mask_volume_voxels": metrics["mask_volume_voxels"], "mask_volume_mm3": metrics["mask_volume_mm3"],
                    "mask_centroid_x_mm": float(m["centroid"][0]), "mask_centroid_y_mm": float(m["centroid"][1]),
                    "mask_centroid_z_mm": float(m["centroid"][2]),
                    "max_voxel_x": metrics["max_voxel_x"], "max_voxel_y": metrics["max_voxel_y"], "max_voxel_z": metrics["max_voxel_z"],
                    "sphere_voxel_count": metrics["sphere_voxel_count"],
                    "scan_start_s": scan_start, "scan_duration_s": pet_timing["scan_duration_s"], "scan_midpoint_s": scan_midpoint_s,
                    "IF_AUC_0_to_midpoint_Bq_s_mL": fur_auc,
                    "plasma_auc_kBq_s_mL": plasma_auc_result["plasma_auc_kbq_s_ml"] if plasma_auc_result else np.nan,
                    "plasma_mean_kBq_mL": plasma_mean,
                    "plasma_samples_in_window": plasma_auc_result["plasma_samples_in_window"] if plasma_auc_result else np.nan,
                    "decay_reference": pet_timing["decay_correction"], "pet_sha256": pet_sha,
                    "pet_file": str(pet_file.relative_to(project_root)), "pet_json_file": str(pet_json_file.relative_to(project_root)),
                    "mask_file": str(m["file"].relative_to(project_root)),
                    "blood_file": str(blood_tsv.relative_to(project_root)) if blood_tsv else "",
                })
                for fl in generate_qc_flags(metrics, row["SUVmax"]):
                    fl.update({"subject_id": subject_id, "session": timepoint, "eye": eye, "mask_label_in_filename": label})
                    qc_flags.append(fl)
                    log(f"    {eye}: WARN: {fl['flag_description']}")
                results.append(row)
                eye_rows[eye] = row
                log(f"    {eye} (file: {label}): SUVmax={row['SUVmax']:.3f}, SUVpeak={row['SUVpeak_2mm']:.3f}, "
                    f"SUVR_max={row['SUVR_max']:.3f}, TPR_max={row['TPR_max']:.3f}, FUR_max={row['FUR_max']:.4f}")

            viz_jobs.append({"subject_id": subject_id, "timepoint": timepoint, "pet_file": pet_file,
                             "masks": {m["side"]: m["file"] for m in loaded.values()},
                             "rows": {eye: dict(r) for eye, r in eye_rows.items()}})

    df = pd.DataFrame(results).sort_values(["subject_id", "session_unblinded", "eye"]).reset_index(drop=True)
    output_file = outputs_dir / "ONH_FDG_metrics.csv"
    df.to_csv(output_file, index=False)
    derived_df = pd.DataFrame(derived_data).sort_values(["subject_id", "session_unblinded"]).reset_index(drop=True)
    derived_file = derived_dir / "session_scaling_factors.csv"
    derived_df.to_csv(derived_file, index=False)

    # ---- second-pass QC ----
    log("\n--- Volume-change QC (second pass, per subject and anatomical eye) ---")
    vflags = volume_change_flags(df, CONFIG["volume_change_warn_pct"])
    for fl in vflags:
        log(f"  {fl['subject_id']}/{fl['eye']}: {fl['flag_description']}")
    qc_flags.extend(vflags)

    log("\n--- Left/Right asymmetry check (anatomical eyes) ---")
    for (subj, sess), g in df.groupby(["subject_id", "session_blinded"]):
        if set(g["eye"]) == {"left", "right"}:
            lv = float(g.loc[g["eye"] == "left", "SUVmax"].iloc[0]); rv = float(g.loc[g["eye"] == "right", "SUVmax"].iloc[0])
            if np.isfinite(lv) and np.isfinite(rv) and min(lv, rv) > 0:
                asym = abs(lv - rv) / max(lv, rv) * 100
                if asym > CONFIG["asymmetry_warn_pct"]:
                    tp = g["session_unblinded"].iloc[0]
                    log(f"  WARN: {subj}/{tp}: L/R asymmetry {asym:.1f}% (L={lv:.2f}, R={rv:.2f})")
                    session_flag(subj, tp, "asymmetry", "WARNING", f"Left/right SUVmax asymmetry {asym:.1f}% (L={lv:.2f}, R={rv:.2f})",
                                 f"{asym:.1f}%", f">{CONFIG['asymmetry_warn_pct']:.0f}%", "Review bilateral mask placement")

    log("\n--- SUVmax >= SUVpeak validation ---")
    viol = df[(df["SUVmax"] < df["SUVpeak_2mm"]) & df["SUVmax"].notna() & df["SUVpeak_2mm"].notna()]
    log(f"  {'OK: all rows have SUVmax >= SUVpeak' if len(viol) == 0 else f'WARNING: {len(viol)} rows with SUVmax < SUVpeak'}")

    for m in missing["masks"]:
        subj, sess, lab = m.split("/")
        session_flag(subj, blinding_map.get((subj, sess), sess), "missing_mask", "CRITICAL",
                     f"Mask file not found ({lab}) - data not extracted", "NA", "required", "Create mask delineation")

    flags_df = pd.DataFrame(qc_flags, columns=["subject_id", "session", "eye", "mask_label_in_filename", "flag_category",
                                               "flag_description", "severity", "value", "threshold", "recommendation"])
    if len(flags_df):
        flags_df = flags_df.sort_values(["subject_id", "session", "eye", "flag_category"]).reset_index(drop=True)
    flags_df.to_csv(qc_dir / "QC_flags_report.csv", index=False)
    with open(qc_dir / "QC_summary_report.txt", "w") as f:
        f.write(qc_summary_text(df, derived_df, flags_df, timestamp))

    # ---- summary ----
    log("\n" + "=" * 60)
    log("=== Processing Complete ===")
    log(f"Subjects processed: {df['subject_id'].nunique()}")
    log(f"Sessions processed: {len(derived_df)}")
    log(f"Eyes processed: {len(df)} (anatomical left: {(df['eye'] == 'left').sum()}, right: {(df['eye'] == 'right').sum()})")
    for k, v in missing.items():
        log(f"Missing {k}: {len(v)}" + (f" -> {v}" if v else ""))
    log(f"QC flags: {len(flags_df)} ({flags_df['flag_category'].value_counts().to_dict() if len(flags_df) else {}})")
    log(f"Output saved to: {output_file}")
    log(f"Derived data saved to: {derived_file}")

    # ---- QC images ----
    log("\nGenerating QC visualizations...")
    from qc_visualizations import save_session_qc_image
    n_images = 0
    for job in viz_jobs:
        pet_data, pet_img = load_nifti_with_scaling(job["pet_file"])
        masks = {eye: load_nifti_with_scaling(p)[0] for eye, p in job["masks"].items()}
        out = save_session_qc_image(pet_data, masks, job["rows"], pet_img.affine, get_voxel_dimensions(pet_img),
                                    job["subject_id"], job["timepoint"], qc_dir / "SUVpeak_visualizations" / job["subject_id"],
                                    CONFIG["sphere_radius_mm"])
        n_images += 1
        log(f"  Generated: {out.name}")
    log(f"Generated {n_images} QC visualization images")

    # ---- run manifest ----
    import nibabel, scipy, matplotlib
    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(analysis_dir),
        "python": platform.python_version(),
        "packages": {"numpy": np.__version__, "pandas": pd.__version__, "nibabel": nibabel.__version__,
                     "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "config": CONFIG,
        "inputs": {
            "blinding_key_sha256": sha256_file(project_root / "BlindKey" / "Blinding_key.csv"),
            "ecrf_file": ecrf_df.attrs["source_path"], "ecrf_sha256": sha256_file(Path(ecrf_df.attrs["source_path"])),
            "pet_manifest_sha256": sha256_file(rawdata_dir / "pet_manifest.csv") if (rawdata_dir / "pet_manifest.csv").exists() else None,
            "pet_decay_reference": sorted(df["decay_reference"].unique().tolist()),
            "n_pet_files": int(derived_df["pet_sha256"].nunique()),
        },
        "outputs": {"metrics_csv": str(output_file), "n_rows": int(len(df)), "n_subjects": int(df["subject_id"].nunique()),
                    "qc_flags": int(len(flags_df)), "log": str(log_file)},
        "laterality": "eye = anatomical side from mask centroid in world coordinates (x>0 = right); "
                      "mask_label_in_filename = label used by the delineator (display side)",
    }
    with open(outputs_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log(f"Run manifest: {outputs_dir / 'run_manifest.json'}")
    log("=" * 60)
    return df


def qc_summary_text(df: pd.DataFrame, derived_df: pd.DataFrame, flags_df: pd.DataFrame, timestamp: str) -> str:
    L = ["=" * 80, "ERAP ONH FDG-PET EXTRACTION - QUALITY CONTROL SUMMARY REPORT", f"Generated: {timestamp[:8]}", "=" * 80, "",
         "OVERVIEW", "-" * 40, f"Total subjects processed: {df['subject_id'].nunique()}", f"Total sessions: {len(derived_df)}",
         f"Total eye-visits processed: {len(df)}", f"Total QC flags raised: {len(flags_df)}",
         "Eye labels are ANATOMICAL (derived from the mask centroid); the delineator's filename label is in mask_label_in_filename.", ""]
    L += ["=" * 80, "FLAG SUMMARY BY CATEGORY", "=" * 80, ""]
    if len(flags_df):
        for cat, g in flags_df.groupby("flag_category"):
            L.append(f"{cat.upper()} ({len(g)} flags)"); L.append("-" * 40)
            for _, r in g.iterrows():
                L.append(f"  {r['subject_id']}/{r['session']}/{r['eye']}: {r['flag_description']}")
            L.append("")
    else:
        L += ["No flags raised.", ""]
    L += ["=" * 80, "DATA QUALITY METRICS", "=" * 80, ""]
    if len(df):
        for col, name in (("SUVmax", "SUVmax"), ("SUVR_max", "SUVR_max (vs cerebellum)"), ("TPR_max", "TPR_max"), ("FUR_max", "FUR_max (min^-1)")):
            v = df[col].dropna()
            L += [f"{name}: range {v.min():.4g} - {v.max():.4g}, mean {v.mean():.4g}, missing {len(df) - len(v)}"]
        v = df["mask_volume_voxels"]
        L += [f"Mask volume (voxels): range {v.min():.0f} - {v.max():.0f}, mean {v.mean():.1f}, median {v.median():.1f}, SD {v.std(ddof=1):.1f}",
              f"Scan start (s post-injection): {df['scan_start_s'].min():.0f} - {df['scan_start_s'].max():.0f}", ""]
    L += ["=" * 80, "FILES", "=" * 80, "QC_flags_report.csv, QC_summary_report.txt, SUVpeak_visualizations/<sub>/*.png",
          "../Outputs/ONH_FDG_metrics.csv, ../Outputs/run_manifest.json, ../DerivedData/session_scaling_factors.csv", "=" * 80]
    return "\n".join(L)


if __name__ == "__main__":
    main()
