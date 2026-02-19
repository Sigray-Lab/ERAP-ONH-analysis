#!/usr/bin/env python3
"""
ONH FDG-PET Metric Extraction Script

This script extracts SUVmax, SUVpeak (2mm sphere), SUVR, and TPR metrics from
manually delineated optic nerve head (ONH) masks on FDG-PET images.

Part of the ERAP retinal rapamycin trial analysis.

Usage:
    cd ONH_Analysis/Scripts
    python extract_onh_metrics.py

Output:
    ONH_Analysis/Outputs/ONH_FDG_metrics.csv
    ONH_Analysis/DerivedData/session_scaling_factors.csv

Directory structure (relative paths):
    Script location: ONH_Analysis/Scripts/extract_onh_metrics.py
    Analysis dir:    ONH_Analysis/  (script_dir.parent)
    Project root:    ../            (script_dir.parent.parent)
    RawData:         ../RawData/    (project_root / "RawData")
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Add Scripts directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from utils import (
    load_blinding_key,
    load_ecrf_data,
    get_suv_parameters,
    calculate_suv_scaler,
    convert_to_suv,
    discover_subjects,
    discover_sessions,
    find_pet_file,
    find_pet_json,
    load_pet_json,
    find_mask_file,
    load_nifti_with_scaling,
    get_voxel_dimensions,
    calculate_metrics,
    find_cerebellum_tac,
    load_cerebellum_tac,
    calculate_suvr,
    find_blood_file,
    load_blood_data,
    calculate_plasma_auc,
    calculate_plasma_total_auc,
    calculate_tpr,
    generate_qc_flags,
    # FUR-related functions
    find_input_function_file,
    load_input_function,
    calculate_input_function_auc,
    save_processed_input_function,
    calculate_fur
)


def main():
    """Main extraction pipeline."""

    # Define paths using relative structure
    # Script is at: ONH_Analysis/Scripts/extract_onh_metrics.py
    script_dir = Path(__file__).parent          # ONH_Analysis/Scripts/
    analysis_dir = script_dir.parent            # ONH_Analysis/
    project_root = analysis_dir.parent          # ERAP_FDG_ONH_periodontium_analysis/

    # Shared data (at project root level)
    rawdata_dir = project_root / "RawData"

    # Analysis-specific outputs (within ONH_Analysis/)
    outputs_dir = analysis_dir / "Outputs"
    derived_dir = analysis_dir / "DerivedData"
    lognotes_dir = analysis_dir / "LogNotes"

    # Create output directories if needed
    outputs_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)
    lognotes_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = lognotes_dir / f"extraction_log_{timestamp}.txt"

    def log(msg: str, also_print: bool = True):
        """Log message to file and optionally print."""
        with open(log_file, 'a') as f:
            f.write(f"{datetime.now().strftime('%H:%M:%S')} - {msg}\n")
        if also_print:
            print(msg)

    # Create input function output directory
    input_func_dir = derived_dir / "input_functions"
    input_func_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("ONH FDG-PET Metric Extraction (SUV, SUVR, TPR, FUR)")
    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)

    # Load blinding key
    try:
        blinding_map = load_blinding_key(project_root)
        log(f"Loaded blinding key with {len(blinding_map)} entries")
    except FileNotFoundError as e:
        log(f"ERROR: {e}")
        sys.exit(1)

    # Load eCRF data for SUV calculation
    try:
        ecrf_df = load_ecrf_data(rawdata_dir)
        log(f"Loaded eCRF data with {len(ecrf_df)} subjects")
    except FileNotFoundError as e:
        log(f"ERROR: {e}")
        sys.exit(1)

    # Discover subjects
    subjects = discover_subjects(rawdata_dir)
    log(f"Discovered {len(subjects)} subjects: {', '.join(subjects)}")

    # Pre-load input functions for FUR calculation
    log("\nPre-loading input functions for FUR calculation...")
    input_function_cache = {}  # (subject_id, timepoint) -> if_data
    missing_input_functions = []

    for subject_id in subjects:
        for timepoint in ['Baseline', 'Followup']:
            if_file = find_input_function_file(rawdata_dir, subject_id, timepoint)
            if if_file is None:
                log(f"  {subject_id}/{timepoint}: No input function file found")
                missing_input_functions.append(f"{subject_id}/{timepoint}")
                continue

            try:
                if_data = load_input_function(if_file)
                input_function_cache[(subject_id, timepoint)] = if_data
                log(f"  {subject_id}/{timepoint}: Loaded {if_data['n_idif_samples']} IDIF + {if_data['n_plasma_samples']} plasma samples")
                for w in if_data.get('warnings', []):
                    log(f"    {w}")
            except Exception as e:
                log(f"  {subject_id}/{timepoint}: ERROR loading input function: {e}")
                missing_input_functions.append(f"{subject_id}/{timepoint}")

    log(f"Loaded {len(input_function_cache)} input functions")

    # Track results and issues
    results = []
    derived_data = []  # For DerivedData CSV
    missing_masks = []
    missing_pet = []
    missing_suv_params = []
    missing_cerebellum = []
    missing_blood = []
    all_warnings = []

    # Store baseline metrics for comparison
    baseline_cache = {}  # (subject_id, eye) -> metrics dict

    # Process each subject
    for subject_id in subjects:
        subject_dir = rawdata_dir / subject_id
        sessions = discover_sessions(subject_dir)
        log(f"\n{subject_id}: Found {len(sessions)} sessions: {', '.join(sessions)}")

        for session_id in sessions:
            session_dir = subject_dir / session_id
            pet_dir = session_dir / "pet"

            if not pet_dir.exists():
                log(f"  {session_id}: No pet/ directory found")
                continue

            # Find PET file
            pet_file = find_pet_file(pet_dir, subject_id, session_id)
            if pet_file is None:
                log(f"  {session_id}: PET file not found")
                missing_pet.append(f"{subject_id}/{session_id}")
                continue

            # Get unblinded session name
            key = (subject_id, session_id)
            timepoint = blinding_map.get(key, "Unknown")
            if timepoint == "Unknown":
                log(f"  WARNING: {subject_id}/{session_id} not found in blinding key")
                continue

            # Load PET image
            try:
                pet_data, pet_img = load_nifti_with_scaling(pet_file)
                voxel_dims = get_voxel_dimensions(pet_img)
                log(f"  {session_id} ({timepoint}): Loaded PET image, shape={pet_data.shape}, voxel_dims={voxel_dims}")
            except Exception as e:
                log(f"  {session_id}: ERROR loading PET: {e}")
                continue

            # Load PET JSON for timing info (needed for TPR)
            # Pass timepoint and rawdata_dir to enable lookup in updated JSON folder
            pet_json_file = find_pet_json(pet_dir, subject_id, session_id,
                                          timepoint=timepoint, rawdata_dir=rawdata_dir)
            pet_timing = None
            if pet_json_file:
                try:
                    pet_timing = load_pet_json(pet_json_file)
                    log(f"    PET timing: scan_start={pet_timing['scan_start_s']}s, duration={pet_timing['scan_duration_s']}s")
                    # Log warning if ScanStart was defaulted
                    if 'scan_start_warning' in pet_timing:
                        log(f"    WARNING: {pet_timing['scan_start_warning']}")
                        all_warnings.append(f"{subject_id}/{session_id}: WARN: {pet_timing['scan_start_warning']}")
                except Exception as e:
                    log(f"    WARNING: Could not load PET JSON: {e}")

            # Get SUV parameters from eCRF
            suv_params = get_suv_parameters(ecrf_df, subject_id, timepoint)
            weight_kg = suv_params.get('weight_kg')
            injected_mbq = suv_params.get('injected_mbq')

            suv_scaler = None
            if weight_kg and injected_mbq:
                suv_scaler = calculate_suv_scaler(weight_kg, injected_mbq)
                log(f"    SUV params: weight={weight_kg}kg, dose={injected_mbq}MBq, scaler={suv_scaler:.6e}")
            else:
                log(f"    WARNING: Missing SUV parameters (weight={weight_kg}, dose={injected_mbq})")
                missing_suv_params.append(f"{subject_id}/{session_id}")

            # Load cerebellum TAC for SUVR
            cerebellum_data = None
            cerebellum_tac_file = find_cerebellum_tac(rawdata_dir, subject_id, timepoint)
            if cerebellum_tac_file:
                try:
                    cerebellum_data = load_cerebellum_tac(cerebellum_tac_file)
                    log(f"    Cerebellum: mean={cerebellum_data['cerebellum_mean_bq_ml']:.2f} Bq/mL, AUC={cerebellum_data['cerebellum_auc_bq_s_ml']:.2f} Bq·s/mL")
                except Exception as e:
                    log(f"    WARNING: Could not load cerebellum TAC: {e}")
            else:
                log(f"    WARNING: Cerebellum TAC not found")
                missing_cerebellum.append(f"{subject_id}/{session_id}")

            # Load blood/plasma data for TPR
            plasma_data = None
            plasma_auc_result = None
            blood_tsv, blood_json = find_blood_file(rawdata_dir, subject_id, timepoint)
            if blood_tsv:
                try:
                    blood_data = load_blood_data(blood_tsv, blood_json)
                    log(f"    Blood data: {blood_data['n_samples']} samples")

                    # Calculate plasma AUC during scan window
                    if pet_timing and pet_timing['scan_start_s'] is not None:
                        scan_start = pet_timing['scan_start_s']
                        scan_end = scan_start + pet_timing['scan_duration_s']
                        plasma_auc_result = calculate_plasma_auc(blood_data, scan_start, scan_end)
                        log(f"    Plasma AUC: {plasma_auc_result['plasma_auc_kbq_s_ml']:.2f} kBq·s/mL, mean={plasma_auc_result['plasma_mean_kbq_ml']:.3f} kBq/mL")
                        for w in plasma_auc_result.get('warnings', []):
                            log(f"      {w}")
                            all_warnings.append(f"{subject_id}/{session_id}: {w}")

                        # Store for derived data
                        plasma_data = {
                            'blood_data': blood_data,
                            'plasma_auc': plasma_auc_result,
                            'scan_start': scan_start,
                            'scan_end': scan_end
                        }
                except Exception as e:
                    log(f"    WARNING: Could not load blood data: {e}")
            else:
                log(f"    WARNING: Blood data not found")
                missing_blood.append(f"{subject_id}/{session_id}")

            # Build derived data row for this session
            derived_row = {
                'subject_id': subject_id,
                'session_blinded': session_id,
                'session_unblinded': timepoint,
                'injected_MBq': injected_mbq,
                'body_weight_kg': weight_kg,
                'SUV_scaler': suv_scaler,
                'CER_AUC': cerebellum_data['cerebellum_auc_bq_s_ml'] if cerebellum_data else np.nan,
                'plasma_brain_chunk_AUC': plasma_auc_result['plasma_auc_kbq_s_ml'] if plasma_auc_result else np.nan,
                'plasma_total_session_AUC': calculate_plasma_total_auc(blood_data) if blood_tsv and blood_data else np.nan
            }
            derived_data.append(derived_row)

            # Process each eye
            for eye in ['left', 'right']:
                mask_file = find_mask_file(pet_dir, subject_id, session_id, eye)

                if mask_file is None:
                    log(f"    {eye}: Mask not found")
                    missing_masks.append(f"{subject_id}/{session_id}/{eye}")
                    continue

                # Check for filename typos
                if 'OHN' in mask_file.name:
                    log(f"    {eye}: Note - filename contains 'OHN' typo (expected 'ONH')")

                # Load mask
                try:
                    mask_data, mask_img = load_nifti_with_scaling(mask_file)
                    # Binarize mask (in case it has non-binary values)
                    mask_data = (mask_data > 0).astype(np.float32)
                except Exception as e:
                    log(f"    {eye}: ERROR loading mask: {e}")
                    continue

                # Verify mask and PET have same shape
                if pet_data.shape != mask_data.shape:
                    log(f"    {eye}: ERROR - shape mismatch: PET={pet_data.shape}, mask={mask_data.shape}")
                    continue

                # Calculate raw metrics (intensities in Bq/mL)
                metrics = calculate_metrics(pet_data, mask_data, voxel_dims, sphere_radius_mm=2.0)

                if 'error' in metrics:
                    log(f"    {eye}: ERROR - {metrics['error']}")
                    continue

                # Extract raw intensity values
                intensity_max = metrics['intensity_max']
                intensity_peak = metrics['intensity_peak']
                intensity_top150_mean = metrics['intensity_top150_mean']
                intensity_top150_median = metrics['intensity_top150_median']
                intensity_top150_p90 = metrics['intensity_top150_p90']

                # Calculate SUV values
                suv_max = convert_to_suv(intensity_max, suv_scaler) if suv_scaler else np.nan
                suv_peak = convert_to_suv(intensity_peak, suv_scaler) if suv_scaler else np.nan
                suv_top150_mean = convert_to_suv(intensity_top150_mean, suv_scaler) if suv_scaler else np.nan
                suv_top150_median = convert_to_suv(intensity_top150_median, suv_scaler) if suv_scaler else np.nan
                suv_top150_p90 = convert_to_suv(intensity_top150_p90, suv_scaler) if suv_scaler else np.nan

                # Calculate SUVR values
                if cerebellum_data:
                    suvr_max = calculate_suvr(intensity_max, cerebellum_data['cerebellum_mean_bq_ml'])
                    suvr_peak = calculate_suvr(intensity_peak, cerebellum_data['cerebellum_mean_bq_ml'])
                    suvr_top150_mean = calculate_suvr(intensity_top150_mean, cerebellum_data['cerebellum_mean_bq_ml'])
                    suvr_top150_median = calculate_suvr(intensity_top150_median, cerebellum_data['cerebellum_mean_bq_ml'])
                    suvr_top150_p90 = calculate_suvr(intensity_top150_p90, cerebellum_data['cerebellum_mean_bq_ml'])
                else:
                    suvr_max = np.nan
                    suvr_peak = np.nan
                    suvr_top150_mean = np.nan
                    suvr_top150_median = np.nan
                    suvr_top150_p90 = np.nan

                # Calculate TPR values
                if plasma_auc_result and plasma_auc_result['plasma_mean_kbq_ml']:
                    tpr_max = calculate_tpr(intensity_max, plasma_auc_result['plasma_mean_kbq_ml'])
                    tpr_peak = calculate_tpr(intensity_peak, plasma_auc_result['plasma_mean_kbq_ml'])
                    tpr_top150_mean = calculate_tpr(intensity_top150_mean, plasma_auc_result['plasma_mean_kbq_ml'])
                    tpr_top150_median = calculate_tpr(intensity_top150_median, plasma_auc_result['plasma_mean_kbq_ml'])
                    tpr_top150_p90 = calculate_tpr(intensity_top150_p90, plasma_auc_result['plasma_mean_kbq_ml'])
                else:
                    tpr_max = np.nan
                    tpr_peak = np.nan
                    tpr_top150_mean = np.nan
                    tpr_top150_median = np.nan
                    tpr_top150_p90 = np.nan

                # Calculate FUR values
                fur_max = np.nan
                fur_peak = np.nan
                fur_top150_mean = np.nan
                fur_top150_median = np.nan
                fur_top150_p90 = np.nan
                fur_auc = np.nan
                scan_midpoint_s = np.nan

                if_key = (subject_id, timepoint)
                if if_key in input_function_cache and pet_timing and pet_timing['scan_start_s']:
                    if_data = input_function_cache[if_key]
                    # Calculate scan midpoint: ScanStart + (FrameDuration / 2)
                    scan_midpoint_s = pet_timing['scan_start_s'] + (pet_timing['scan_duration_s'] / 2)

                    # Calculate AUC from 0 to scan midpoint for FUR
                    auc_result = calculate_input_function_auc(if_data, scan_midpoint_s)
                    fur_auc = auc_result['auc_0_to_midpoint_Bq_s_mL']

                    if not np.isnan(fur_auc) and fur_auc > 0:
                        fur_max = calculate_fur(intensity_max, fur_auc)
                        fur_peak = calculate_fur(intensity_peak, fur_auc)
                        fur_top150_mean = calculate_fur(intensity_top150_mean, fur_auc)
                        fur_top150_median = calculate_fur(intensity_top150_median, fur_auc)
                        fur_top150_p90 = calculate_fur(intensity_top150_p90, fur_auc)

                        # Save processed input function (once per session, on first eye)
                        if eye == 'left':
                            output_if_path = input_func_dir / f"{subject_id}_ses-{timepoint}_if_processed.csv"
                            if not output_if_path.exists():
                                save_processed_input_function(if_data, scan_midpoint_s, output_if_path)
                                log(f"    Saved processed IF (truncated at {scan_midpoint_s:.0f}s = {scan_midpoint_s/60:.1f} min)")

                    # Log any FUR-related warnings
                    for w in auc_result.get('warnings', []):
                        if 'ERROR' in w or 'WARN' in w:
                            log(f"    FUR: {w}")
                            all_warnings.append(f"{subject_id}/{session_id}: FUR: {w}")

                # Get baseline metrics for comparison (if this is followup)
                baseline_metrics = baseline_cache.get((subject_id, eye))

                # Generate QC warnings
                warnings = generate_qc_flags(metrics, eye, baseline_metrics, suv_max=suv_max if not np.isnan(suv_max) else None)
                if warnings:
                    for w in warnings:
                        log(f"    {eye}: {w}")
                        all_warnings.append(f"{subject_id}/{session_id}/{eye}: {w}")

                # Cache baseline metrics
                if timepoint == "Baseline":
                    baseline_cache[(subject_id, eye)] = metrics

                # Prepare result row
                row = {
                    'subject_id': subject_id,
                    'session_blinded': session_id,
                    'session_unblinded': timepoint,
                    'eye': eye,
                    'SUVmax': suv_max,
                    'SUVpeak_2mm': suv_peak,
                    'SUVtop150_mean': suv_top150_mean,
                    'SUVtop150_median': suv_top150_median,
                    'SUVtop150_p90': suv_top150_p90,
                    'SUVR_max': suvr_max,
                    'SUVR_peak_2mm': suvr_peak,
                    'SUVR_top150_mean': suvr_top150_mean,
                    'SUVR_top150_median': suvr_top150_median,
                    'SUVR_top150_p90': suvr_top150_p90,
                    'TPR_max': tpr_max,
                    'TPR_peak_2mm': tpr_peak,
                    'TPR_top150_mean': tpr_top150_mean,
                    'TPR_top150_median': tpr_top150_median,
                    'TPR_top150_p90': tpr_top150_p90,
                    'FUR_max': fur_max,
                    'FUR_peak_2mm': fur_peak,
                    'FUR_top150_mean': fur_top150_mean,
                    'FUR_top150_median': fur_top150_median,
                    'FUR_top150_p90': fur_top150_p90,
                    'intensity_max_Bq_ml': intensity_max,
                    'intensity_peak_Bq_ml': intensity_peak,
                    'intensity_top150_mean_Bq_ml': intensity_top150_mean,
                    'intensity_top150_median_Bq_ml': intensity_top150_median,
                    'intensity_top150_p90_Bq_ml': intensity_top150_p90,
                    'mask_volume_voxels': metrics['mask_volume_voxels'],
                    'mask_volume_mm3': metrics['mask_volume_mm3'],
                    'max_voxel_x': metrics['max_voxel_x'],
                    'max_voxel_y': metrics['max_voxel_y'],
                    'max_voxel_z': metrics['max_voxel_z'],
                    'sphere_voxel_count': metrics['sphere_voxel_count'],
                    'scan_start_s': pet_timing['scan_start_s'] if pet_timing else np.nan,
                    'scan_duration_s': pet_timing['scan_duration_s'] if pet_timing else np.nan,
                    'scan_midpoint_s': scan_midpoint_s,
                    'IF_AUC_0_to_midpoint_Bq_s_mL': fur_auc,
                    'plasma_auc_kBq_s_mL': plasma_auc_result['plasma_auc_kbq_s_ml'] if plasma_auc_result else np.nan,
                    'plasma_mean_kBq_mL': plasma_auc_result['plasma_mean_kbq_ml'] if plasma_auc_result else np.nan,
                    'plasma_samples_in_window': plasma_auc_result['plasma_samples_in_window'] if plasma_auc_result else np.nan,
                    'pet_file': str(pet_file.relative_to(project_root)),
                    'mask_file': str(mask_file.relative_to(project_root)),
                    'blood_file': str(blood_tsv.relative_to(project_root)) if blood_tsv else ''
                }
                results.append(row)
                # Format FUR for logging (handle NaN)
                fur_max_str = f"{fur_max:.4f}" if not np.isnan(fur_max) else "NaN"
                log(f"    {eye}: SUVmax={suv_max:.3f}, SUVpeak={suv_peak:.3f}, SUVR_max={suvr_max:.3f}, TPR_max={tpr_max:.3f}, FUR_max={fur_max_str}")

    # Create output DataFrame
    df = pd.DataFrame(results)

    # Sort by subject, session, eye
    df = df.sort_values(['subject_id', 'session_unblinded', 'eye'])

    # Save main output
    output_file = outputs_dir / "ONH_FDG_metrics.csv"
    df.to_csv(output_file, index=False)

    # Save derived data
    derived_df = pd.DataFrame(derived_data)
    derived_df = derived_df.sort_values(['subject_id', 'session_unblinded'])
    derived_file = derived_dir / "session_scaling_factors.csv"
    derived_df.to_csv(derived_file, index=False)

    # Check left/right symmetry within sessions
    log("\n--- Left/Right Symmetry Check ---")
    symmetry_warnings = []
    for (subj, sess), group in df.groupby(['subject_id', 'session_blinded']):
        if len(group) == 2:
            left_row = group[group['eye'] == 'left']
            right_row = group[group['eye'] == 'right']
            if len(left_row) == 1 and len(right_row) == 1:
                left_val = left_row['SUVmax'].values[0]
                right_val = right_row['SUVmax'].values[0]
                if not np.isnan(left_val) and not np.isnan(right_val) and min(left_val, right_val) > 0:
                    asymmetry = abs(left_val - right_val) / max(left_val, right_val) * 100
                    if asymmetry > 50:
                        warn_msg = f"{subj}/{sess}: L/R asymmetry {asymmetry:.1f}% (L={left_val:.2f}, R={right_val:.2f})"
                        log(f"  WARN: {warn_msg}")
                        symmetry_warnings.append(warn_msg)

    # Verify SUVmax > SUVpeak for all rows
    log("\n--- SUVmax vs SUVpeak Validation ---")
    violations = df[(df['SUVmax'] < df['SUVpeak_2mm']) & df['SUVmax'].notna() & df['SUVpeak_2mm'].notna()]
    if len(violations) > 0:
        log(f"  WARNING: {len(violations)} rows have SUVmax < SUVpeak (unexpected)")
        for _, row in violations.iterrows():
            log(f"    {row['subject_id']}/{row['session_blinded']}/{row['eye']}")
    else:
        log("  OK: All rows have SUVmax >= SUVpeak (as expected)")

    # Print summary
    unique_subjects = df['subject_id'].nunique()
    unique_sessions = len(df.groupby(['subject_id', 'session_blinded']))
    left_count = len(df[df['eye'] == 'left'])
    right_count = len(df[df['eye'] == 'right'])

    log("\n" + "=" * 60)
    log("=== Processing Complete ===")
    log(f"Subjects processed: {unique_subjects}")
    log(f"Sessions processed: {unique_sessions}")
    log(f"Eyes processed: {len(df)} (Left: {left_count}, Right: {right_count})")
    log(f"Rows in output CSV: {len(df)}")

    if missing_masks:
        log(f"Missing masks: {len(missing_masks)}")
        for m in missing_masks:
            log(f"  - {m}")
    else:
        log("Missing masks: 0")

    if missing_pet:
        log(f"Missing PET files: {len(missing_pet)}")
        for m in missing_pet:
            log(f"  - {m}")

    if missing_suv_params:
        log(f"Missing SUV parameters: {len(missing_suv_params)}")
        for m in missing_suv_params:
            log(f"  - {m}")

    if missing_cerebellum:
        log(f"Missing cerebellum TACs: {len(missing_cerebellum)}")

    if missing_blood:
        log(f"Missing blood data: {len(missing_blood)}")

    if missing_input_functions:
        log(f"Missing input functions: {len(missing_input_functions)}")

    total_warnings = len(all_warnings) + len(symmetry_warnings)
    log(f"Total warnings: {total_warnings}")

    log(f"\nOutput saved to: {output_file}")
    log(f"Derived data saved to: {derived_file}")
    log(f"Log saved to: {log_file}")

    # Generate QC reports
    qc_dir = analysis_dir / "QC"
    qc_dir.mkdir(parents=True, exist_ok=True)

    qc_flags, qc_summary = generate_qc_reports(
        df=df,
        derived_df=derived_df,
        missing_masks=missing_masks,
        all_warnings=all_warnings,
        symmetry_warnings=symmetry_warnings,
        timestamp=timestamp
    )

    # Save QC flags CSV
    qc_flags_file = qc_dir / "QC_flags_report.csv"
    qc_flags.to_csv(qc_flags_file, index=False)
    log(f"QC flags saved to: {qc_flags_file}")

    # Save QC summary
    qc_summary_file = qc_dir / "QC_summary_report.txt"
    with open(qc_summary_file, 'w') as f:
        f.write(qc_summary)
    log(f"QC summary saved to: {qc_summary_file}")

    # Generate QC visualizations (SUVpeak sphere placement)
    log("\nGenerating QC visualizations...")
    try:
        from qc_visualizations import generate_all_qc_visualizations
        n_images = generate_all_qc_visualizations(project_root, df, log_func=log, analysis_dir=analysis_dir)
        log(f"Generated {n_images} QC visualization images")
        log(f"QC images saved to: {qc_dir / 'SUVpeak_visualizations'}")
    except ImportError as e:
        log(f"WARNING: Could not import qc_visualizations module: {e}")
    except Exception as e:
        log(f"WARNING: Error generating QC visualizations: {e}")

    log("=" * 60)

    return df


def generate_qc_reports(df: pd.DataFrame, derived_df: pd.DataFrame,
                        missing_masks: list, all_warnings: list,
                        symmetry_warnings: list, timestamp: str) -> tuple:
    """
    Generate QC flags CSV and summary report.

    Returns:
        Tuple of (flags_dataframe, summary_text)
    """
    qc_flags = []

    # 1. Missing mask flags
    for item in missing_masks:
        parts = item.split('/')
        if len(parts) == 3:
            subj, sess, eye = parts
            # Get timepoint from df or derive from session
            timepoint = "Unknown"
            matching = df[(df['subject_id'] == subj) & (df['session_blinded'] == sess)]
            if len(matching) > 0:
                timepoint = matching.iloc[0]['session_unblinded']
            else:
                # Check derived data
                derived_match = derived_df[(derived_df['subject_id'] == subj) & (derived_df['session_blinded'] == sess)]
                if len(derived_match) > 0:
                    timepoint = derived_match.iloc[0]['session_unblinded']

            qc_flags.append({
                'subject_id': subj,
                'session': timepoint,
                'eye': eye,
                'flag_category': 'missing_mask',
                'flag_description': 'Mask file not found - data not extracted',
                'severity': 'CRITICAL',
                'value': 'NA',
                'threshold': 'required',
                'recommendation': 'Create mask delineation'
            })

    # 2. Parse warnings for mask volume and other issues
    for warning in all_warnings:
        # Parse warning format: "subject_id/session_id/eye: WARN: message"
        if ': WARN:' in warning or ': INFO:' in warning:
            parts = warning.split(': ', 1)
            if len(parts) >= 2:
                location = parts[0]
                message = parts[1].replace('WARN: ', '').replace('INFO: ', '')
                loc_parts = location.split('/')

                if len(loc_parts) >= 2:
                    subj = loc_parts[0]
                    sess_or_eye = loc_parts[1]

                    # Determine eye (if present)
                    eye = 'both'
                    session = sess_or_eye
                    if len(loc_parts) == 3:
                        session = loc_parts[1]
                        eye = loc_parts[2]

                    # Get timepoint
                    timepoint = session
                    if session.startswith('ses-'):
                        derived_match = derived_df[(derived_df['subject_id'] == subj) & (derived_df['session_blinded'] == session)]
                        if len(derived_match) > 0:
                            timepoint = derived_match.iloc[0]['session_unblinded']

                    # Categorize the warning
                    category = 'other'
                    severity = 'WARNING'
                    value = ''
                    threshold = ''
                    recommendation = ''

                    if 'Mask volume suspiciously large' in message:
                        category = 'mask_volume'
                        # Extract voxel count
                        import re
                        match = re.search(r'\((\d+) voxels\)', message)
                        if match:
                            value = match.group(1)
                        threshold = '>500'
                        recommendation = 'Review mask delineation'

                    elif 'Mask volume differs >25%' in message:
                        category = 'volume_change'
                        match = re.search(r'\((\d+\.?\d*)%\)', message)
                        if match:
                            value = f"{match.group(1)}%"
                        threshold = '>25%'
                        recommendation = 'Verify delineation consistency'

                    elif 'No plasma samples before scan start' in message:
                        category = 'plasma_sampling'
                        value = 'scan_start=0s'
                        threshold = 'plasma_first=1200s'
                        recommendation = 'Scan timing issue - verify with logs'

                    elif 'Only' in message and 'plasma sample' in message:
                        category = 'plasma_sampling'
                        match = re.search(r'Only (\d+) plasma', message)
                        if match:
                            value = match.group(1)
                        threshold = '>=2'
                        recommendation = 'Review blood sampling protocol'

                    elif 'plasma sample(s) with NA' in message:
                        category = 'plasma_data'
                        severity = 'INFO'
                        match = re.search(r'times: \[(\d+)\]', message)
                        if match:
                            value = f"time={match.group(1)}s"
                        threshold = 'NA'
                        recommendation = 'Data quality issue - verify with lab'

                    elif 'SUVmax unusually' in message:
                        category = 'suv_value'
                        match = re.search(r'\((\d+\.?\d*)\)', message)
                        if match:
                            value = match.group(1)
                        if 'low' in message:
                            threshold = '<0.5'
                        else:
                            threshold = '>30'
                        recommendation = 'Review PET data and mask placement'

                    qc_flags.append({
                        'subject_id': subj,
                        'session': timepoint,
                        'eye': eye,
                        'flag_category': category,
                        'flag_description': message,
                        'severity': severity,
                        'value': value,
                        'threshold': threshold,
                        'recommendation': recommendation
                    })

    # 3. Add symmetry warnings
    for warning in symmetry_warnings:
        # Format: "subj/sess: L/R asymmetry X% (L=Y, R=Z)"
        parts = warning.split(': ', 1)
        if len(parts) == 2:
            location = parts[0]
            message = parts[1]
            loc_parts = location.split('/')
            if len(loc_parts) == 2:
                subj, sess = loc_parts
                import re
                match = re.search(r'(\d+\.?\d*)%', message)
                value = match.group(1) + '%' if match else ''

                timepoint = sess
                derived_match = derived_df[(derived_df['subject_id'] == subj) & (derived_df['session_blinded'] == sess)]
                if len(derived_match) > 0:
                    timepoint = derived_match.iloc[0]['session_unblinded']

                qc_flags.append({
                    'subject_id': subj,
                    'session': timepoint,
                    'eye': 'both',
                    'flag_category': 'asymmetry',
                    'flag_description': f'Left/right SUVmax asymmetry >50%: {message}',
                    'severity': 'WARNING',
                    'value': value,
                    'threshold': '>50%',
                    'recommendation': 'Review bilateral mask placement'
                })

    # Convert to DataFrame
    flags_df = pd.DataFrame(qc_flags)
    if len(flags_df) > 0:
        flags_df = flags_df.sort_values(['subject_id', 'session', 'eye', 'flag_category'])

    # Generate summary report
    summary = generate_qc_summary(df, derived_df, flags_df, missing_masks, timestamp)

    return flags_df, summary


def generate_qc_summary(df: pd.DataFrame, derived_df: pd.DataFrame,
                        flags_df: pd.DataFrame, missing_masks: list,
                        timestamp: str) -> str:
    """Generate human-readable QC summary report."""

    lines = []
    lines.append("=" * 80)
    lines.append("ERAP ONH FDG-PET EXTRACTION - QUALITY CONTROL SUMMARY REPORT")
    lines.append(f"Generated: {timestamp[:8]}")
    lines.append("=" * 80)
    lines.append("")

    # Overview
    n_subjects = df['subject_id'].nunique()
    n_sessions = len(derived_df)
    n_eyes = len(df)
    n_missing = len(missing_masks)

    lines.append("OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"Total subjects processed: {n_subjects}")
    lines.append(f"Total sessions: {n_sessions}")
    lines.append(f"Total eyes processed: {n_eyes} ({n_missing} masks missing)")
    lines.append(f"Total QC flags raised: {len(flags_df)}")
    lines.append("")

    # Flag summary by category
    lines.append("=" * 80)
    lines.append("FLAG SUMMARY BY CATEGORY")
    lines.append("=" * 80)
    lines.append("")

    if len(flags_df) > 0:
        for category in flags_df['flag_category'].unique():
            cat_flags = flags_df[flags_df['flag_category'] == category]
            lines.append(f"{category.upper()} ({len(cat_flags)} flags)")
            lines.append("-" * 40)

            for _, row in cat_flags.iterrows():
                lines.append(f"  {row['subject_id']}/{row['session']}/{row['eye']}: {row['flag_description']}")

            lines.append("")
    else:
        lines.append("No flags raised - all data passed QC checks.")
        lines.append("")

    # Data quality metrics
    lines.append("=" * 80)
    lines.append("DATA QUALITY METRICS")
    lines.append("=" * 80)
    lines.append("")

    if len(df) > 0:
        lines.append("SUV Values:")
        lines.append(f"  Range: {df['SUVmax'].min():.2f} - {df['SUVmax'].max():.2f}")
        lines.append(f"  Mean:  {df['SUVmax'].mean():.2f}")
        lines.append("")

        lines.append("SUVR Values (vs cerebellum):")
        lines.append(f"  Range: {df['SUVR_max'].min():.2f} - {df['SUVR_max'].max():.2f}")
        lines.append(f"  Mean:  {df['SUVR_max'].mean():.2f}")
        lines.append("")

        tpr_valid = df['TPR_max'].dropna()
        if len(tpr_valid) > 0:
            lines.append("TPR Values:")
            lines.append(f"  Range: {tpr_valid.min():.2f} - {tpr_valid.max():.2f}")
            lines.append(f"  Mean:  {tpr_valid.mean():.2f}")
            lines.append(f"  Missing: {len(df) - len(tpr_valid)} sessions")
        lines.append("")

        lines.append("Mask Volumes (voxels):")
        lines.append(f"  Range: {df['mask_volume_voxels'].min():.0f} - {df['mask_volume_voxels'].max():.0f}")
        lines.append(f"  Mean:  {df['mask_volume_voxels'].mean():.0f}")
        lines.append("")

    # Recommendations
    lines.append("=" * 80)
    lines.append("RECOMMENDATIONS")
    lines.append("=" * 80)
    lines.append("")

    critical_flags = flags_df[flags_df['severity'] == 'CRITICAL'] if len(flags_df) > 0 else pd.DataFrame()
    if len(critical_flags) > 0:
        lines.append("HIGH PRIORITY (CRITICAL):")
        for _, row in critical_flags.iterrows():
            lines.append(f"  - {row['subject_id']}/{row['session']}/{row['eye']}: {row['recommendation']}")
        lines.append("")

    warning_flags = flags_df[flags_df['severity'] == 'WARNING'] if len(flags_df) > 0 else pd.DataFrame()
    if len(warning_flags) > 0:
        # Group by recommendation
        rec_counts = warning_flags['recommendation'].value_counts()
        lines.append("MEDIUM PRIORITY (WARNINGS):")
        for rec, count in rec_counts.items():
            lines.append(f"  - {rec} ({count} instances)")
        lines.append("")

    # Files generated
    lines.append("=" * 80)
    lines.append("FILES GENERATED")
    lines.append("=" * 80)
    lines.append("")
    lines.append("QC_flags_report.csv   - Detailed CSV of all individual flags")
    lines.append("QC_summary_report.txt - This summary document")
    lines.append("")
    lines.append("Related output files:")
    lines.append("  ../Outputs/ONH_FDG_metrics.csv          - Main metrics table")
    lines.append("  ../DerivedData/session_scaling_factors.csv - Per-session scaling factors")
    lines.append("  ../LogNotes/extraction_log_*.txt        - Processing logs")
    lines.append("")
    lines.append("=" * 80)

    return '\n'.join(lines)


if __name__ == "__main__":
    main()
