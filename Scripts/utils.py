"""
Utility functions for ONH FDG-PET metric extraction.

This module provides helper functions for:
- File discovery (PET images and masks)
- NIfTI image loading with proper scaling
- Sphere mask creation for SUVpeak calculation
- Blinding key parsing
- SUV calculation from eCRF data
- SUVR calculation using cerebellum reference
- TPR calculation using plasma input function
"""

import os
import re
import json
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


def load_blinding_key(project_root: Path) -> Dict[Tuple[str, str], str]:
    """
    Load the blinding key CSV and create a mapping from (subject, blinded_session) to timepoint.

    Args:
        project_root: Path to the project root directory

    Returns:
        Dictionary mapping (subject_id, blinded_session_code) to timepoint (Baseline/Followup)
    """
    # Check possible locations (project root and BlindKey subfolder)
    possible_paths = [
        project_root / "Blinding_key.csv",
        project_root / "BlindKey" / "Blinding_key.csv"
    ]

    blinding_file = None
    for path in possible_paths:
        if path.exists():
            blinding_file = path
            break

    if blinding_file is None:
        raise FileNotFoundError(f"Blinding key not found in expected locations: {possible_paths}")

    df = pd.read_csv(blinding_file)

    # Create mapping: (participant_id, ses-{Blind.code}) -> Session
    mapping = {}
    for _, row in df.iterrows():
        subject_id = row['participant_id']
        blinded_code = f"ses-{row['Blind.code']}"
        timepoint = row['Session']
        mapping[(subject_id, blinded_code)] = timepoint

    return mapping


def load_ecrf_data(rawdata_dir: Path) -> pd.DataFrame:
    """
    Load the eCRF data CSV containing weight and injected dose information.

    Args:
        rawdata_dir: Path to RawData directory

    Returns:
        DataFrame with eCRF data
    """
    ecrf_path = rawdata_dir / "eCRF_data" / "K8ERAPKIH22001_DATA_2025-05-19_1128.csv"
    if not ecrf_path.exists():
        raise FileNotFoundError(f"eCRF data not found at: {ecrf_path}")

    df = pd.read_csv(ecrf_path, encoding='utf-8-sig')
    return df


def get_suv_parameters(ecrf_df: pd.DataFrame, subject_id: str, timepoint: str) -> Dict[str, float]:
    """
    Extract SUV calculation parameters from eCRF data.

    Args:
        ecrf_df: eCRF DataFrame
        subject_id: Subject ID (e.g., 'sub-101')
        timepoint: 'Baseline' or 'Followup'

    Returns:
        Dictionary with 'weight_kg' and 'injected_mbq' keys
    """
    # Extract numeric subject ID
    subj_num = int(subject_id.replace('sub-', ''))

    # Find the row for this subject
    row = ecrf_df[ecrf_df['subject_id'] == subj_num]
    if len(row) == 0:
        return {'weight_kg': None, 'injected_mbq': None, 'error': f'Subject {subj_num} not found in eCRF'}

    row = row.iloc[0]

    # Determine which PET session (pet_1 = Baseline, pet_2 = Followup)
    if timepoint == 'Baseline':
        weight_col = 'weight_kg_pet_1'
        dose_col = 'injected_mbq_pet_1'
    else:  # Followup
        weight_col = 'weight_kg_pet_2'
        dose_col = 'injected_mbq_pet_2'

    # Get values, handling potential string/numeric issues
    weight = row.get(weight_col)
    dose = row.get(dose_col)

    # Clean weight value (may have comma as decimal separator)
    if pd.notna(weight):
        if isinstance(weight, str):
            weight = float(weight.replace(',', '.'))
        else:
            weight = float(weight)
    else:
        weight = None

    # Clean dose value
    if pd.notna(dose):
        if isinstance(dose, str):
            dose = float(dose.replace(',', '.'))
        else:
            dose = float(dose)
    else:
        dose = None

    return {'weight_kg': weight, 'injected_mbq': dose}


def calculate_suv_scaler(weight_kg: float, injected_mbq: float) -> float:
    """
    Calculate the SUV scaling factor.

    SUV = (PET_Bq_ml) / (injected_Bq / weight_g)
        = (PET_Bq_ml * weight_g) / injected_Bq

    Since PET is in Bq/mL and we want dimensionless SUV:
    - injected_mbq needs to be converted to Bq (multiply by 1e6)
    - weight_kg needs to be converted to g (multiply by 1000)

    SUV = PET_Bq_ml / (injected_Bq / weight_g)
        = PET_Bq_ml * weight_g / injected_Bq
        = PET_Bq_ml * (weight_kg * 1000) / (injected_mbq * 1e6)
        = PET_Bq_ml * weight_kg / (injected_mbq * 1000)

    Args:
        weight_kg: Body weight in kg
        injected_mbq: Injected activity in MBq

    Returns:
        SUV scaling factor (multiply PET Bq/mL by this to get SUV)
    """
    # SUV_scaler = weight_kg / (injected_mbq * 1000)
    # This converts: Bq/mL * (kg / (MBq * 1000)) = Bq/mL * kg / (Bq * 1e-3 * 1000) = Bq/mL * kg / Bq = kg/mL
    # We need: (Bq/mL) / (Bq/g) = g/mL which is dimensionless when density = 1g/mL

    # Correct formula:
    # SUV = C_tissue / (injected_dose / body_weight)
    # C_tissue in Bq/mL
    # injected_dose in Bq = MBq * 1e6
    # body_weight in g = kg * 1000

    # SUV = C_tissue * body_weight_g / injected_dose_Bq
    #     = C_tissue * (weight_kg * 1000) / (injected_mbq * 1e6)
    #     = C_tissue * weight_kg / (injected_mbq * 1000)

    return weight_kg / (injected_mbq * 1000)


def convert_to_suv(pet_value_bq_ml: float, suv_scaler: float) -> float:
    """
    Convert PET intensity (Bq/mL) to SUV.

    Args:
        pet_value_bq_ml: PET value in Bq/mL
        suv_scaler: SUV scaling factor from calculate_suv_scaler()

    Returns:
        SUV value (dimensionless)
    """
    return pet_value_bq_ml * suv_scaler


def discover_subjects(rawdata_dir: Path) -> List[str]:
    """
    Discover all subject directories in RawData.

    Args:
        rawdata_dir: Path to RawData directory

    Returns:
        Sorted list of subject IDs (e.g., ['sub-101', 'sub-102', ...])
    """
    subjects = []
    for item in rawdata_dir.iterdir():
        if item.is_dir() and item.name.startswith('sub-') and 'ScalarVolume' not in item.name:
            subjects.append(item.name)
    return sorted(subjects)


def discover_sessions(subject_dir: Path) -> List[str]:
    """
    Discover all session directories for a given subject.

    Args:
        subject_dir: Path to subject directory

    Returns:
        List of session IDs (e.g., ['ses-fnfgs', 'ses-qbimm'])
    """
    sessions = []
    for item in subject_dir.iterdir():
        if item.is_dir() and item.name.startswith('ses-') and 'ScalarVolume' not in item.name:
            sessions.append(item.name)
    return sorted(sessions)


def find_pet_file(pet_dir: Path, subject_id: str, session_id: str) -> Optional[Path]:
    """
    Find the FDG-PET image file in the PET directory.

    Args:
        pet_dir: Path to the pet/ subdirectory
        subject_id: Subject ID (e.g., 'sub-101')
        session_id: Session ID (e.g., 'ses-fnfgs')

    Returns:
        Path to PET file or None if not found
    """
    # Expected pattern: *_chunk-brain_rec-StaticMoCo_trc-18FFDG_pet.nii
    pattern = f"{subject_id}_{session_id}_chunk-brain_rec-StaticMoCo_trc-18FFDG_pet.nii"

    pet_file = pet_dir / pattern
    if pet_file.exists():
        return pet_file

    # Try with .gz extension
    pet_file_gz = pet_dir / (pattern + ".gz")
    if pet_file_gz.exists():
        return pet_file_gz

    # Fallback: look for any PET file matching the pattern
    for f in pet_dir.glob("*_pet.nii*"):
        if 'ScalarVolume' not in str(f) and 'mask' not in f.name.lower():
            return f

    return None


def find_pet_json(pet_dir: Path, subject_id: str, session_id: str,
                  timepoint: Optional[str] = None, rawdata_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Find the PET JSON sidecar file.

    First checks the updated JSON folder (json_side_cars_updated) with timepoint-based naming,
    then falls back to the original location with blinded session naming.

    Args:
        pet_dir: Path to the pet/ subdirectory
        subject_id: Subject ID
        session_id: Session ID (blinded)
        timepoint: 'Baseline' or 'Followup' (optional, used for updated JSON lookup)
        rawdata_dir: Path to RawData directory (optional, needed for updated JSON lookup)

    Returns:
        Path to PET JSON file or None if not found
    """
    # First, check for updated JSON files with timepoint-based naming
    if timepoint and rawdata_dir:
        updated_json_dir = rawdata_dir / "json_side_cars_updated"
        if updated_json_dir.exists():
            # Pattern: sub-XXX_ses-{Baseline/Followup}_trc-18FFDG_rec-StaticMoCo_chunk-1_pet.json
            updated_pattern = f"{subject_id}_ses-{timepoint}_trc-18FFDG_rec-StaticMoCo_chunk-1_pet.json"
            updated_json = updated_json_dir / updated_pattern
            if updated_json.exists():
                return updated_json

    # Fallback: check original location with blinded session
    pattern = f"{subject_id}_{session_id}_chunk-brain_rec-StaticMoCo_trc-18FFDG_pet.json"
    json_file = pet_dir / pattern
    if json_file.exists():
        return json_file

    # Fallback: look for any PET JSON file
    for f in pet_dir.glob("*_pet.json"):
        if 'ScalarVolume' not in str(f):
            return f

    return None


def load_pet_json(json_path: Path) -> Dict[str, Any]:
    """
    Load PET JSON sidecar and extract timing information.

    Args:
        json_path: Path to PET JSON file

    Returns:
        Dictionary with scan timing info
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract relevant fields
    scan_start_raw = data.get('ScanStart', None)  # seconds after TimeZero

    # Handle missing or invalid ScanStart (0 is not physiologically valid for brain FDG)
    # Brain FDG scans typically start 30-60 min post-injection
    scan_start_warning = None
    if scan_start_raw is None or scan_start_raw == 0:
        # Default to 1800s (30 min) if ScanStart is missing or zero
        scan_start_warning = f"ScanStart was {scan_start_raw} in JSON - using default 1800s (30 min post-injection)"
        scan_start = 1800
    else:
        scan_start = scan_start_raw

    # FrameDuration can be in milliseconds or seconds
    frame_duration = data.get('FrameDuration', [1800000])
    if isinstance(frame_duration, list):
        frame_duration = frame_duration[0]
    # Convert to seconds if in milliseconds
    if frame_duration > 10000:  # Likely milliseconds
        frame_duration = frame_duration / 1000

    result = {
        'scan_start_s': scan_start,
        'scan_start_raw': scan_start_raw,
        'scan_duration_s': frame_duration,
        'time_zero': data.get('TimeZero', None),
        'units': data.get('Units', 'Bq/mL'),
        'injected_activity_bq': data.get('InjectedRadioactivity', None)
    }

    if scan_start_warning:
        result['scan_start_warning'] = scan_start_warning

    return result


def find_mask_file(pet_dir: Path, subject_id: str, session_id: str, eye: str) -> Optional[Path]:
    """
    Find the mask file for a given eye, handling various naming conventions.

    Handles:
    - *_left_ONH_mask.nii.gz (preferred)
    - *_left_OHN_mask.nii.gz (typo variant)
    - *_left_mask.nii (older naming)
    - *_left_mask.nii.gz

    Args:
        pet_dir: Path to the pet/ subdirectory
        subject_id: Subject ID
        session_id: Session ID
        eye: 'left' or 'right'

    Returns:
        Path to mask file or None if not found
    """
    base_pattern = f"{subject_id}_{session_id}_chunk-brain_rec-StaticMoCo_trc-18FFDG_pet"

    # Patterns to try, in order of preference
    patterns = [
        f"{base_pattern}_{eye}_ONH_mask.nii.gz",
        f"{base_pattern}_{eye}_ONH_mask.nii",
        f"{base_pattern}_{eye}_OHN_mask.nii.gz",  # Typo variant
        f"{base_pattern}_{eye}_OHN_mask.nii",
        f"{base_pattern}_{eye}_mask.nii.gz",
        f"{base_pattern}_{eye}_mask.nii",
    ]

    for pattern in patterns:
        mask_file = pet_dir / pattern
        if mask_file.exists():
            return mask_file

    # Fallback: search for any file matching the eye and mask pattern
    for f in pet_dir.glob(f"*{eye}*mask*"):
        if 'ScalarVolume' not in str(f):
            return f

    return None


def load_nifti_with_scaling(filepath: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """
    Load a NIfTI image with proper scaling applied.

    Args:
        filepath: Path to NIfTI file

    Returns:
        Tuple of (scaled data array, nibabel image object)
    """
    img = nib.load(filepath)
    # get_fdata() handles scl_slope and scl_inter automatically
    data = img.get_fdata(dtype=np.float32)
    return data, img


def get_voxel_dimensions(img: nib.Nifti1Image) -> np.ndarray:
    """
    Get voxel dimensions from NIfTI header.

    Args:
        img: nibabel image object

    Returns:
        Array of voxel dimensions [x, y, z] in mm
    """
    return np.array(img.header.get_zooms()[:3])


def create_sphere_mask(center: Tuple[int, int, int],
                       radius_mm: float,
                       voxel_dims: np.ndarray,
                       image_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Create a spherical mask centered at a given voxel coordinate.

    Args:
        center: (x, y, z) voxel coordinates of sphere center
        radius_mm: Radius of sphere in millimeters
        voxel_dims: Voxel dimensions in mm
        image_shape: Shape of the image (for bounds clipping)

    Returns:
        Boolean mask array of the same shape as the image
    """
    cx, cy, cz = center

    # Calculate radius in voxel units for each dimension
    radius_vox = radius_mm / voxel_dims

    # Determine bounding box to limit computation
    # Add 1 to ensure we capture edge voxels
    x_range = int(np.ceil(radius_vox[0])) + 1
    y_range = int(np.ceil(radius_vox[1])) + 1
    z_range = int(np.ceil(radius_vox[2])) + 1

    # Create output mask
    mask = np.zeros(image_shape, dtype=bool)

    # Iterate only within bounding box
    for dx in range(-x_range, x_range + 1):
        for dy in range(-y_range, y_range + 1):
            for dz in range(-z_range, z_range + 1):
                x, y, z = cx + dx, cy + dy, cz + dz

                # Check bounds
                if 0 <= x < image_shape[0] and 0 <= y < image_shape[1] and 0 <= z < image_shape[2]:
                    # Calculate distance in mm (accounting for anisotropic voxels)
                    dist = np.sqrt(
                        (dx * voxel_dims[0])**2 +
                        (dy * voxel_dims[1])**2 +
                        (dz * voxel_dims[2])**2
                    )
                    if dist <= radius_mm:
                        mask[x, y, z] = True

    return mask


def calculate_metrics(pet_data: np.ndarray,
                      mask_data: np.ndarray,
                      voxel_dims: np.ndarray,
                      sphere_radius_mm: float = 2.0) -> Dict:
    """
    Calculate max and peak values within a mask.

    Args:
        pet_data: 3D PET image array
        mask_data: 3D binary mask array
        voxel_dims: Voxel dimensions in mm
        sphere_radius_mm: Radius for peak sphere

    Returns:
        Dictionary containing all calculated metrics (raw intensities)
    """
    # Find mask voxels
    mask_indices = np.where(mask_data > 0)
    mask_voxel_count = len(mask_indices[0])

    if mask_voxel_count == 0:
        return {
            'error': 'Empty mask',
            'mask_volume_voxels': 0,
            'mask_volume_mm3': 0
        }

    # Calculate mask volume
    voxel_volume = np.prod(voxel_dims)
    mask_volume_mm3 = mask_voxel_count * voxel_volume

    # Extract PET values within mask
    pet_values_in_mask = pet_data[mask_indices]

    # Max value
    intensity_max = np.max(pet_values_in_mask)
    max_idx = np.argmax(pet_values_in_mask)
    max_coords = (mask_indices[0][max_idx],
                  mask_indices[1][max_idx],
                  mask_indices[2][max_idx])

    # Create sphere around max voxel
    sphere_mask = create_sphere_mask(max_coords, sphere_radius_mm, voxel_dims, pet_data.shape)

    # Get PET values in sphere, excluding zeros (air/background)
    sphere_values = pet_data[sphere_mask]
    nonzero_sphere_values = sphere_values[sphere_values > 0]

    # Peak (mean of non-zero values in sphere)
    sphere_total_count = np.sum(sphere_mask)
    sphere_nonzero_count = len(nonzero_sphere_values)

    if sphere_nonzero_count > 0:
        intensity_peak = np.mean(nonzero_sphere_values)
    else:
        intensity_peak = intensity_max  # Fallback to max if all zeros

    # Calculate zero percentage in sphere
    zero_percentage = (sphere_total_count - sphere_nonzero_count) / sphere_total_count * 100 if sphere_total_count > 0 else 0

    # Top-150 voxel metrics (robust alternative to max)
    # 150 voxels approximates 2 resolution elements at ONH location (~5.2mm FWHM)
    # 1 resolution element = ~74 voxels, so 150 ≈ 2 elements
    # Sort all mask voxels by intensity and take top 150
    top_n = 150
    sorted_values = np.sort(pet_values_in_mask)[::-1]  # Descending order
    if len(sorted_values) >= top_n:
        top_n_values = sorted_values[:top_n]
        intensity_top150_mean = np.mean(top_n_values)
        intensity_top150_median = np.median(top_n_values)
        intensity_top150_p90 = np.percentile(top_n_values, 90)  # 90th percentile of top 150
    else:
        # Fallback if mask has fewer than 150 voxels (shouldn't happen with current data)
        intensity_top150_mean = np.mean(sorted_values)
        intensity_top150_median = np.median(sorted_values)
        intensity_top150_p90 = np.percentile(sorted_values, 90)

    return {
        'intensity_max': intensity_max,
        'intensity_peak': intensity_peak,
        'intensity_top150_mean': intensity_top150_mean,
        'intensity_top150_median': intensity_top150_median,
        'intensity_top150_p90': intensity_top150_p90,
        'mask_volume_voxels': mask_voxel_count,
        'mask_volume_mm3': mask_volume_mm3,
        'max_voxel_x': int(max_coords[0]),
        'max_voxel_y': int(max_coords[1]),
        'max_voxel_z': int(max_coords[2]),
        'sphere_voxel_count': int(sphere_nonzero_count),
        'sphere_total_voxels': int(sphere_total_count),
        'sphere_zero_percentage': zero_percentage
    }


# ============================================================================
# CEREBELLUM / SUVR FUNCTIONS
# ============================================================================

def find_cerebellum_tac(rawdata_dir: Path, subject_id: str, timepoint: str) -> Optional[Path]:
    """
    Find the cerebellum TAC file for a subject/session.

    Args:
        rawdata_dir: Path to RawData directory
        subject_id: Subject ID (e.g., 'sub-101')
        timepoint: 'Baseline' or 'Followup'

    Returns:
        Path to cerebellum TAC TSV file or None
    """
    tac_dir = rawdata_dir / "Cerebellum_tacs"
    if not tac_dir.exists():
        return None

    pattern = f"{subject_id}_ses-{timepoint}_label-cerebellum_tacs.tsv"
    tac_file = tac_dir / pattern
    if tac_file.exists():
        return tac_file

    return None


def load_cerebellum_tac(tac_path: Path) -> Dict[str, Any]:
    """
    Load cerebellum TAC data and calculate mean activity during scan.

    The TAC file contains 10 frames with Mean(Bq/mL) values.
    We compute the time-weighted average (AUC / duration) across all frames.

    Args:
        tac_path: Path to cerebellum TAC TSV file

    Returns:
        Dictionary with cerebellum reference metrics
    """
    df = pd.read_csv(tac_path, sep='\t')

    # Expected columns: Frame, ROI, Mean(Bq/mL), FrameStart(s), FrameDuration(s)
    # Calculate AUC using frame durations as weights
    mean_values = df['Mean(Bq/mL)'].values
    frame_durations = df['FrameDuration(s)'].values
    frame_starts = df['FrameStart(s)'].values

    # Total duration
    total_duration = np.sum(frame_durations)

    # Time-weighted mean (AUC / duration)
    auc = np.sum(mean_values * frame_durations)
    cerebellum_mean = auc / total_duration

    return {
        'cerebellum_mean_bq_ml': cerebellum_mean,
        'cerebellum_auc_bq_s_ml': auc,
        'cerebellum_total_duration_s': total_duration,
        'cerebellum_n_frames': len(df)
    }


def calculate_suvr(pet_value_bq_ml: float, cerebellum_mean_bq_ml: float) -> float:
    """
    Calculate SUV ratio (SUVR) using cerebellum as reference.

    SUVR = tissue_activity / reference_activity

    Args:
        pet_value_bq_ml: PET value in Bq/mL (tissue)
        cerebellum_mean_bq_ml: Mean cerebellum activity in Bq/mL

    Returns:
        SUVR value (dimensionless)
    """
    if cerebellum_mean_bq_ml <= 0:
        return np.nan
    return pet_value_bq_ml / cerebellum_mean_bq_ml


# ============================================================================
# BLOOD PLASMA / TPR FUNCTIONS
# ============================================================================

def find_blood_file(rawdata_dir: Path, subject_id: str, timepoint: str) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Find the blood TSV and JSON files for a subject/session.

    Args:
        rawdata_dir: Path to RawData directory
        subject_id: Subject ID (e.g., 'sub-101')
        timepoint: 'Baseline' or 'Followup'

    Returns:
        Tuple of (TSV path, JSON path) - either may be None
    """
    blood_dir = rawdata_dir / "BloodPlasma"
    if not blood_dir.exists():
        return None, None

    tsv_pattern = f"{subject_id}_ses-{timepoint}_recording-manual_blood.tsv"
    json_pattern = f"{subject_id}_ses-{timepoint}_recording-manual_blood.json"

    tsv_file = blood_dir / tsv_pattern
    json_file = blood_dir / json_pattern

    return (tsv_file if tsv_file.exists() else None,
            json_file if json_file.exists() else None)


def load_blood_data(tsv_path: Path, json_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load blood/plasma data from TSV and optional JSON sidecar.

    Handles missing/NA values by filtering them out and reporting.

    Args:
        tsv_path: Path to blood TSV file
        json_path: Path to blood JSON sidecar (optional)

    Returns:
        Dictionary with blood data and metadata
    """
    df = pd.read_csv(tsv_path, sep='\t')

    # Get raw values
    times_raw = df['time'].values
    plasma_raw = df['plasma_radioactivity'].values
    whole_blood_raw = df['whole_blood_radioactivity'].values

    # Count original samples
    n_total = len(df)

    # Filter out rows where plasma is NaN/missing
    valid_mask = ~pd.isna(plasma_raw)
    n_valid = np.sum(valid_mask)
    n_missing = n_total - n_valid

    # Get valid samples only for plasma-related calculations
    times_valid = times_raw[valid_mask].astype(float)
    plasma_valid = plasma_raw[valid_mask].astype(float)

    result = {
        'time_s': times_valid,
        'plasma_kbq_ml': plasma_valid,
        'whole_blood_kbq_ml': whole_blood_raw,  # Keep original for reference
        'n_samples': n_valid,
        'n_samples_total': n_total,
        'n_samples_missing': n_missing
    }

    if n_missing > 0:
        result['missing_plasma_times'] = times_raw[~valid_mask].tolist()

    # Load metadata from JSON if available
    if json_path and json_path.exists():
        with open(json_path, 'r') as f:
            meta = json.load(f)
        result['metadata'] = meta

    return result


def interpolate_at_time(times: np.ndarray, values: np.ndarray, target_time: float) -> float:
    """
    Linear interpolation to estimate value at a specific time.

    Args:
        times: Array of time points
        values: Array of values at those time points
        target_time: Time at which to interpolate

    Returns:
        Interpolated value
    """
    # Find bracketing indices
    if target_time <= times[0]:
        return values[0]  # Use first value if before first sample
    if target_time >= times[-1]:
        return values[-1]  # Use last value if after last sample

    # Find the interval containing target_time
    for i in range(len(times) - 1):
        if times[i] <= target_time <= times[i + 1]:
            # Linear interpolation
            t0, t1 = times[i], times[i + 1]
            v0, v1 = values[i], values[i + 1]
            return v0 + (v1 - v0) * (target_time - t0) / (t1 - t0)

    return np.nan


def calculate_plasma_auc(blood_data: Dict, scan_start_s: float, scan_end_s: float) -> Dict[str, Any]:
    """
    Calculate plasma AUC during the scan window using trapezoidal integration.

    Handles missing plasma samples by skipping them and interpolating.

    Args:
        blood_data: Dictionary from load_blood_data()
        scan_start_s: Scan start time in seconds post-injection
        scan_end_s: Scan end time in seconds post-injection

    Returns:
        Dictionary with AUC metrics and QC info
    """
    times = blood_data['time_s']
    plasma = blood_data['plasma_kbq_ml']

    warnings = []

    # Check for missing samples that were filtered
    if blood_data.get('n_samples_missing', 0) > 0:
        missing_times = blood_data.get('missing_plasma_times', [])
        warnings.append(f"INFO: {blood_data['n_samples_missing']} plasma sample(s) with NA values were skipped (times: {missing_times})")

    # Check if we have enough valid samples
    if len(times) < 2:
        warnings.append("ERROR: Insufficient valid plasma samples for AUC calculation")
        return {
            'plasma_auc_kbq_s_ml': np.nan,
            'plasma_mean_kbq_ml': np.nan,
            'plasma_samples_in_window': 0,
            'interpolated_start_kbq_ml': np.nan,
            'interpolated_end_kbq_ml': np.nan,
            'warnings': warnings
        }

    # Check if any samples bracket the scan window
    samples_before_start = np.sum(times <= scan_start_s)
    samples_after_end = np.sum(times >= scan_end_s)

    if samples_before_start == 0:
        warnings.append("WARN: No plasma samples before scan start - extrapolating")
    if samples_after_end == 0:
        warnings.append("WARN: No plasma samples after scan end - extrapolating")

    # Build working sample list with interpolated boundaries
    working_times = []
    working_values = []

    # Add interpolated start value
    start_val = interpolate_at_time(times, plasma, scan_start_s)
    working_times.append(scan_start_s)
    working_values.append(start_val)

    # Add original samples within window
    samples_in_window = 0
    for t, v in zip(times, plasma):
        if scan_start_s < t < scan_end_s:
            working_times.append(t)
            working_values.append(v)
            samples_in_window += 1

    # Add interpolated end value
    end_val = interpolate_at_time(times, plasma, scan_end_s)
    working_times.append(scan_end_s)
    working_values.append(end_val)

    # Convert to arrays
    working_times = np.array(working_times)
    working_values = np.array(working_values)

    # Sort by time (should already be sorted, but just in case)
    sort_idx = np.argsort(working_times)
    working_times = working_times[sort_idx]
    working_values = working_values[sort_idx]

    # Calculate AUC using trapezoidal rule
    auc = 0.0
    for i in range(len(working_times) - 1):
        dt = working_times[i + 1] - working_times[i]
        auc += dt * (working_values[i] + working_values[i + 1]) / 2

    # Mean plasma concentration
    scan_duration = scan_end_s - scan_start_s
    mean_plasma = auc / scan_duration if scan_duration > 0 else np.nan

    # QC checks
    if samples_in_window < 2:
        warnings.append(f"WARN: Only {samples_in_window} plasma samples in scan window - interpolation may be unreliable")

    if np.any(plasma < 0.5) or np.any(plasma > 50):
        warnings.append("WARN: Plasma values outside typical range (0.5-50 kBq/mL)")

    return {
        'plasma_auc_kbq_s_ml': auc,
        'plasma_mean_kbq_ml': mean_plasma,
        'plasma_samples_in_window': samples_in_window,
        'interpolated_start_kbq_ml': start_val,
        'interpolated_end_kbq_ml': end_val,
        'warnings': warnings
    }


def calculate_plasma_total_auc(blood_data: Dict) -> float:
    """
    Calculate total AUC of plasma curve (all samples).

    Args:
        blood_data: Dictionary from load_blood_data()

    Returns:
        Total plasma AUC in kBq·s/mL
    """
    times = blood_data['time_s']
    plasma = blood_data['plasma_kbq_ml']

    # Trapezoidal integration over all samples
    auc = 0.0
    for i in range(len(times) - 1):
        dt = times[i + 1] - times[i]
        auc += dt * (plasma[i] + plasma[i + 1]) / 2

    return auc


def calculate_tpr(pet_value_bq_ml: float, plasma_mean_kbq_ml: float) -> float:
    """
    Calculate tissue-to-plasma ratio (TPR).

    Note: PET is in Bq/mL, plasma is in kBq/mL.
    Convert PET to kBq/mL (divide by 1000) before computing ratio.

    Args:
        pet_value_bq_ml: PET value in Bq/mL
        plasma_mean_kbq_ml: Mean plasma concentration in kBq/mL

    Returns:
        TPR value (dimensionless)
    """
    if plasma_mean_kbq_ml <= 0 or np.isnan(plasma_mean_kbq_ml):
        return np.nan

    # Convert PET from Bq/mL to kBq/mL
    pet_kbq_ml = pet_value_bq_ml / 1000.0

    return pet_kbq_ml / plasma_mean_kbq_ml


# ============================================================================
# QC FUNCTIONS
# ============================================================================

def generate_qc_flags(metrics: Dict, eye: str,
                      baseline_metrics: Optional[Dict] = None,
                      suv_max: Optional[float] = None) -> List[str]:
    """
    Generate QC warning flags for a measurement.

    Args:
        metrics: Dictionary of calculated metrics
        eye: 'left' or 'right'
        baseline_metrics: Metrics from baseline session for comparison (optional)
        suv_max: SUVmax value for additional QC (optional)

    Returns:
        List of warning messages
    """
    warnings = []

    # Check mask volume
    if metrics.get('mask_volume_voxels', 0) < 5:
        warnings.append(f"WARN: Mask volume suspiciously small ({metrics['mask_volume_voxels']} voxels)")

    if metrics.get('mask_volume_voxels', 0) > 500:
        warnings.append(f"WARN: Mask volume suspiciously large ({metrics['mask_volume_voxels']} voxels) - may include non-ONH tissue")

    # Check SUV values if provided
    if suv_max is not None:
        if suv_max < 0.5:
            warnings.append(f"WARN: SUVmax unusually low ({suv_max:.2f})")

        if suv_max > 30:
            warnings.append(f"WARN: SUVmax unusually high ({suv_max:.2f}) - may include extraocular muscle")

    # Check sphere zero percentage
    if metrics.get('sphere_zero_percentage', 0) > 50:
        warnings.append(f"WARN: Sphere contains >50% zeros ({metrics['sphere_zero_percentage']:.1f}%) - poor localization")

    # Compare to baseline if available
    if baseline_metrics:
        baseline_vol = baseline_metrics.get('mask_volume_voxels', 0)
        current_vol = metrics.get('mask_volume_voxels', 0)
        if baseline_vol > 0:
            vol_diff = abs(current_vol - baseline_vol) / baseline_vol * 100
            if vol_diff > 25:
                warnings.append(f"WARN: Mask volume differs >25% from baseline ({vol_diff:.1f}%)")

    return warnings


# ============================================================================
# INPUT FUNCTION / FUR FUNCTIONS
# ============================================================================

def find_input_function_file(rawdata_dir: Path, subject_id: str,
                              timepoint: str) -> Optional[Path]:
    """
    Find the input function TSV file for a subject/session.

    Args:
        rawdata_dir: Path to RawData directory
        subject_id: Subject ID (e.g., 'sub-101')
        timepoint: 'Baseline' or 'Followup'

    Returns:
        Path to input function TSV file or None
    """
    if_dir = rawdata_dir / "InputFunctions"
    if not if_dir.exists():
        return None

    pattern = f"{subject_id}_ses-{timepoint}_desc-IF_tacs.tsv"
    if_file = if_dir / pattern
    return if_file if if_file.exists() else None


def load_input_function(if_path: Path) -> Dict[str, Any]:
    """
    Load combined IDIF (aorta) + plasma input function from TSV file.

    Filters for ROI = "aorta" or "plasma", concatenates and sorts by time.
    Removes negative times and zero activity values at negative times.

    Args:
        if_path: Path to input function TSV file

    Returns:
        Dictionary with:
            - times: Array of time points (seconds), sorted
            - activities: Array of activity values (Bq/mL)
            - n_idif_samples: Count of IDIF (aorta) samples
            - n_plasma_samples: Count of plasma samples
            - warnings: List of any warnings
    """
    df = pd.read_csv(if_path, sep='\t')

    warnings = []

    # Filter for aorta and plasma ROIs only (exclude wbl = whole blood)
    mask = df['ROI'].isin(['aorta', 'plasma'])
    df_filtered = df[mask].copy()

    # Count samples by type
    n_idif = len(df_filtered[df_filtered['ROI'] == 'aorta'])
    n_plasma = len(df_filtered[df_filtered['ROI'] == 'plasma'])

    # Filter out negative times (pre-injection)
    df_filtered = df_filtered[df_filtered['Time(s)'] >= 0]

    # Filter out zero activity values (only at early times, keep real zeros if they exist later)
    # Actually, keep all data points and let interpolation handle it

    # Sort by time
    df_filtered = df_filtered.sort_values('Time(s)')

    times = df_filtered['Time(s)'].values.astype(float)
    activities = df_filtered['Radioactivity(Bq/mL)'].values.astype(float)

    # QC checks
    if n_idif < 5:
        warnings.append(f"WARN: Only {n_idif} IDIF (aorta) samples (expected >=5)")

    if n_plasma < 2:
        warnings.append(f"WARN: Only {n_plasma} plasma samples (expected >=2)")

    # Check for peak in IDIF
    idif_mask = df_filtered['ROI'] == 'aorta'
    if np.any(idif_mask):
        idif_peak = df_filtered.loc[idif_mask, 'Radioactivity(Bq/mL)'].max()
        if idif_peak < 10000:
            warnings.append(f"WARN: IDIF peak unusually low ({idif_peak:.0f} Bq/mL)")

    # Check for gap between IDIF and plasma
    if n_idif > 0 and n_plasma > 0:
        idif_times = df_filtered.loc[df_filtered['ROI'] == 'aorta', 'Time(s)']
        plasma_times = df_filtered.loc[df_filtered['ROI'] == 'plasma', 'Time(s)']
        gap = plasma_times.min() - idif_times.max()
        if gap > 800:
            warnings.append(f"INFO: Large gap between IDIF and plasma ({gap:.0f}s) - will interpolate")

    return {
        'times': times,
        'activities': activities,
        'n_idif_samples': n_idif,
        'n_plasma_samples': n_plasma,
        'warnings': warnings
    }


def calculate_input_function_auc(if_data: Dict[str, Any],
                                  scan_midpoint_s: float) -> Dict[str, Any]:
    """
    Calculate input function AUC from t=0 to scan_midpoint for FUR.

    Uses linear interpolation to create a dense time series, then
    trapezoidal integration.

    Args:
        if_data: Dictionary from load_input_function()
        scan_midpoint_s: Midpoint of PET scan in seconds post-injection

    Returns:
        Dictionary with:
            - auc_0_to_midpoint_Bq_s_mL: AUC for FUR calculation
            - interpolated_times: Dense time array used for integration
            - interpolated_activities: Interpolated activity values
            - warnings: List of any issues
    """
    from scipy.interpolate import interp1d

    times = if_data['times']
    activities = if_data['activities']
    warnings = list(if_data.get('warnings', []))

    if len(times) < 2:
        return {
            'auc_0_to_midpoint_Bq_s_mL': np.nan,
            'interpolated_times': np.array([]),
            'interpolated_activities': np.array([]),
            'warnings': warnings + ['ERROR: Insufficient data points for interpolation']
        }

    # Create interpolation function
    # Use linear interpolation with extrapolation for boundaries
    interp_func = interp1d(times, activities, kind='linear',
                           bounds_error=False, fill_value='extrapolate')

    # Determine integration range
    t_start = 0.0
    t_end = scan_midpoint_s

    # Check if we need to extrapolate significantly
    if t_end > times[-1]:
        warnings.append(f"INFO: scan_midpoint ({t_end:.0f}s) beyond last sample ({times[-1]:.0f}s) - extrapolating")

    # Create dense time array for integration (1 second resolution)
    n_points = int(t_end - t_start) + 1
    interp_times = np.linspace(t_start, t_end, n_points)
    interp_activities = interp_func(interp_times)

    # Ensure no negative activities (can happen with extrapolation)
    interp_activities = np.maximum(interp_activities, 0)

    # Calculate AUC using trapezoidal rule
    auc = np.trapz(interp_activities, interp_times)

    # QC check on AUC value
    if auc < 1e6:  # Less than 1e6 Bq·s/mL seems too low
        warnings.append(f"WARN: AUC unusually low ({auc:.0f} Bq·s/mL)")

    return {
        'auc_0_to_midpoint_Bq_s_mL': auc,
        'interpolated_times': interp_times,
        'interpolated_activities': interp_activities,
        'scan_midpoint_s': scan_midpoint_s,
        'warnings': warnings
    }


def save_processed_input_function(if_data: Dict[str, Any],
                                   scan_midpoint_s: float,
                                   output_path: Path) -> None:
    """
    Save processed input function to CSV, truncated at scan midpoint.

    Args:
        if_data: Dictionary from load_input_function()
        scan_midpoint_s: Time to truncate at (seconds)
        output_path: Path for output CSV file
    """
    times = if_data['times']
    activities = if_data['activities']

    # Filter to only include times up to scan_midpoint
    mask = times <= scan_midpoint_s
    times_truncated = times[mask]
    activities_truncated = activities[mask]

    # Create DataFrame and save
    df = pd.DataFrame({
        'time_s': times_truncated,
        'activity_Bq_mL': activities_truncated
    })

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)


def calculate_fur(intensity_bq_ml: float, auc_Bq_s_mL: float) -> float:
    """
    Calculate Fractional Uptake Rate (FUR).

    FUR = C_tissue(T) / integral_0^T(C_input(t) dt)

    Raw units: (Bq/mL) / (Bq·s/mL) = s^-1
    Reported in: min^-1 (multiply by 60)

    Args:
        intensity_bq_ml: Tissue activity at measurement time in Bq/mL
        auc_Bq_s_mL: Cumulative AUC from 0 to scan midpoint in Bq·s/mL

    Returns:
        FUR value in min^-1
    """
    if auc_Bq_s_mL <= 0 or np.isnan(auc_Bq_s_mL):
        return np.nan

    fur_per_s = intensity_bq_ml / auc_Bq_s_mL
    return fur_per_s * 60.0  # Convert s^-1 to min^-1
