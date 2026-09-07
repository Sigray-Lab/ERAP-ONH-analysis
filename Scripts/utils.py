"""
Utility functions for ONH FDG-PET metric extraction.

Revised 2026-09-07 after the adversarial review (Review_adversarial/REVIEW_REPORT.md):
- PET sidecar is read from the file adjacent to the NIfTI and must declare decay correction to
  injection (F01); ScanStart/FrameDuration are validated, not defaulted (F13).
- Anatomical laterality is derived from the mask centroid in world coordinates (F02).
- Input discovery requires unique matches; timepoints are an enumerated mapping (F14).
- Blood samples are sorted and validated before interpolation; non-finite IF values are rejected (F16, F17).
- Mask/PET geometry, binary content and finite values are asserted (F17).
- QC flags are structured records; volume-change flags are computed in a second pass (F07, F15).
- np.trapezoid replaces np.trapz (F19).
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

F18_HALF_LIFE_S = 6586.2

# eCRF PET-visit suffix per timepoint. Any other timepoint label is an error, not "Followup".
TIMEPOINTS = {"Baseline": "1", "Followup": "2"}


class InputError(RuntimeError):
    """A raw-data file is missing, ambiguous or fails validation."""


class AmbiguousInputError(InputError):
    pass


class DecayReferenceError(InputError):
    pass


# ============================================================================
# GENERIC HELPERS
# ============================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _unique(matches: List[Path], what: str) -> Optional[Path]:
    matches = [m for m in matches if "ScalarVolume" not in str(m)]
    if len(matches) == 0:
        return None
    if len(matches) > 1:
        raise AmbiguousInputError(f"{what}: {len(matches)} candidates, expected exactly one: {[str(m) for m in matches]}")
    return matches[0]


def _to_float(value) -> Optional[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return None
    if isinstance(value, str):
        value = value.strip().replace(",", ".")
        if value == "":
            return None
    return float(value)


# ============================================================================
# BLINDING / eCRF / SUV
# ============================================================================

def load_blinding_key(project_root: Path) -> Dict[Tuple[str, str], str]:
    """Map (subject_id, 'ses-<blind code>') -> 'Baseline' | 'Followup'."""
    possible = [project_root / "Blinding_key.csv", project_root / "BlindKey" / "Blinding_key.csv"]
    files = [p for p in possible if p.exists()]
    if not files:
        raise FileNotFoundError(f"Blinding key not found in: {possible}")
    df = pd.read_csv(files[0])
    mapping = {}
    for _, row in df.iterrows():
        tp = row["Session"]
        if tp not in TIMEPOINTS:
            raise InputError(f"Blinding key: unknown Session label {tp!r} for {row['participant_id']}")
        key = (row["participant_id"], f"ses-{row['Blind.code']}")
        if key in mapping:
            raise InputError(f"Blinding key: duplicate entry {key}")
        mapping[key] = tp
    return mapping


def load_ecrf_data(rawdata_dir: Path, filename: Optional[str] = None) -> pd.DataFrame:
    """Load the REDCap export. Exactly one K8ERAPKIH22001_DATA_*.csv must exist unless a filename is given."""
    ecrf_dir = rawdata_dir / "eCRF_data"
    if filename:
        path = ecrf_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"eCRF data not found at: {path}")
    else:
        path = _unique(sorted(ecrf_dir.glob("K8ERAPKIH22001_DATA_*.csv")), "eCRF export")
        if path is None:
            raise FileNotFoundError(f"No eCRF export (K8ERAPKIH22001_DATA_*.csv) in {ecrf_dir}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.attrs["source_path"] = str(path)
    return df


def get_suv_parameters(ecrf_df: pd.DataFrame, subject_id: str, timepoint: str) -> Dict[str, Optional[float]]:
    """Body weight (kg) and injected activity (MBq) on the PET day."""
    if timepoint not in TIMEPOINTS:
        raise InputError(f"Unknown timepoint {timepoint!r}; expected one of {list(TIMEPOINTS)}")
    visit = TIMEPOINTS[timepoint]
    subj_num = int(subject_id.replace("sub-", ""))
    rows = ecrf_df[ecrf_df["subject_id"] == subj_num]
    if len(rows) != 1:
        return {"weight_kg": None, "injected_mbq": None, "error": f"{len(rows)} eCRF rows for subject {subj_num}"}
    row = rows.iloc[0]
    return {
        "weight_kg": _to_float(row.get(f"weight_kg_pet_{visit}")),
        "injected_mbq": _to_float(row.get(f"injected_mbq_pet_{visit}")),
    }


def calculate_suv_scaler(weight_kg: float, injected_mbq: float) -> float:
    """SUV = C[Bq/mL] * weight[g] / dose[Bq] = C * weight_kg / (injected_mbq * 1000)."""
    return weight_kg / (injected_mbq * 1000.0)


def convert_to_suv(pet_value_bq_ml: float, suv_scaler: float) -> float:
    return pet_value_bq_ml * suv_scaler


# ============================================================================
# FILE DISCOVERY
# ============================================================================

def discover_subjects(rawdata_dir: Path) -> List[str]:
    return sorted(p.name for p in rawdata_dir.iterdir()
                  if p.is_dir() and p.name.startswith("sub-") and "ScalarVolume" not in p.name)


def discover_sessions(subject_dir: Path) -> List[str]:
    return sorted(p.name for p in subject_dir.iterdir()
                  if p.is_dir() and p.name.startswith("ses-") and "ScalarVolume" not in p.name)


def find_pet_file(pet_dir: Path, subject_id: str, session_id: str) -> Optional[Path]:
    stem = f"{subject_id}_{session_id}_chunk-brain_rec-StaticMoCo_trc-18FFDG_pet"
    exact = [pet_dir / f"{stem}.nii", pet_dir / f"{stem}.nii.gz"]
    found = _unique([p for p in exact if p.exists()], f"{subject_id}/{session_id} PET")
    if found is not None:
        return found
    fallback = [p for p in pet_dir.glob("*_pet.nii*") if "mask" not in p.name.lower()]
    return _unique(fallback, f"{subject_id}/{session_id} PET (fallback glob)")


def find_pet_json(pet_file: Path) -> Optional[Path]:
    """The sidecar adjacent to the NIfTI (same stem). No other location is consulted for quantification."""
    name = pet_file.name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            json_path = pet_file.with_name(name[: -len(suffix)] + ".json")
            return json_path if json_path.exists() else None
    return None


def load_pet_json(json_path: Path) -> Dict[str, Any]:
    """
    Load and validate the PET sidecar.

    Requirements (all raise on failure, nothing is defaulted):
      DecayCorrection == "INJECTION" and ImageDecayCorrected == true  (F01)
      Units == "Bq/mL"
      ScanStart present, 600-7200 s post-injection
      FrameDuration a single value in milliseconds within 1 790 000-1 810 000 (local sidecar convention;
      converted to seconds)
    """
    with open(json_path) as f:
        data = json.load(f)

    dc = data.get("DecayCorrection")
    idc = data.get("ImageDecayCorrected")
    if dc != "INJECTION" or idc is not True:
        raise DecayReferenceError(
            f"{json_path}: DecayCorrection={dc!r}, ImageDecayCorrected={idc!r}. The pipeline requires images "
            f"decay-corrected to injection (TimeZero) so that SUV, SUVR, TPR and FUR share one reference with the "
            f"blood, IDIF and cerebellum data. Run Scripts/00_install_injection_corrected_pet.py.")

    units = data.get("Units")
    if units != "Bq/mL":
        raise InputError(f"{json_path}: Units={units!r}, expected 'Bq/mL'")

    scan_start = data.get("ScanStart")
    if scan_start is None or not (600 <= float(scan_start) <= 7200):
        raise InputError(f"{json_path}: ScanStart={scan_start!r} is missing or outside 600-7200 s")

    fd = data.get("FrameDuration")
    if isinstance(fd, list):
        if len(fd) != 1:
            raise InputError(f"{json_path}: FrameDuration has {len(fd)} entries, expected one static frame")
        fd = fd[0]
    if fd is None or not (1_790_000 <= float(fd) <= 1_810_000):
        raise InputError(f"{json_path}: FrameDuration={fd!r}; expected ~1 800 000 ms (30-min static frame)")

    return {
        "scan_start_s": float(scan_start),
        "scan_duration_s": float(fd) / 1000.0,
        "time_zero": data.get("TimeZero"),
        "scan_start_time": data.get("ScanStartTime"),
        "units": units,
        "decay_correction": dc,
        "decay_correction_factor": (data.get("DecayCorrectionFactor") or [None])[0],
        "radionuclide_total_dose_bq": data.get("RadionuclideTotalDose"),
        "json_file": str(json_path),
    }


def check_updated_sidecar(pet_timing: Dict[str, Any], rawdata_dir: Path, subject_id: str, timepoint: str) -> List[str]:
    """Consistency check against RawData/json_side_cars_updated (if present). Returns list of mismatches."""
    path = rawdata_dir / "json_side_cars_updated" / f"{subject_id}_ses-{timepoint}_trc-18FFDG_rec-StaticMoCo_chunk-1_pet.json"
    if not path.exists():
        return []
    with open(path) as f:
        ref = json.load(f)
    mismatches = []
    ref_ss = ref.get("ScanStart")
    if ref_ss is None or not np.isfinite(float(ref_ss)) or float(ref_ss) != pet_timing["scan_start_s"]:
        mismatches.append(f"ScanStart {pet_timing['scan_start_s']} vs updated sidecar {ref_ss}")
    ref_fd = ref.get("FrameDuration")
    ref_fd = ref_fd[0] if isinstance(ref_fd, list) and len(ref_fd) == 1 else (None if isinstance(ref_fd, list) else ref_fd)
    if ref_fd is None or not np.isfinite(float(ref_fd)) or abs(float(ref_fd) / 1000.0 - pet_timing["scan_duration_s"]) > 1e-6:
        mismatches.append(f"FrameDuration {pet_timing['scan_duration_s']} s vs updated sidecar {ref.get('FrameDuration')}")
    if ref.get("DecayCorrection") != pet_timing["decay_correction"]:
        mismatches.append(f"DecayCorrection {pet_timing['decay_correction']} vs updated sidecar {ref.get('DecayCorrection')}")
    return mismatches


def find_mask_file(pet_dir: Path, subject_id: str, session_id: str, eye_label: str) -> Optional[Path]:
    """
    Find the mask file whose *filename* carries `eye_label` ('left'/'right'). The filename label is a display
    convention; anatomical laterality is derived separately (mask_physical_eye).
    """
    base = f"{subject_id}_{session_id}_chunk-brain_rec-StaticMoCo_trc-18FFDG_pet"
    patterns = [f"{base}_{eye_label}_ONH_mask.nii.gz", f"{base}_{eye_label}_ONH_mask.nii",
                f"{base}_{eye_label}_OHN_mask.nii.gz", f"{base}_{eye_label}_OHN_mask.nii",
                f"{base}_{eye_label}_mask.nii.gz", f"{base}_{eye_label}_mask.nii"]
    found = _unique([pet_dir / p for p in patterns if (pet_dir / p).exists()], f"{subject_id}/{session_id} {eye_label} mask")
    if found is not None:
        return found
    return _unique(list(pet_dir.glob(f"*{eye_label}*mask*")), f"{subject_id}/{session_id} {eye_label} mask (fallback glob)")


# ============================================================================
# NIfTI
# ============================================================================

def load_nifti_with_scaling(filepath: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(filepath)
    data = img.get_fdata(dtype=np.float32)  # applies scl_slope / scl_inter
    return data, img


def get_voxel_dimensions(img: nib.Nifti1Image) -> np.ndarray:
    return np.array(img.header.get_zooms()[:3])


def validate_mask(mask_data: np.ndarray, mask_img: nib.Nifti1Image,
                  pet_data: np.ndarray, pet_img: nib.Nifti1Image, name: str) -> None:
    """Same grid, same affine, binary content, non-empty, finite PET inside the mask."""
    if mask_data.shape != pet_data.shape:
        raise InputError(f"{name}: shape {mask_data.shape} != PET {pet_data.shape}")
    if not np.allclose(mask_img.affine, pet_img.affine, atol=1e-3):
        raise InputError(f"{name}: affine differs from PET affine")
    vals = np.unique(mask_data)
    if not np.all(np.isin(vals, [0.0, 1.0])):
        raise InputError(f"{name}: mask is not binary (values {vals[:10]})")
    n = int((mask_data > 0).sum())
    if n == 0:
        raise InputError(f"{name}: empty mask")
    if not np.all(np.isfinite(pet_data[mask_data > 0])):
        raise InputError(f"{name}: non-finite PET values inside mask")


def mask_physical_eye(mask_data: np.ndarray, affine: np.ndarray) -> Tuple[str, np.ndarray]:
    """
    Anatomical side of a mask from its centroid in world (RAS) coordinates: x > 0 is the subject's right.
    Returns ('left' | 'right', centroid_world_mm).
    """
    idx = np.argwhere(mask_data > 0)
    centroid_vox = idx.mean(axis=0)
    world = affine[:3, :3] @ centroid_vox + affine[:3, 3]
    side = "right" if world[0] > 0 else "left"
    return side, world


# ============================================================================
# METRICS
# ============================================================================

def create_sphere_mask(center: Tuple[int, int, int], radius_mm: float,
                       voxel_dims: np.ndarray, image_shape: Tuple[int, int, int]) -> np.ndarray:
    cx, cy, cz = center
    radius_vox = radius_mm / voxel_dims
    rng = [int(np.ceil(r)) + 1 for r in radius_vox]
    mask = np.zeros(image_shape, dtype=bool)
    for dx in range(-rng[0], rng[0] + 1):
        for dy in range(-rng[1], rng[1] + 1):
            for dz in range(-rng[2], rng[2] + 1):
                x, y, z = cx + dx, cy + dy, cz + dz
                if 0 <= x < image_shape[0] and 0 <= y < image_shape[1] and 0 <= z < image_shape[2]:
                    dist = np.sqrt((dx * voxel_dims[0]) ** 2 + (dy * voxel_dims[1]) ** 2 + (dz * voxel_dims[2]) ** 2)
                    if dist <= radius_mm:
                        mask[x, y, z] = True
    return mask


def calculate_metrics(pet_data: np.ndarray, mask_data: np.ndarray, voxel_dims: np.ndarray,
                      sphere_radius_mm: float = 2.0, top_n: int = 150) -> Dict:
    """
    Raw-intensity metrics within a mask: max, peak (mean of non-zero voxels in a sphere on the max voxel),
    and mean/median/p90 of the top_n hottest mask voxels (falls back to all voxels if the mask is smaller).
    Note: hottest-N statistics are monotone non-decreasing in the mask support (a superset can only add
    hotter voxels); they are *weakly*, not zero, dependent on delineation. See sensitivity_analysis.py.
    """
    mask_indices = np.where(mask_data > 0)
    n_mask = len(mask_indices[0])
    if n_mask == 0:
        return {"error": "Empty mask", "mask_volume_voxels": 0, "mask_volume_mm3": 0}

    voxel_volume = float(np.prod(voxel_dims))
    values = pet_data[mask_indices]
    intensity_max = float(np.max(values))
    max_idx = int(np.argmax(values))
    max_coords = (int(mask_indices[0][max_idx]), int(mask_indices[1][max_idx]), int(mask_indices[2][max_idx]))

    sphere = create_sphere_mask(max_coords, sphere_radius_mm, voxel_dims, pet_data.shape)
    sphere_values = pet_data[sphere]
    nonzero = sphere_values[sphere_values > 0]
    sphere_total = int(sphere.sum())
    sphere_nonzero = int(len(nonzero))
    intensity_peak = float(np.mean(nonzero)) if sphere_nonzero > 0 else intensity_max
    zero_pct = (sphere_total - sphere_nonzero) / sphere_total * 100 if sphere_total > 0 else 0.0

    sorted_values = np.sort(values)[::-1]
    top = sorted_values[:top_n] if len(sorted_values) >= top_n else sorted_values

    return {
        "intensity_max": intensity_max,
        "intensity_peak": intensity_peak,
        "intensity_top150_mean": float(np.mean(top)),
        "intensity_top150_median": float(np.median(top)),
        "intensity_top150_p90": float(np.percentile(top, 90)),
        "top_n_used": int(len(top)),
        "mask_volume_voxels": int(n_mask),
        "mask_volume_mm3": float(n_mask * voxel_volume),
        "max_voxel_x": max_coords[0], "max_voxel_y": max_coords[1], "max_voxel_z": max_coords[2],
        "sphere_voxel_count": sphere_nonzero,
        "sphere_total_voxels": sphere_total,
        "sphere_zero_percentage": float(zero_pct),
    }


# ============================================================================
# CEREBELLUM / SUVR
# ============================================================================

def find_cerebellum_tac(rawdata_dir: Path, subject_id: str, timepoint: str) -> Optional[Path]:
    p = rawdata_dir / "Cerebellum_tacs" / f"{subject_id}_ses-{timepoint}_label-cerebellum_tacs.tsv"
    return p if p.exists() else None


def load_cerebellum_tac(tac_path: Path, scan_start_s: Optional[float] = None,
                        scan_duration_s: Optional[float] = None, tolerance_s: float = 2.0) -> Dict[str, Any]:
    """Time-weighted mean of the cerebellum TAC; frames must cover the static scan window when timing is given."""
    df = pd.read_csv(tac_path, sep="\t")
    means = df["Mean(Bq/mL)"].to_numpy(float)
    durations = df["FrameDuration(s)"].to_numpy(float)
    starts = df["FrameStart(s)"].to_numpy(float)
    if len(df) == 0 or not np.all(np.isfinite(means)) or not np.all(np.isfinite(starts)) \
            or not np.all(np.isfinite(durations)) or np.any(durations <= 0):
        raise InputError(f"{tac_path}: empty TAC, non-finite values, or non-positive frame durations")
    if len(df) > 1 and not np.allclose(starts[1:], starts[:-1] + durations[:-1], rtol=0, atol=tolerance_s):
        raise InputError(f"{tac_path}: frames are not contiguous")
    total = float(durations.sum())
    if scan_start_s is not None and abs(starts[0] - scan_start_s) > tolerance_s:
        raise InputError(f"{tac_path}: first frame starts at {starts[0]} s, static scan at {scan_start_s} s")
    if scan_duration_s is not None and abs(total - scan_duration_s) > tolerance_s:
        raise InputError(f"{tac_path}: frames cover {total} s, static scan lasts {scan_duration_s} s")
    auc = float(np.sum(means * durations))
    return {"cerebellum_mean_bq_ml": auc / total, "cerebellum_auc_bq_s_ml": auc,
            "cerebellum_total_duration_s": total, "cerebellum_n_frames": int(len(df)),
            "cerebellum_first_frame_start_s": float(starts[0])}


def calculate_suvr(pet_value_bq_ml: float, cerebellum_mean_bq_ml: float) -> float:
    if cerebellum_mean_bq_ml is None or cerebellum_mean_bq_ml <= 0:
        return np.nan
    return pet_value_bq_ml / cerebellum_mean_bq_ml


# ============================================================================
# BLOOD PLASMA / TPR
# ============================================================================

def find_blood_file(rawdata_dir: Path, subject_id: str, timepoint: str) -> Tuple[Optional[Path], Optional[Path]]:
    d = rawdata_dir / "BloodPlasma"
    tsv = d / f"{subject_id}_ses-{timepoint}_recording-manual_blood.tsv"
    js = d / f"{subject_id}_ses-{timepoint}_recording-manual_blood.json"
    return (tsv if tsv.exists() else None, js if js.exists() else None)


def load_blood_data(tsv_path: Path, json_path: Optional[Path] = None) -> Dict[str, Any]:
    """Manual samples (kBq/mL), sorted by time; NA plasma rows are dropped and reported."""
    df = pd.read_csv(tsv_path, sep="\t")
    for col in ("time", "whole_blood_radioactivity", "plasma_radioactivity"):
        if col not in df.columns:
            raise InputError(f"{tsv_path}: missing column {col}")
    df = df.sort_values("time").reset_index(drop=True)
    times_all = df["time"].to_numpy(float)
    if not np.all(np.isfinite(times_all)) or len(np.unique(times_all)) != len(times_all):
        raise InputError(f"{tsv_path}: sample times must be finite and unique")
    valid = df["plasma_radioactivity"].notna().to_numpy()
    if not np.all(np.isfinite(df.loc[valid, "plasma_radioactivity"].to_numpy(float))):
        raise InputError(f"{tsv_path}: non-finite plasma activity")
    if np.any(df.loc[valid, "plasma_radioactivity"].to_numpy(float) < 0):
        raise InputError(f"{tsv_path}: negative plasma activity")
    result = {
        "time_s": times_all[valid],
        "plasma_kbq_ml": df.loc[valid, "plasma_radioactivity"].to_numpy(float),
        "whole_blood_kbq_ml": df["whole_blood_radioactivity"].to_numpy(float),
        "whole_blood_time_s": times_all,
        "n_samples": int(valid.sum()),
        "n_samples_total": int(len(df)),
        "n_samples_missing": int((~valid).sum()),
        "source_file": str(tsv_path),
    }
    if result["n_samples_missing"] > 0:
        result["missing_plasma_times"] = times_all[~valid].tolist()
    if json_path is not None and json_path.exists():
        with open(json_path) as f:
            result["metadata"] = json.load(f)
    return result


def interpolate_at_time(times: np.ndarray, values: np.ndarray, target_time: float) -> float:
    """Linear interpolation; flat (constant) extrapolation outside the sampled range."""
    if target_time <= times[0]:
        return float(values[0])
    if target_time >= times[-1]:
        return float(values[-1])
    return float(np.interp(target_time, times, values))


def calculate_plasma_auc(blood_data: Dict, scan_start_s: float, scan_end_s: float) -> Dict[str, Any]:
    """Trapezoidal plasma AUC and mean over [scan_start, scan_end], with interpolated window endpoints."""
    times = np.asarray(blood_data["time_s"], float)
    plasma = np.asarray(blood_data["plasma_kbq_ml"], float)
    order = np.argsort(times)
    times, plasma = times[order], plasma[order]
    warnings: List[str] = []
    if blood_data.get("n_samples_missing", 0) > 0:
        warnings.append(f"INFO: {blood_data['n_samples_missing']} plasma sample(s) with NA values were skipped "
                        f"(times: {blood_data.get('missing_plasma_times', [])})")
    if len(times) < 2:
        warnings.append("ERROR: Insufficient valid plasma samples for AUC calculation")
        return {"plasma_auc_kbq_s_ml": np.nan, "plasma_mean_kbq_ml": np.nan, "plasma_samples_in_window": 0,
                "interpolated_start_kbq_ml": np.nan, "interpolated_end_kbq_ml": np.nan, "warnings": warnings}
    if np.sum(times <= scan_start_s) == 0:
        warnings.append("WARN: No plasma samples before scan start - flat extrapolation")
    if np.sum(times >= scan_end_s) == 0:
        warnings.append("WARN: No plasma samples after scan end - flat extrapolation")

    inside = (times > scan_start_s) & (times < scan_end_s)
    wt = np.concatenate([[scan_start_s], times[inside], [scan_end_s]])
    wv = np.concatenate([[interpolate_at_time(times, plasma, scan_start_s)], plasma[inside],
                         [interpolate_at_time(times, plasma, scan_end_s)]])
    auc = float(np.trapezoid(wv, wt))
    duration = scan_end_s - scan_start_s
    mean_plasma = auc / duration if duration > 0 else np.nan
    n_in = int(inside.sum())
    if n_in < 2:
        warnings.append(f"WARN: Only {n_in} plasma samples in scan window - interpolation may be unreliable")
    if np.any(plasma < 0.5) or np.any(plasma > 50):
        warnings.append("WARN: Plasma values outside typical range (0.5-50 kBq/mL)")
    return {"plasma_auc_kbq_s_ml": auc, "plasma_mean_kbq_ml": mean_plasma, "plasma_samples_in_window": n_in,
            "interpolated_start_kbq_ml": float(wv[0]), "interpolated_end_kbq_ml": float(wv[-1]), "warnings": warnings}


def calculate_plasma_total_auc(blood_data: Dict) -> float:
    times = np.asarray(blood_data["time_s"], float)
    plasma = np.asarray(blood_data["plasma_kbq_ml"], float)
    order = np.argsort(times)
    return float(np.trapezoid(plasma[order], times[order]))


def calculate_tpr(pet_value_bq_ml: float, plasma_mean_kbq_ml: float) -> float:
    if plasma_mean_kbq_ml is None or np.isnan(plasma_mean_kbq_ml) or plasma_mean_kbq_ml <= 0:
        return np.nan
    return (pet_value_bq_ml / 1000.0) / plasma_mean_kbq_ml


# ============================================================================
# INPUT FUNCTION / FUR
# ============================================================================

def find_input_function_file(rawdata_dir: Path, subject_id: str, timepoint: str) -> Optional[Path]:
    p = rawdata_dir / "InputFunctions" / f"{subject_id}_ses-{timepoint}_desc-IF_tacs.tsv"
    return p if p.exists() else None


def load_input_function(if_path: Path) -> Dict[str, Any]:
    """Aorta IDIF (early) + venous plasma (late), Bq/mL, sorted by time; negative times dropped; NaN rejected."""
    df = pd.read_csv(if_path, sep="\t")
    warnings: List[str] = []
    sel = df[df["ROI"].isin(["aorta", "plasma"])].copy()
    n_idif = int((sel["ROI"] == "aorta").sum())
    n_plasma = int((sel["ROI"] == "plasma").sum())
    if not np.all(np.isfinite(sel["Time(s)"].to_numpy(float))) or not np.all(np.isfinite(sel["Radioactivity(Bq/mL)"].to_numpy(float))):
        raise InputError(f"{if_path}: non-finite time or activity in the input function")
    sel = sel[sel["Time(s)"] >= 0].sort_values("Time(s)")
    times = sel["Time(s)"].to_numpy(float)
    acts = sel["Radioactivity(Bq/mL)"].to_numpy(float)
    if len(times) < 2 or not np.all(np.diff(times) > 0):
        raise InputError(f"{if_path}: input-function times must be unique and strictly increasing (after dropping t<0)")
    if np.any(acts < 0):
        raise InputError(f"{if_path}: negative activity in the input function")
    if n_idif < 5:
        warnings.append(f"WARN: Only {n_idif} IDIF (aorta) samples (expected >=5)")
    if n_plasma < 2:
        warnings.append(f"WARN: Only {n_plasma} plasma samples (expected >=2)")
    if n_idif > 0:
        peak = float(sel.loc[sel["ROI"] == "aorta", "Radioactivity(Bq/mL)"].max())
        if peak < 10000:
            warnings.append(f"WARN: IDIF peak unusually low ({peak:.0f} Bq/mL)")
    gap = None
    if n_idif > 0 and n_plasma > 0:
        gap = float(sel.loc[sel["ROI"] == "plasma", "Time(s)"].min() - sel.loc[sel["ROI"] == "aorta", "Time(s)"].max())
        if gap > 800:
            warnings.append(f"INFO: Large gap between IDIF and plasma ({gap:.0f}s) - will interpolate")
    return {"times": times, "activities": acts, "roi": sel["ROI"].to_numpy(), "n_idif_samples": n_idif,
            "n_plasma_samples": n_plasma, "idif_plasma_gap_s": gap, "warnings": warnings, "source_file": str(if_path)}


def calculate_input_function_auc(if_data: Dict[str, Any], scan_midpoint_s: float) -> Dict[str, Any]:
    """AUC of the linearly interpolated input function from t=0 to scan midpoint (1 s grid, trapezoid)."""
    from scipy.interpolate import interp1d
    times, acts = if_data["times"], if_data["activities"]
    warnings = list(if_data.get("warnings", []))
    if len(times) < 2:
        return {"auc_0_to_midpoint_Bq_s_mL": np.nan, "interpolated_times": np.array([]),
                "interpolated_activities": np.array([]), "warnings": warnings + ["ERROR: Insufficient data points"]}
    f = interp1d(times, acts, kind="linear", bounds_error=False, fill_value="extrapolate")
    if scan_midpoint_s > times[-1]:
        warnings.append(f"INFO: scan midpoint ({scan_midpoint_s:.0f}s) beyond last sample ({times[-1]:.0f}s) - extrapolating")
    n = int(scan_midpoint_s) + 1
    grid = np.linspace(0.0, scan_midpoint_s, n)
    vals = np.maximum(f(grid), 0.0)
    auc = float(np.trapezoid(vals, grid))
    if auc < 1e6:
        warnings.append(f"WARN: AUC unusually low ({auc:.0f} Bq*s/mL)")
    return {"auc_0_to_midpoint_Bq_s_mL": auc, "interpolated_times": grid, "interpolated_activities": vals,
            "scan_midpoint_s": scan_midpoint_s, "warnings": warnings}


def save_processed_input_function(if_data: Dict[str, Any], auc_result: Dict[str, Any], output_path: Path) -> None:
    """Always (re)write: observed knots up to the midpoint plus the interpolated t=0 and midpoint endpoints."""
    mid = auc_result["scan_midpoint_s"]
    grid, vals = auc_result["interpolated_times"], auc_result["interpolated_activities"]
    keep = if_data["times"] <= mid
    obs_times = set(float(t) for t in if_data["times"][keep])
    rows = [{"time_s": float(t), "activity_Bq_mL": float(a),
             "source": str(r) + ("_endpoint" if float(t) in (0.0, float(mid)) else "")}
            for t, a, r in zip(if_data["times"][keep], if_data["activities"][keep], if_data["roi"][keep])]
    if 0.0 not in obs_times:
        rows.append({"time_s": 0.0, "activity_Bq_mL": float(vals[0]), "source": "interpolated_endpoint"})
    if float(mid) not in obs_times:
        rows.append({"time_s": float(mid), "activity_Bq_mL": float(vals[-1]), "source": "interpolated_endpoint"})
    df = pd.DataFrame(rows).sort_values("time_s")
    assert df["time_s"].is_unique
    df.attrs["auc"] = auc_result["auc_0_to_midpoint_Bq_s_mL"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"# AUC_0_to_midpoint_Bq_s_mL={auc_result['auc_0_to_midpoint_Bq_s_mL']:.6f}; "
                f"scan_midpoint_s={mid}; source={if_data.get('source_file')}; "
                f"contract=linear interpolation of the knots on a 1 s grid from 0 to the midpoint, trapezoid; "
                f"these rows are the observed knots plus endpoints, not the integration grid\n")
        df.to_csv(f, index=False)


def calculate_fur(intensity_bq_ml: float, auc_Bq_s_mL: float) -> float:
    """FUR = C_tissue(frame mean) / int_0^midpoint C_input dt, in min^-1."""
    if auc_Bq_s_mL is None or np.isnan(auc_Bq_s_mL) or auc_Bq_s_mL <= 0:
        return np.nan
    return intensity_bq_ml / auc_Bq_s_mL * 60.0


# ============================================================================
# QC
# ============================================================================

def generate_qc_flags(metrics: Dict, suv_max: Optional[float] = None) -> List[Dict[str, Any]]:
    """Per-eye QC flags as structured records (category, severity, description, value, threshold, recommendation)."""
    flags: List[Dict[str, Any]] = []
    vol = metrics.get("mask_volume_voxels", 0)
    if vol < 5:
        flags.append({"flag_category": "mask_volume", "severity": "WARNING", "value": vol, "threshold": "<5",
                      "flag_description": f"Mask volume suspiciously small ({vol} voxels)",
                      "recommendation": "Review mask delineation"})
    if vol > 500:
        flags.append({"flag_category": "mask_volume", "severity": "WARNING", "value": vol, "threshold": ">500",
                      "flag_description": f"Mask volume suspiciously large ({vol} voxels) - may include non-ONH tissue",
                      "recommendation": "Review mask delineation"})
    if metrics.get("top_n_used", 150) < 150:
        flags.append({"flag_category": "mask_volume", "severity": "WARNING", "value": metrics["top_n_used"], "threshold": "<150",
                      "flag_description": f"Mask has fewer than 150 voxels; Top-150 uses all {metrics['top_n_used']}",
                      "recommendation": "Review mask delineation"})
    if suv_max is not None and not np.isnan(suv_max):
        if suv_max < 0.5:
            flags.append({"flag_category": "suv_value", "severity": "WARNING", "value": round(suv_max, 3), "threshold": "<0.5",
                          "flag_description": f"SUVmax unusually low ({suv_max:.2f})", "recommendation": "Review PET data and mask placement"})
        if suv_max > 30:
            flags.append({"flag_category": "suv_value", "severity": "WARNING", "value": round(suv_max, 3), "threshold": ">30",
                          "flag_description": f"SUVmax unusually high ({suv_max:.2f}) - may include extraocular muscle",
                          "recommendation": "Review PET data and mask placement"})
    zp = metrics.get("sphere_zero_percentage", 0)
    if zp > 50:
        flags.append({"flag_category": "sphere_zeros", "severity": "WARNING", "value": round(zp, 1), "threshold": ">50%",
                      "flag_description": f"Sphere contains >50% zeros ({zp:.1f}%) - poor localization",
                      "recommendation": "Review max-voxel location"})
    return flags


def volume_change_flags(df: pd.DataFrame, threshold_pct: float = 25.0) -> List[Dict[str, Any]]:
    """Second-pass QC: |Followup - Baseline| mask volume > threshold, per (subject, physical eye)."""
    flags = []
    for (subj, eye), g in df.groupby(["subject_id", "eye"]):
        b = g[g["session_unblinded"] == "Baseline"]
        f = g[g["session_unblinded"] == "Followup"]
        if len(b) != 1 or len(f) != 1:
            continue
        vb, vf = float(b["mask_volume_voxels"].iloc[0]), float(f["mask_volume_voxels"].iloc[0])
        if vb <= 0:
            continue
        pct = (vf - vb) / vb * 100.0
        if abs(pct) > threshold_pct:
            flags.append({"subject_id": subj, "session": "Followup", "eye": eye,
                          "mask_label_in_filename": f["mask_label_in_filename"].iloc[0],
                          "flag_category": "volume_change", "severity": "WARNING",
                          "flag_description": f"Mask volume differs >{threshold_pct:.0f}% from baseline ({pct:+.1f}%: {vb:.0f} -> {vf:.0f} voxels)",
                          "value": f"{pct:+.1f}%", "threshold": f">{threshold_pct:.0f}%",
                          "recommendation": "Verify delineation consistency"})
    return flags
