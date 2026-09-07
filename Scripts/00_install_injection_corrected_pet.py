#!/usr/bin/env python3
"""
Install the injection-corrected static brain PET images into RawData.

Background (adversarial review 2026-09-06, finding F01): the static MoCo PET NIfTIs originally placed
in RawData/ (export of 2025-03-13) are decay-corrected to the *scan start* (sidecar
DecayCorrection = "START"), whereas the blood samples, the aorta IDIF and the cerebellum TAC are
referenced to *injection* (TimeZero). The BIDS export of 2026-02-05 (BIDS_20260205/raw) contains the
same images multiplied by 2^(ScanStart / T_half), i.e. re-referenced to injection, together with the
corrected sidecars. This script:

  1. archives the original NIfTI + JSON of every session under RawData/_archive_START_corrected_pet/,
  2. verifies that the BIDS image has identical geometry and equals the original times
     2^(ScanStart/6586.2) at every voxel,
  3. writes the BIDS image under the existing blinded filename (uncompressed .nii, float32) and replaces
     the adjacent JSON with the corrected sidecar,
  4. writes RawData/pet_manifest.csv with SHA-256 of source, installed and archived files.

Re-runnable: already-archived originals are not overwritten.

Usage:
    python 00_install_injection_corrected_pet.py [--bids-root /path/to/BIDS_20260205/raw]
"""
import argparse
import glob
import hashlib
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

F18_HALF_LIFE_S = 6586.2


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids-root", default="/Users/pontusps/Documents/ERAP_backupdata/BIDS_20260205/raw")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    raw = project_root / "RawData"
    bids = Path(args.bids_root)
    archive = raw / "_archive_START_corrected_pet"

    key = pd.read_csv(project_root / "BlindKey" / "Blinding_key.csv")
    rows = []
    for _, r in key.iterrows():
        sub, tp, code = r["participant_id"], r["Session"], "ses-" + r["Blind.code"]
        pet_dir = raw / sub / code / "pet"
        stem = f"{sub}_{code}_chunk-brain_rec-StaticMoCo_trc-18FFDG_pet"
        nii, js = pet_dir / f"{stem}.nii", pet_dir / f"{stem}.json"
        src_nii = bids / sub / f"ses-{tp}" / "pet" / f"{sub}_ses-{tp}_trc-18FFDG_rec-StaticMoCo_chunk-1_pet.nii.gz"
        src_js = src_nii.with_name(src_nii.name.replace(".nii.gz", ".json"))
        if not (src_nii.exists() and src_js.exists()):
            raise FileNotFoundError(f"{sub}/{tp}: BIDS source missing: {src_nii}")

        adir = archive / sub / code
        adir.mkdir(parents=True, exist_ok=True)
        if not (adir / nii.name).exists():
            shutil.copy2(nii, adir / nii.name)
            shutil.copy2(js, adir / js.name)
        old = nib.load(adir / nii.name)
        old_js = json.load(open(adir / js.name))
        if old_js.get("DecayCorrection") != "START":
            raise RuntimeError(f"{sub}/{tp}: archived original is not START-corrected ({old_js.get('DecayCorrection')})")

        new = nib.load(src_nii)
        new_js = json.load(open(src_js))
        if new_js.get("DecayCorrection") != "INJECTION" or new_js.get("ImageDecayCorrected") is not True:
            raise RuntimeError(f"{sub}/{tp}: BIDS sidecar is not injection-corrected")
        if new.shape != old.shape or not np.allclose(new.affine, old.affine, atol=1e-4):
            raise RuntimeError(f"{sub}/{tp}: geometry differs between BIDS and original")

        a = old.get_fdata(dtype=np.float32)
        b = new.get_fdata(dtype=np.float32)
        expected = 2 ** (new_js["ScanStart"] / F18_HALF_LIFE_S)
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            raise RuntimeError(f"{sub}/{tp}: non-finite voxels")
        zero = a == 0
        if not np.all(b[zero] == 0):
            raise RuntimeError(f"{sub}/{tp}: {int((b[zero] != 0).sum())} zero voxels became non-zero")
        pred = a[~zero] * expected
        if not np.all(np.abs(b[~zero] - pred) <= 1e-4 * np.abs(pred) + 1e-2):
            bad = int((np.abs(b[~zero] - pred) > 1e-4 * np.abs(pred) + 1e-2).sum())
            raise RuntimeError(f"{sub}/{tp}: {bad} voxels deviate from original x {expected:.6f}")
        ratio = b[a > 100] / a[a > 100]

        for mask_path in glob.glob(str(pet_dir / "*mask*.nii*")):
            mk = nib.load(mask_path)
            if mk.shape != new.shape or not np.allclose(mk.affine, new.affine, atol=1e-4):
                raise RuntimeError(f"{sub}/{tp}: mask geometry differs: {mask_path}")

        out = nib.Nifti1Image(b, new.affine, header=new.header)
        out.set_data_dtype(np.float32)
        nib.save(out, nii)
        with open(js, "w") as f:
            json.dump(new_js, f, indent=4)
        chk = nib.load(nii)
        if not np.array_equal(chk.get_fdata(dtype=np.float32), b):
            raise RuntimeError(f"{sub}/{tp}: written file does not round-trip")

        rows.append({
            "subject_id": sub, "session_blinded": code, "session_unblinded": tp,
            "pet_file": str(nii.relative_to(project_root)),
            "pet_sha256": sha256_file(nii),
            "source": str(src_nii), "source_sha256": sha256_file(src_nii),
            "archived_original": str((adir / nii.name).relative_to(project_root)),
            "original_sha256": sha256_file(adir / nii.name),
            "DecayCorrection": new_js["DecayCorrection"], "ScanStart_s": new_js["ScanStart"],
            "ratio_installed_over_original": round(float(np.median(ratio)), 6),
            "expected_2pow_ScanStart_over_Thalf": round(expected, 6),
        })
        print(f"{sub} {tp}: installed (ratio {np.median(ratio):.4f})")

    pd.DataFrame(rows).to_csv(raw / "pet_manifest.csv", index=False)
    print(f"Wrote {raw / 'pet_manifest.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
