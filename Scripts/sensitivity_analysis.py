#!/usr/bin/env python3
"""
Sensitivity analysis for the ONH Top-150 endpoints (supplementary table).

Variants (each recomputes the affected quantity per session and repeats the paired tests):
  as_run                      the pipeline result (Outputs/ONH_FDG_metrics.csv)
  scanner_dose                SUV with the scanner-recorded RadionuclideTotalDose instead of the eCRF dose
                              (4 sessions differ by 2-10 %; source of truth pending the radiopharmacy log)
  sub110_followup_ecrf_times  FUR with the sub-110 Follow-up plasma samples placed at the eCRF times
                              (the TSV times differ by 96-870 s; provenance pending the study binder)
  mask_erode_1mm / mask_dilate_1mm
                              Top-150 statistics after a 6-connected 1 mm erosion / dilation of every mask
                              (hottest-N statistics are monotone in the mask support; this bounds the effect)
  exponential_bridge          FUR with the IDIF-to-plasma gap bridged by an exponential instead of a chord

Writes Outputs/sensitivity_table.csv (all rows) and Outputs/sensitivity_table.md (p and dz overview).
Usage: python sensitivity_analysis.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))
from statistical_analysis import LATERALITIES, fmt_p, paired_analysis, paired_frames  # noqa: E402
from utils import (calculate_input_function_auc, calculate_metrics, find_input_function_file, find_mask_file,  # noqa: E402
                   find_pet_file, get_voxel_dimensions, load_ecrf_data, load_input_function, load_nifti_with_scaling,
                   mask_physical_eye)

OUTCOMES = [("SUVtop150_mean", "SUV_top150_mean"), ("SUVtop150_median", "SUV_top150_median"),
            ("FUR_top150_mean", "FUR_top150_mean"), ("FUR_top150_median", "FUR_top150_median")]


def ecrf_time_to_s(v) -> float:
    s = str(v).strip()
    if ":" in s:
        m, sec = s.split(":")
        return int(m) * 60 + int(sec)
    return float(s.replace(",", ".")) * 60.0


def stats_for(df: pd.DataFrame, version: str, note: str) -> list:
    cols = [c for c, _ in OUTCOMES]
    frames = paired_frames(df, cols)
    rows = []
    for lat in LATERALITIES:
        b, f = frames[lat]
        for col, label in OUTCOMES:
            bv, fv = b[col].to_numpy(float), f[col].to_numpy(float)
            ok = np.isfinite(bv) & np.isfinite(fv)
            r = paired_analysis(bv[ok], fv[ok])
            rows.append({"version": version, "Outcome": label, "Eye": lat.capitalize(), "n": r["n"],
                         "baseline_mean": r["baseline_mean"], "followup_mean": r["followup_mean"], "delta_mean": r["delta_mean"],
                         "delta_ci_low": r["delta_ci_low"], "delta_ci_high": r["delta_ci_high"], "percent_change": r["percent_change"],
                         "cohens_dz": r["cohens_dz"], "p_value": r["p_value"], "wilcoxon_p": r["wilcoxon_p"], "note": note})
    return rows


def main():
    analysis_dir = script_dir.parent
    project_root = analysis_dir.parent
    rawdata_dir = project_root / "RawData"
    out_dir = analysis_dir / "Outputs"
    df = pd.read_csv(out_dir / "ONH_FDG_metrics.csv")
    sf = pd.read_csv(analysis_dir / "DerivedData" / "session_scaling_factors.csv")
    key = df.drop_duplicates(["subject_id", "session_unblinded"])[["subject_id", "session_blinded", "session_unblinded"]]
    rows = stats_for(df, "as_run", "pipeline result")
    manifest = {"versions": {}}

    # (b) scanner dose -----------------------------------------------------------------------------------
    d = df.merge(sf[["subject_id", "session_unblinded", "injected_MBq", "scanner_RadionuclideTotalDose_MBq"]], on=["subject_id", "session_unblinded"])
    factor = d["injected_MBq"] / d["scanner_RadionuclideTotalDose_MBq"]
    v = df.copy()
    for c in ("SUVtop150_mean", "SUVtop150_median"):
        v[c] = d[c] * factor
    changed = d.loc[np.abs(factor - 1) > 0.01, ["subject_id", "session_unblinded", "injected_MBq", "scanner_RadionuclideTotalDose_MBq"]].drop_duplicates()
    manifest["versions"]["scanner_dose"] = changed.to_dict("records")
    rows += stats_for(v, "scanner_dose", f"SUV x eCRF/scanner dose; {len(changed)} sessions differ >1%")

    # (c) sub-110 Follow-up plasma at eCRF times ---------------------------------------------------------------
    ecrf = load_ecrf_data(rawdata_dir)
    r110 = ecrf[ecrf["subject_id"] == 110].iloc[0]
    ecrf_times = [ecrf_time_to_s(r110[f"time_blood_samp{i}_pet_2"]) for i in range(1, 6)]
    if_file = find_input_function_file(rawdata_dir, "sub-110", "Followup")
    if_data = load_input_function(if_file)
    mid = float(df.loc[(df.subject_id == "sub-110") & (df.session_unblinded == "Followup"), "scan_midpoint_s"].iloc[0])
    auc_run = calculate_input_function_auc(if_data, mid)["auc_0_to_midpoint_Bq_s_mL"]
    alt = dict(if_data)
    t = if_data["times"].copy(); roi = if_data["roi"]
    plasma_idx = np.where(roi == "plasma")[0]
    assert len(plasma_idx) == 5
    t[plasma_idx] = ecrf_times
    order = np.argsort(t)
    alt["times"], alt["activities"], alt["roi"] = t[order], if_data["activities"][order], roi[order]
    auc_alt = calculate_input_function_auc(alt, mid)["auc_0_to_midpoint_Bq_s_mL"]
    v = df.copy()
    sel = (v.subject_id == "sub-110") & (v.session_unblinded == "Followup")
    for c in ("FUR_top150_mean", "FUR_top150_median"):
        v.loc[sel, c] = v.loc[sel, c] * auc_run / auc_alt
    manifest["versions"]["sub110_followup_ecrf_times"] = {"tsv_times_s": if_data["times"][plasma_idx].tolist(), "ecrf_times_s": ecrf_times,
                                                          "IF_AUC_run": auc_run, "IF_AUC_ecrf_times": auc_alt}
    rows += stats_for(v, "sub110_followup_ecrf_times", f"sub-110 FU IF AUC {auc_run:.4e} -> {auc_alt:.4e}")

    # (e) exponential bridge across the IDIF-plasma gap, all sessions ------------------------------------------------
    v = df.copy()
    bridge = {}
    for _, k in key.iterrows():
        f = find_input_function_file(rawdata_dir, k.subject_id, k.session_unblinded)
        ifd = load_input_function(f)
        midp = float(df.loc[(df.subject_id == k.subject_id) & (df.session_unblinded == k.session_unblinded), "scan_midpoint_s"].iloc[0])
        base = calculate_input_function_auc(ifd, midp)["auc_0_to_midpoint_Bq_s_mL"]
        ai = np.where(ifd["roi"] == "aorta")[0]; pi = np.where(ifd["roi"] == "plasma")[0]
        t0, a0 = ifd["times"][ai[-1]], ifd["activities"][ai[-1]]
        t1, a1 = ifd["times"][pi[0]], ifd["activities"][pi[0]]
        lam = np.log(a0 / a1) / (t1 - t0)
        grid = np.arange(np.ceil(t0), np.floor(t1) + 1)
        expo = a0 * np.exp(-lam * (grid - t0))
        chord = a0 + (a1 - a0) * (grid - t0) / (t1 - t0)
        # replace the chord contribution on [t0, t1] (clipped at the midpoint) by the exponential
        m = grid <= midp
        delta = float(np.trapezoid(expo[m], grid[m]) - np.trapezoid(chord[m], grid[m])) if m.sum() > 1 else 0.0
        new_auc = base + delta
        bridge[f"{k.subject_id}/{k.session_unblinded}"] = {"gap_s": float(t1 - t0), "auc_change_pct": delta / base * 100}
        sel = (v.subject_id == k.subject_id) & (v.session_unblinded == k.session_unblinded)
        for c in ("FUR_top150_mean", "FUR_top150_median"):
            v.loc[sel, c] = v.loc[sel, c] * base / new_auc
    manifest["versions"]["exponential_bridge"] = bridge
    rows += stats_for(v, "exponential_bridge", "IDIF->plasma gap bridged exponentially instead of linearly")

    # (d) mask erosion / dilation by 1 mm ---------------------------------------------------------------------------
    struct = ndimage.generate_binary_structure(3, 1)  # 6-connected, 1 voxel = 1 mm
    for name, op in (("mask_erode_1mm", ndimage.binary_erosion), ("mask_dilate_1mm", ndimage.binary_dilation)):
        v = df.copy()
        vol_info = []
        for _, k in key.iterrows():
            pet_dir = rawdata_dir / k.subject_id / k.session_blinded / "pet"
            pet_file = find_pet_file(pet_dir, k.subject_id, k.session_blinded)
            pet, pet_img = load_nifti_with_scaling(pet_file)
            vox = get_voxel_dimensions(pet_img)
            for label in ("left", "right"):
                mfile = find_mask_file(pet_dir, k.subject_id, k.session_blinded, label)
                mask, mimg = load_nifti_with_scaling(mfile)
                eye, _ = mask_physical_eye(mask, mimg.affine)
                new_mask = op(mask > 0, structure=struct).astype(np.float32)
                met = calculate_metrics(pet, new_mask, vox, 2.0, 150)
                sel = (v.subject_id == k.subject_id) & (v.session_unblinded == k.session_unblinded) & (v.eye == eye)
                assert sel.sum() == 1
                row = v.loc[sel].iloc[0]
                suv_scaler = row["SUVtop150_mean"] / row["intensity_top150_mean_Bq_ml"]
                fur_scaler = row["FUR_top150_mean"] / row["intensity_top150_mean_Bq_ml"]
                v.loc[sel, "SUVtop150_mean"] = met["intensity_top150_mean"] * suv_scaler
                v.loc[sel, "SUVtop150_median"] = met["intensity_top150_median"] * suv_scaler
                v.loc[sel, "FUR_top150_mean"] = met["intensity_top150_mean"] * fur_scaler
                v.loc[sel, "FUR_top150_median"] = met["intensity_top150_median"] * fur_scaler
                vol_info.append({"subject": k.subject_id, "session": k.session_unblinded, "eye": eye,
                                 "voxels_original": int(row["mask_volume_voxels"]), "voxels_modified": met["mask_volume_voxels"],
                                 "top_n_used": met["top_n_used"], "top150_mean_change_pct": (met["intensity_top150_mean"] / row["intensity_top150_mean_Bq_ml"] - 1) * 100})
        vi = pd.DataFrame(vol_info)
        manifest["versions"][name] = {"voxels_modified_min": int(vi.voxels_modified.min()), "voxels_modified_max": int(vi.voxels_modified.max()),
                                      "n_masks_below_150": int((vi.top_n_used < 150).sum()),
                                      "top150_mean_change_pct": {"min": float(vi.top150_mean_change_pct.min()), "median": float(vi.top150_mean_change_pct.median()),
                                                                 "max": float(vi.top150_mean_change_pct.max())}}
        rows += stats_for(v, name, f"{name}: masks {vi.voxels_modified.min()}-{vi.voxels_modified.max()} voxels, "
                                   f"{int((vi.top_n_used < 150).sum())} masks <150 voxels, top150 mean change "
                                   f"{vi.top150_mean_change_pct.min():+.1f}..{vi.top150_mean_change_pct.max():+.1f}%")

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "sensitivity_table.csv", index=False)
    with open(out_dir / "sensitivity_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=float)

    # overview markdown: p (dz) per version x outcome x eye
    versions = list(dict.fromkeys(res["version"]))
    lines = ["**Supplementary Table S1.** Sensitivity of the Top-150 endpoints: paired-t p (Cohen's dz).", "",
             "| Version | " + " | ".join(f"{o} {e}" for o, _ in [(l, c) for c, l in OUTCOMES] for e in ("L", "R", "Bi")) + " |",
             "|---|" + "---|" * (len(OUTCOMES) * 3)]
    for ver in versions:
        cells = []
        for col, label in OUTCOMES:
            for lat in ("Left", "Right", "Bilateral"):
                r = res[(res.version == ver) & (res.Outcome == label) & (res.Eye == lat)].iloc[0]
                cells.append(f"{fmt_p(r.p_value)} ({r.cohens_dz:.2f})")
        lines.append(f"| {ver} | " + " | ".join(cells) + " |")
    lines += ["", "Notes: " + "; ".join(f"{ver}: {res[res.version == ver]['note'].iloc[0]}" for ver in versions if ver != "as_run"),
              "", "Eye = anatomical eye; L/R/Bi = left/right/bilateral mean; n = 13 pairs throughout."]
    (out_dir / "sensitivity_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
