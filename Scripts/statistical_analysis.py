#!/usr/bin/env python3
"""
Pre-post statistics for the ONH FDG-PET metrics (Baseline vs Follow-up, paired) and the exploratory
correlation of the per-subject change with rapamycin blood concentration.

Revised 2026-09-07 after the adversarial review:
  * `eye` in the metrics CSV is the anatomical eye (asserted here); tables are labelled accordingly.
  * All manuscript numbers come from this script: ONH_pre_post_statistics.csv (60 rows, numeric + formatted),
    table1.md / table2.md (SUV, FUR), ancillary_numbers.json, ONH_rapa_correlation.csv, figures,
    README_outputs.md.
  * Wilcoxon signed-rank p is reported next to the paired t-test p as a robustness column (n = 13).
  * Rapamycin file: project copy RawData/All_outcomes_20250630.csv (byte-identical to the BIDS copy);
    exclusion rationale recorded; duplicate concentration rows must agree; skipped tests are logged.

Usage:
    cd ONH_Analysis/Scripts
    python statistical_analysis.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Metrics: (column, label, decimals)
METRICS = [
    ("SUVmax", "SUV_max", 2), ("SUVpeak_2mm", "SUV_peak", 2), ("SUVtop150_mean", "SUV_top150_mean", 2),
    ("SUVtop150_median", "SUV_top150_median", 2), ("SUVtop150_p90", "SUV_top150_p90", 2),
    ("SUVR_max", "SUVR_max", 2), ("SUVR_peak_2mm", "SUVR_peak", 2), ("SUVR_top150_mean", "SUVR_top150_mean", 2),
    ("SUVR_top150_median", "SUVR_top150_median", 2), ("SUVR_top150_p90", "SUVR_top150_p90", 2),
    ("TPR_max", "TPR_max", 2), ("TPR_peak_2mm", "TPR_peak", 2), ("TPR_top150_mean", "TPR_top150_mean", 2),
    ("TPR_top150_median", "TPR_top150_median", 2), ("TPR_top150_p90", "TPR_top150_p90", 2),
    ("FUR_max", "FUR_max", 4), ("FUR_peak_2mm", "FUR_peak", 4), ("FUR_top150_mean", "FUR_top150_mean", 4),
    ("FUR_top150_median", "FUR_top150_median", 4), ("FUR_top150_p90", "FUR_top150_p90", 4),
]
LATERALITIES = ["left", "right", "bilateral"]

# Rapamycin correlation: subjects excluded and why (study-team record, see REVIEW_RESPONSE.md / sibling
# cardiovascular REVIEW_RESPONSE F17): sub-104 and sub-107 were on a different rapamycin dose at the time of
# the 48 h PK sample and the dose was changed afterwards, so their concentrations do not represent the
# analysed treatment period. Not an assay error.
RAPA_EXCLUDE = {"sub-104": "different rapamycin dose at the 48 h PK sample; dose changed afterwards",
                "sub-107": "different rapamycin dose at the 48 h PK sample; dose changed afterwards"}


def paired_analysis(baseline: np.ndarray, followup: np.ndarray) -> dict:
    n = len(baseline)
    d = followup - baseline
    se = stats.sem(d)
    tcrit = stats.t.ppf(0.975, df=n - 1)
    t_stat, p = stats.ttest_rel(followup, baseline)
    sd = np.std(d, ddof=1)
    try:
        w_p = float(stats.wilcoxon(followup, baseline).pvalue)
    except ValueError:
        w_p = np.nan
    return {"n": n, "baseline_mean": baseline.mean(), "baseline_sd": np.std(baseline, ddof=1),
            "followup_mean": followup.mean(), "followup_sd": np.std(followup, ddof=1),
            "delta_mean": d.mean(), "delta_ci_low": d.mean() - tcrit * se, "delta_ci_high": d.mean() + tcrit * se,
            "percent_change": d.mean() / baseline.mean() * 100.0, "cohens_dz": d.mean() / sd if sd > 0 else np.nan,
            "t_statistic": float(t_stat), "p_value": float(p), "wilcoxon_p": w_p}


def fmt_p(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def paired_frames(df: pd.DataFrame, cols, log=print):
    """Baseline/Followup frames indexed by subject for each laterality (bilateral = mean of both eyes).
    Requires unique (subject, session, eye) keys; a bilateral value is formed only when both eyes are present."""
    dup = df.duplicated(["subject_id", "session_unblinded", "eye"])
    if dup.any():
        raise RuntimeError(f"duplicate (subject, session, eye) rows: {df.loc[dup, ['subject_id', 'session_unblinded', 'eye']].values.tolist()}")
    out = {}
    for lat in LATERALITIES:
        if lat == "bilateral":
            counts = df.groupby(["subject_id", "session_unblinded"])["eye"].nunique()
            incomplete = counts[counts < 2]
            for (s, t), n in incomplete.items():
                log(f"SKIPPED bilateral {s}/{t}: only {n} eye(s) present")
            complete = df.set_index(["subject_id", "session_unblinded"]).index.isin(counts[counts == 2].index)
            g = df[complete].groupby(["subject_id", "session_unblinded"])[cols].mean().reset_index()
        else:
            g = df[df["eye"] == lat]
        b = g[g["session_unblinded"] == "Baseline"].set_index("subject_id")[cols]
        f = g[g["session_unblinded"] == "Followup"].set_index("subject_id")[cols]
        common = b.index.intersection(f.index)
        out[lat] = (b.loc[common], f.loc[common])
    return out


def run_paired_stats(df: pd.DataFrame, log) -> pd.DataFrame:
    cols = [c for c, _, _ in METRICS]
    frames = paired_frames(df, cols, log)
    rows = []
    for lat in LATERALITIES:
        b, f = frames[lat]
        for col, label, dec in METRICS:
            bv, fv = b[col].to_numpy(float), f[col].to_numpy(float)
            ok = np.isfinite(bv) & np.isfinite(fv)
            if ok.sum() < 3:
                log(f"SKIPPED {label} {lat}: only {ok.sum()} complete pairs")
                continue
            r = paired_analysis(bv[ok], fv[ok])
            rows.append({
                "Organ_system": "ONH", "Eye": lat.capitalize(), "Outcome": label, "metric_column": col, "n": r["n"],
                "baseline_mean": r["baseline_mean"], "baseline_sd": r["baseline_sd"],
                "followup_mean": r["followup_mean"], "followup_sd": r["followup_sd"],
                "delta_mean": r["delta_mean"], "delta_ci_low": r["delta_ci_low"], "delta_ci_high": r["delta_ci_high"],
                "percent_change": r["percent_change"], "cohens_dz": r["cohens_dz"], "t_statistic": r["t_statistic"],
                "p_value": r["p_value"], "wilcoxon_p": r["wilcoxon_p"],
                "Baseline_(Mean_±_SD)": f"{r['baseline_mean']:.{dec}f} ± {r['baseline_sd']:.{dec}f}",
                "Follow-up_(Mean_±_SD)": f"{r['followup_mean']:.{dec}f} ± {r['followup_sd']:.{dec}f}",
                "Δ_(95%_CI)": f"{r['delta_mean']:+.{dec}f} ({r['delta_ci_low']:+.{dec}f}, {r['delta_ci_high']:+.{dec}f})",
                "%Δ": f"{r['percent_change']:+.1f}", "Cohens_dz": f"{r['cohens_dz']:.2f}", "p": fmt_p(r["p_value"]),
                "p_wilcoxon": fmt_p(r["wilcoxon_p"]), "Significant": "*" if r["p_value"] < 0.05 else "",
            })
    return pd.DataFrame(rows)


def write_table_md(res: pd.DataFrame, prefix: str, title: str, path: Path, scale: float = 1.0, unit_note: str = ""):
    sub = res[res["Outcome"].str.startswith(prefix + "_")].copy()
    order = {f"{prefix}_max": 0, f"{prefix}_peak": 1, f"{prefix}_top150_mean": 2, f"{prefix}_top150_median": 3, f"{prefix}_top150_p90": 4}
    lat_order = {"Left": 0, "Right": 1, "Bilateral": 2}
    sub["o"] = sub["Outcome"].map(order); sub["l"] = sub["Eye"].map(lat_order)
    sub = sub.sort_values(["o", "l"])
    dec = 1 if scale != 1.0 else 2
    lines = [f"**{title}**", "", "| Metric | Eye | n | Baseline (SD) | Follow-up (SD) | Δ (95% CI) | % Δ | Cohen's dz | Paired p | Wilcoxon p |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in sub.iterrows():
        s = scale
        lines.append(f"| {r['Outcome']} | {r['Eye']} | {r['n']} | {r['baseline_mean']*s:.{dec}f} ({r['baseline_sd']*s:.{dec}f}) | "
                     f"{r['followup_mean']*s:.{dec}f} ({r['followup_sd']*s:.{dec}f}) | {r['delta_mean']*s:+.{dec}f} "
                     f"({r['delta_ci_low']*s:+.{dec}f}, {r['delta_ci_high']*s:+.{dec}f}) | {r['percent_change']:+.1f} | "
                     f"{r['cohens_dz']:.2f} | {'**' if r['p_value'] < 0.05 else ''}{fmt_p(r['p_value'])}{'**' if r['p_value'] < 0.05 else ''} | {fmt_p(r['wilcoxon_p'])} |")
    if unit_note:
        lines += ["", unit_note]
    lines += ["", "Eye = anatomical eye (from the mask centroid in world coordinates). Paired t-tests, two-sided, uncorrected; "
              "95 % CI from the t distribution; Cohen's dz = mean/SD of the paired differences."]
    path.write_text("\n".join(lines) + "\n")


def correlation_analysis(df: pd.DataFrame, project_root: Path, output_dir: Path, log):
    rapa_file = project_root / "RawData" / "All_outcomes_20250630.csv"
    if not rapa_file.exists():
        log(f"WARNING: rapamycin data not found at {rapa_file}; correlation skipped")
        return None
    raw = pd.read_csv(rapa_file)
    raw["subject_id"] = "sub-" + raw["Subject"].astype(str)
    nun = raw.groupby("subject_id")["rapa_conc_48h"].nunique()
    if (nun > 1).any():
        raise RuntimeError(f"rapa_conc_48h differs between rows of the same subject: {nun[nun > 1].index.tolist()}")
    conc = raw.drop_duplicates("subject_id").set_index("subject_id")["rapa_conc_48h"]
    conc = conc.drop(list(RAPA_EXCLUDE), errors="ignore")
    pd.DataFrame([{"subject_id": s, "reason": r} for s, r in RAPA_EXCLUDE.items()]).to_csv(output_dir / "rapa_correlation_exclusions.csv", index=False)

    suv = [("SUVmax", "SUV max"), ("SUVpeak_2mm", "SUV peak"), ("SUVtop150_mean", "SUV top150 mean"),
           ("SUVtop150_median", "SUV top150 median"), ("SUVtop150_p90", "SUV top150 p90")]
    fur = [("FUR_max", "FUR max"), ("FUR_peak_2mm", "FUR peak"), ("FUR_top150_mean", "FUR top150 mean"),
           ("FUR_top150_median", "FUR top150 median"), ("FUR_top150_p90", "FUR top150 p90")]
    cols = [c for c, _ in suv + fur]
    frames = paired_frames(df, cols, log)
    deltas = {lat: (frames[lat][1] - frames[lat][0]) for lat in LATERALITIES}

    rows = []
    for lat in LATERALITIES:
        merged = deltas[lat].join(conc, how="inner")
        for col, name in suv + fur:
            v = merged[[col, "rapa_conc_48h"]].dropna()
            if len(v) < 4:
                log(f"SKIPPED correlation {name} {lat}: n={len(v)}")
                continue
            r, p = stats.pearsonr(v["rapa_conc_48h"], v[col])
            rows.append({"Metric": name, "Eye": lat.capitalize(), "n": len(v), "Pearson_r": round(r, 3), "R2": round(r ** 2, 3), "p_value": round(p, 4)})
    corr = pd.DataFrame(rows)
    corr.to_csv(output_dir / "ONH_rapa_correlation.csv", index=False)

    figures_dir = output_dir / "Figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for group, label in ((suv, "SUV"), (fur, "FUR")):
        fig, axes = plt.subplots(3, 5, figsize=(20, 12), constrained_layout=True)
        fig.suptitle(f"{label}: Δ (FU − BL) vs rapamycin concentration (48 h); n = {len(conc)} after documented exclusions", fontsize=14, fontweight="bold")
        for i, lat in enumerate(LATERALITIES):
            merged = deltas[lat].join(conc, how="inner")
            for j, (col, name) in enumerate(group):
                ax = axes[i, j]
                v = merged[[col, "rapa_conc_48h"]].dropna()
                x, y = v["rapa_conc_48h"].to_numpy(), v[col].to_numpy()
                ax.scatter(x, y, s=40, alpha=0.8, edgecolors="k", linewidths=0.5, zorder=3)
                if len(x) >= 4:
                    slope, intercept = np.polyfit(x, y, 1)
                    xl = np.linspace(x.min(), x.max(), 50)
                    ax.plot(xl, slope * xl + intercept, "r-", linewidth=1.5, alpha=0.7)
                    r, p = stats.pearsonr(x, y)
                    ax.text(0.05, 0.95, f"R² = {r**2:.2f}\np = {fmt_p(p)}", transform=ax.transAxes, fontsize=8, va="top",
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))
                if i == 0:
                    ax.set_title(name.replace(f"{label} ", ""), fontsize=10, fontweight="bold")
                ax.set_ylabel(f"{lat.capitalize()} eye\nΔ ({label})" if j == 0 else "", fontsize=9)
                ax.set_xlabel("Rapa conc. 48 h" if i == 2 else "", fontsize=9)
                ax.axhline(0, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
        fig.savefig(figures_dir / f"ONH_rapa_correlation_{label}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    log(f"Correlation results: {output_dir / 'ONH_rapa_correlation.csv'} ({len(corr)} rows)")
    return corr


def ancillary_numbers(df: pd.DataFrame, res: pd.DataFrame, corr) -> dict:
    vol = df["mask_volume_voxels"]
    per_session = df.drop_duplicates(["subject_id", "session_unblinded"])
    return {
        "n_subjects": int(df["subject_id"].nunique()), "n_sessions": int(len(per_session)), "n_eye_visits": int(len(df)),
        "n_eyes_per_laterality": int(df["subject_id"].nunique()), "n_bilateral_pairs": int(res[res["Eye"] == "Bilateral"]["n"].max()),
        "mask_volume_voxels": {"min": int(vol.min()), "max": int(vol.max()), "mean": round(float(vol.mean()), 1),
                               "median": float(vol.median()), "sd": round(float(vol.std(ddof=1)), 1), "ratio_max_min": round(float(vol.max() / vol.min()), 2)},
        "scan_start_min_post_injection": {"min": round(float(per_session["scan_start_s"].min() / 60), 1), "max": round(float(per_session["scan_start_s"].max() / 60), 1)},
        "scan_end_min_post_injection": {"min": round(float((per_session["scan_start_s"] + per_session["scan_duration_s"]).min() / 60), 1),
                                        "max": round(float((per_session["scan_start_s"] + per_session["scan_duration_s"]).max() / 60), 1)},
        "decay_reference": sorted(df["decay_reference"].unique().tolist()),
        "normalisations_computed": ["SUV", "SUVR", "TPR", "FUR"], "normalisations_reported": ["SUV", "FUR"],
        "n_paired_tests_run": int(len(res)), "n_paired_tests_reported": int(res["Outcome"].str.match(r"^(SUV|FUR)_").sum()),
        "n_significant_uncorrected": int((res["p_value"] < 0.05).sum()),
        "rapa_correlation_n": int(corr["n"].iloc[0]) if corr is not None and len(corr) else None,
        "rapa_excluded": RAPA_EXCLUDE,
    }


def write_readme_outputs(output_dir: Path, ancillary: dict):
    txt = f"""# Outputs of the ONH FDG-PET pipeline (generated by statistical_analysis.py)

| File | Content | Used for |
|---|---|---|
| `ONH_FDG_metrics.csv` | 52 eye-visits x all metrics; `eye` = anatomical eye, `mask_label_in_filename` = delineator's label | everything below |
| `run_manifest.json` | git commit, package versions, input hashes, config of the extraction run | provenance |
| `ONH_pre_post_statistics.csv` | 60 paired tests (20 metrics x Left/Right/Bilateral), numeric and formatted columns | Tables 1-2, supplement |
| `table1.md`, `table2.md` | Manuscript Tables 1 (SUV) and 2 (FUR) | manuscript, paste as-is |
| `ancillary_numbers.json` | counts, mask-volume statistics, scan timing, number of tests | Methods/Results text |
| `ONH_rapa_correlation.csv`, `rapa_correlation_exclusions.csv`, `Figures/ONH_rapa_correlation_*.png` | exploratory correlation of the change with rapamycin concentration | Results/supplement |
| `sensitivity_table.csv` / `.md` | sensitivity of the Top-150 results to dose source, sub-110 follow-up sample times, ±1 mm delineation, IF bridging | supplement |
| `../QC/QC_flags_report.csv`, `../QC/QC_summary_report.txt`, `../QC/SUVpeak_visualizations/` | QC flags (structured) and per-session images | QC |
| `../DerivedData/session_scaling_factors.csv` | per-session dose, weight, SUV scaler, cerebellum mean, plasma and IF AUCs | audit |
| `archive_v1_as_reviewed/` | outputs as reviewed on 2026-09-06 (START-referenced PET, filename laterality) - do not use | history |

**Manuscript update notice:** `../RESULTS_UPDATE_FOR_MANUSCRIPT.md` lists every superseded number, the old→new mapping
and the sentences to change.

Key facts of this run: {ancillary['n_subjects']} subjects, {ancillary['n_sessions']} sessions, {ancillary['n_eye_visits']} eye-visits;
PET decay reference {ancillary['decay_reference']}; mask volume {ancillary['mask_volume_voxels']['min']}-{ancillary['mask_volume_voxels']['max']} voxels;
{ancillary['n_paired_tests_run']} paired tests run, {ancillary['n_paired_tests_reported']} reported (SUV, FUR).
"""
    (output_dir / "README_outputs.md").write_text(txt)


def main():
    script_dir = Path(__file__).resolve().parent
    analysis_dir = script_dir.parent
    project_root = analysis_dir.parent
    output_dir = analysis_dir / "Outputs"
    log = lambda m: print(m)  # noqa: E731

    df = pd.read_csv(output_dir / "ONH_FDG_metrics.csv")
    if "mask_label_in_filename" not in df.columns or set(df["eye"].unique()) != {"left", "right"}:
        raise RuntimeError("Metrics CSV lacks the anatomical-eye mapping (mask_label_in_filename); re-run extract_onh_metrics.py")
    if set(df["decay_reference"].unique()) != {"INJECTION"}:
        raise RuntimeError(f"PET decay reference is {df['decay_reference'].unique()}; expected INJECTION only")

    res = run_paired_stats(df, log)
    res.to_csv(output_dir / "ONH_pre_post_statistics.csv", index=False)
    write_table_md(res, "SUV", "Table 1. SUV at the ONH: baseline vs follow-up (paired t-tests, n = 13).", output_dir / "table1.md")
    write_table_md(res, "FUR", "Table 2. FUR (min⁻¹, values ×10⁻³) at the ONH: baseline vs follow-up (paired t-tests, n = 13).",
                   output_dir / "table2.md", scale=1000.0, unit_note="Values are FUR × 10⁻³ min⁻¹.")
    corr = correlation_analysis(df, project_root, output_dir, log)
    anc = ancillary_numbers(df, res, corr)
    with open(output_dir / "ancillary_numbers.json", "w") as f:
        json.dump(anc, f, indent=2)
    write_readme_outputs(output_dir, anc)

    print("\n" + "=" * 120)
    print("ONH FDG-PET Pre-Post Analysis: Baseline vs Follow-up (paired t-tests; Eye = anatomical)")
    print("=" * 120)
    print(res[["Eye", "Outcome", "n", "Baseline_(Mean_±_SD)", "Follow-up_(Mean_±_SD)", "Δ_(95%_CI)", "%Δ", "Cohens_dz", "p", "p_wilcoxon", "Significant"]].to_string(index=False))
    print(f"\nResults saved to: {output_dir / 'ONH_pre_post_statistics.csv'}; tables: table1.md, table2.md; ancillary_numbers.json; README_outputs.md")
    return res


if __name__ == "__main__":
    main()
