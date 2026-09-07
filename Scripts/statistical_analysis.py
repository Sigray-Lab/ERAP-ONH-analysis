#!/usr/bin/env python3
"""
Statistical Analysis of Pre-Post Differences for ONH FDG-PET Metrics.

Performs paired t-tests comparing Baseline vs Followup for:
- SUVmax, SUVpeak_2mm
- SUVR_max, SUVR_peak_2mm
- TPR_max, TPR_peak_2mm
- FUR_max, FUR_peak_2mm

Separated by eye (left/right).

Output: CSV table with Mean±SD, Δ (95% CI), Cohen's dz, and p-values.
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def calculate_cohens_dz(differences: np.ndarray) -> float:
    """
    Calculate Cohen's dz for paired samples.

    dz = mean(differences) / std(differences)

    This is the appropriate effect size for paired t-tests.
    """
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)  # Use sample std
    if std_diff == 0:
        return np.nan
    return mean_diff / std_diff


def paired_analysis(baseline: np.ndarray, followup: np.ndarray) -> dict:
    """
    Perform paired t-test and calculate statistics.

    Returns dict with:
    - n: sample size
    - baseline_mean, baseline_sd
    - followup_mean, followup_sd
    - delta_mean: mean difference (followup - baseline)
    - delta_ci_low, delta_ci_high: 95% CI for the difference
    - cohens_dz: effect size
    - p_value: from paired t-test
    """
    n = len(baseline)
    differences = followup - baseline

    # Descriptive stats
    baseline_mean = np.mean(baseline)
    baseline_sd = np.std(baseline, ddof=1)
    followup_mean = np.mean(followup)
    followup_sd = np.std(followup, ddof=1)

    # Difference stats
    delta_mean = np.mean(differences)
    delta_se = stats.sem(differences)

    # 95% CI for difference
    t_crit = stats.t.ppf(0.975, df=n-1)
    delta_ci_low = delta_mean - t_crit * delta_se
    delta_ci_high = delta_mean + t_crit * delta_se

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(followup, baseline)

    # Effect size
    cohens_dz = calculate_cohens_dz(differences)

    return {
        'n': n,
        'baseline_mean': baseline_mean,
        'baseline_sd': baseline_sd,
        'followup_mean': followup_mean,
        'followup_sd': followup_sd,
        'delta_mean': delta_mean,
        'delta_ci_low': delta_ci_low,
        'delta_ci_high': delta_ci_high,
        'cohens_dz': cohens_dz,
        'p_value': p_value
    }


def format_mean_sd(mean: float, sd: float, decimals: int = 2) -> str:
    """Format as Mean ± SD."""
    return f"{mean:.{decimals}f} ± {sd:.{decimals}f}"


def format_delta_ci(delta: float, ci_low: float, ci_high: float, decimals: int = 2) -> str:
    """Format as Δ (95% CI)."""
    return f"{delta:+.{decimals}f} ({ci_low:+.{decimals}f}, {ci_high:+.{decimals}f})"


def format_p_value(p: float) -> str:
    """Format p-value with appropriate precision."""
    if p < 0.001:
        return "<0.001"
    elif p < 0.01:
        return f"{p:.3f}"
    else:
        return f"{p:.2f}"


def correlation_analysis(df, analysis_dir):
    """Correlate delta-change in SUV/FUR with rapamycin blood concentration."""

    output_dir = analysis_dir / "Outputs"
    figures_dir = output_dir / "Figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # --- Load rapamycin concentration data ---
    rapa_file = analysis_dir.parent.parent / "BIDS_20260205" / "raw" / "All_outcomes_20250630.csv"
    if not rapa_file.exists():
        print(f"WARNING: Rapamycin data not found at {rapa_file}, skipping correlation analysis.")
        return None

    rapa_df = pd.read_csv(rapa_file)
    rapa_df['subject_id'] = 'sub-' + rapa_df['Subject'].astype(str)
    rapa_per_subject = rapa_df.drop_duplicates(subset='subject_id')[['subject_id', 'rapa_conc_48h']].set_index('subject_id')

    # Exclude subjects with erroneous rapamycin concentration values
    rapa_exclude = ['sub-104', 'sub-107']
    rapa_per_subject = rapa_per_subject.drop(rapa_exclude, errors='ignore')

    # --- Define metrics (SUV + FUR only) ---
    suv_metrics = [
        ('SUVmax', 'SUV max'),
        ('SUVpeak_2mm', 'SUV peak'),
        ('SUVtop150_mean', 'SUV top150 mean'),
        ('SUVtop150_median', 'SUV top150 median'),
        ('SUVtop150_p90', 'SUV top150 p90'),
    ]
    fur_metrics = [
        ('FUR_max', 'FUR max'),
        ('FUR_peak_2mm', 'FUR peak'),
        ('FUR_top150_mean', 'FUR top150 mean'),
        ('FUR_top150_median', 'FUR top150 median'),
        ('FUR_top150_p90', 'FUR top150 p90'),
    ]
    all_metrics = suv_metrics + fur_metrics
    metric_cols = [col for col, _ in all_metrics]

    # --- Compute deltas per eye ---
    deltas = {}
    for eye in ['left', 'right']:
        eye_data = df[df['eye'] == eye]
        bl = eye_data[eye_data['session_unblinded'] == 'Baseline'].set_index('subject_id')[metric_cols]
        fu = eye_data[eye_data['session_unblinded'] == 'Followup'].set_index('subject_id')[metric_cols]
        common = bl.index.intersection(fu.index)
        deltas[eye.capitalize()] = fu.loc[common] - bl.loc[common]

    # --- Compute bilateral deltas ---
    left_data = df[df['eye'] == 'left']
    right_data = df[df['eye'] == 'right']
    bi_base_parts = {}
    bi_fup_parts = {}
    for tp in ['Baseline', 'Followup']:
        lt = left_data[left_data['session_unblinded'] == tp].set_index('subject_id')[metric_cols]
        rt = right_data[right_data['session_unblinded'] == tp].set_index('subject_id')[metric_cols]
        common = lt.index.intersection(rt.index)
        avg = (lt.loc[common] + rt.loc[common]) / 2
        if tp == 'Baseline':
            bi_base_parts = avg
        else:
            bi_fup_parts = avg
    common_bi = bi_base_parts.index.intersection(bi_fup_parts.index)
    deltas['Bilateral'] = bi_fup_parts.loc[common_bi] - bi_base_parts.loc[common_bi]

    # --- Correlations ---
    corr_results = []
    for eye_label in ['Left', 'Right', 'Bilateral']:
        delta_df = deltas[eye_label]
        merged = delta_df.join(rapa_per_subject, how='inner')

        for col, display_name in all_metrics:
            valid = merged[[col, 'rapa_conc_48h']].dropna()
            n = len(valid)
            if n < 4:
                continue
            r, p = stats.pearsonr(valid['rapa_conc_48h'], valid[col])
            corr_results.append({
                'Metric': display_name,
                'Eye': eye_label,
                'n': n,
                'Pearson_r': round(r, 3),
                'R2': round(r**2, 3),
                'p_value': round(p, 4),
            })

    corr_df = pd.DataFrame(corr_results)
    corr_csv = output_dir / "ONH_rapa_correlation.csv"
    corr_df.to_csv(corr_csv, index=False)
    print(f"\nCorrelation results saved to: {corr_csv}")

    # --- Figures ---
    eyes_order = ['Left', 'Right', 'Bilateral']

    for metric_group, group_label in [(suv_metrics, 'SUV'), (fur_metrics, 'FUR')]:
        fig, axes = plt.subplots(3, 5, figsize=(20, 12), constrained_layout=True)
        fig.suptitle(f'{group_label}: Δ (FU − BL) vs Rapamycin Concentration', fontsize=14, fontweight='bold')

        for row_idx, eye_label in enumerate(eyes_order):
            delta_df = deltas[eye_label]
            merged = delta_df.join(rapa_per_subject, how='inner')

            for col_idx, (col, display_name) in enumerate(metric_group):
                ax = axes[row_idx, col_idx]
                valid = merged[[col, 'rapa_conc_48h']].dropna()
                x = valid['rapa_conc_48h'].values
                y = valid[col].values

                ax.scatter(x, y, s=40, alpha=0.8, edgecolors='k', linewidths=0.5, zorder=3)

                # Regression line
                if len(x) >= 4:
                    slope, intercept = np.polyfit(x, y, 1)
                    x_line = np.linspace(x.min(), x.max(), 50)
                    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=1.5, alpha=0.7)

                    r, p = stats.pearsonr(x, y)
                    p_str = f'p < 0.001' if p < 0.001 else f'p = {p:.3f}' if p < 0.01 else f'p = {p:.2f}'
                    ax.text(0.05, 0.95, f'R² = {r**2:.2f}\n{p_str}',
                            transform=ax.transAxes, fontsize=8, verticalalignment='top',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

                # Labels
                short_name = display_name.replace(f'{group_label} ', '')
                if row_idx == 0:
                    ax.set_title(short_name, fontsize=10, fontweight='bold')
                if col_idx == 0:
                    ax.set_ylabel(f'{eye_label}\nΔ ({group_label})', fontsize=9)
                else:
                    ax.set_ylabel('')
                if row_idx == 2:
                    ax.set_xlabel('Rapa conc. 48h', fontsize=9)
                else:
                    ax.set_xlabel('')

                ax.axhline(0, color='grey', linewidth=0.5, linestyle='--', alpha=0.5)

        fig_path = figures_dir / f"ONH_rapa_correlation_{group_label}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Figure saved to: {fig_path}")

    # --- Print summary ---
    print("\n" + "=" * 80)
    print("Rapamycin Concentration vs Delta-Change Correlations (Pearson)")
    print("=" * 80)
    print(corr_df.to_string(index=False))

    return corr_df


def main():
    """Run statistical analysis."""

    # Load data using relative paths
    # Script is at: ONH_Analysis/Scripts/statistical_analysis.py
    script_dir = Path(__file__).parent          # ONH_Analysis/Scripts/
    analysis_dir = script_dir.parent            # ONH_Analysis/

    metrics_file = analysis_dir / "Outputs" / "ONH_FDG_metrics.csv"
    output_dir = analysis_dir / "Outputs"

    df = pd.read_csv(metrics_file)

    # Metrics to analyze: (column_name, display_label, decimals)
    # FUR values are small (~0.01-0.03 min⁻¹) so need more decimal places
    metrics = [
        ('SUVmax', 'SUV_max', 2),
        ('SUVpeak_2mm', 'SUV_peak', 2),
        ('SUVtop150_mean', 'SUV_top150_mean', 2),
        ('SUVtop150_median', 'SUV_top150_median', 2),
        ('SUVtop150_p90', 'SUV_top150_p90', 2),
        ('SUVR_max', 'SUVR_max', 2),
        ('SUVR_peak_2mm', 'SUVR_peak', 2),
        ('SUVR_top150_mean', 'SUVR_top150_mean', 2),
        ('SUVR_top150_median', 'SUVR_top150_median', 2),
        ('SUVR_top150_p90', 'SUVR_top150_p90', 2),
        ('TPR_max', 'TPR_max', 2),
        ('TPR_peak_2mm', 'TPR_peak', 2),
        ('TPR_top150_mean', 'TPR_top150_mean', 2),
        ('TPR_top150_median', 'TPR_top150_median', 2),
        ('TPR_top150_p90', 'TPR_top150_p90', 2),
        ('FUR_max', 'FUR_max', 4),
        ('FUR_peak_2mm', 'FUR_peak', 4),
        ('FUR_top150_mean', 'FUR_top150_mean', 4),
        ('FUR_top150_median', 'FUR_top150_median', 4),
        ('FUR_top150_p90', 'FUR_top150_p90', 4)
    ]

    results = []

    for eye in ['left', 'right']:
        eye_data = df[df['eye'] == eye]

        # Get paired data (subjects with both baseline and followup)
        baseline_data = eye_data[eye_data['session_unblinded'] == 'Baseline'].set_index('subject_id')
        followup_data = eye_data[eye_data['session_unblinded'] == 'Followup'].set_index('subject_id')

        # Find subjects with both timepoints
        common_subjects = baseline_data.index.intersection(followup_data.index)

        baseline_paired = baseline_data.loc[common_subjects]
        followup_paired = followup_data.loc[common_subjects]

        for col_name, outcome_label, decimals in metrics:
            baseline_vals = baseline_paired[col_name].values
            followup_vals = followup_paired[col_name].values

            # Skip if any NaN
            valid_mask = ~(np.isnan(baseline_vals) | np.isnan(followup_vals))
            baseline_clean = baseline_vals[valid_mask]
            followup_clean = followup_vals[valid_mask]

            if len(baseline_clean) < 3:
                continue

            stats_result = paired_analysis(baseline_clean, followup_clean)

            # Determine significance
            significant = stats_result['p_value'] < 0.05

            results.append({
                'Organ_system': 'ONH',
                'Eye': eye.capitalize(),
                'Outcome': outcome_label,
                'n': stats_result['n'],
                'Baseline_(Mean_±_SD)': format_mean_sd(stats_result['baseline_mean'], stats_result['baseline_sd'], decimals),
                'Follow-up_(Mean_±_SD)': format_mean_sd(stats_result['followup_mean'], stats_result['followup_sd'], decimals),
                'Δ_(95%_CI)': format_delta_ci(stats_result['delta_mean'],
                                               stats_result['delta_ci_low'],
                                               stats_result['delta_ci_high'], decimals),
                'Cohens_dz': f"{stats_result['cohens_dz']:.2f}",
                'p': format_p_value(stats_result['p_value']),
                'Significant': '*' if significant else ''
            })

    # --- Bilateral analysis (average of left and right per subject) ---
    numeric_cols = [col for col, _, _ in metrics]
    left_data = df[df['eye'] == 'left']
    right_data = df[df['eye'] == 'right']

    for timepoint_label in ['Baseline', 'Followup']:
        left_tp = left_data[left_data['session_unblinded'] == timepoint_label].set_index('subject_id')[numeric_cols]
        right_tp = right_data[right_data['session_unblinded'] == timepoint_label].set_index('subject_id')[numeric_cols]
        common = left_tp.index.intersection(right_tp.index)
        if timepoint_label == 'Baseline':
            bilateral_baseline = (left_tp.loc[common] + right_tp.loc[common]) / 2
        else:
            bilateral_followup = (left_tp.loc[common] + right_tp.loc[common]) / 2

    bilateral_subjects = bilateral_baseline.index.intersection(bilateral_followup.index)
    bilateral_base = bilateral_baseline.loc[bilateral_subjects]
    bilateral_fup = bilateral_followup.loc[bilateral_subjects]

    for col_name, outcome_label, decimals in metrics:
        baseline_vals = bilateral_base[col_name].values
        followup_vals = bilateral_fup[col_name].values

        valid_mask = ~(np.isnan(baseline_vals) | np.isnan(followup_vals))
        baseline_clean = baseline_vals[valid_mask]
        followup_clean = followup_vals[valid_mask]

        if len(baseline_clean) < 3:
            continue

        stats_result = paired_analysis(baseline_clean, followup_clean)
        significant = stats_result['p_value'] < 0.05

        results.append({
            'Organ_system': 'ONH',
            'Eye': 'Bilateral',
            'Outcome': outcome_label,
            'n': stats_result['n'],
            'Baseline_(Mean_±_SD)': format_mean_sd(stats_result['baseline_mean'], stats_result['baseline_sd'], decimals),
            'Follow-up_(Mean_±_SD)': format_mean_sd(stats_result['followup_mean'], stats_result['followup_sd'], decimals),
            'Δ_(95%_CI)': format_delta_ci(stats_result['delta_mean'],
                                           stats_result['delta_ci_low'],
                                           stats_result['delta_ci_high'], decimals),
            'Cohens_dz': f"{stats_result['cohens_dz']:.2f}",
            'p': format_p_value(stats_result['p_value']),
            'Significant': '*' if significant else ''
        })

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Save to CSV
    output_file = output_dir / "ONH_pre_post_statistics.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")

    # Print formatted table
    print("\n" + "=" * 120)
    print("ONH FDG-PET Pre-Post Analysis: Baseline vs Follow-up (Paired t-tests)")
    print("=" * 120)

    # Print for display
    display_cols = ['Organ_system', 'Eye', 'Outcome', 'n', 'Baseline_(Mean_±_SD)',
                    'Follow-up_(Mean_±_SD)', 'Δ_(95%_CI)', 'Cohens_dz', 'p', 'Significant']

    print(results_df[display_cols].to_string(index=False))

    print("\n" + "-" * 80)
    print("Note: * indicates p < 0.05")
    print("Cohen's dz interpretation: 0.2 = small, 0.5 = medium, 0.8 = large effect")
    print("-" * 80)

    # --- Rapamycin correlation analysis ---
    corr_df = correlation_analysis(df, analysis_dir)

    return results_df


if __name__ == "__main__":
    results = main()
