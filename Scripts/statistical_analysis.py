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

    return results_df


if __name__ == "__main__":
    results = main()
