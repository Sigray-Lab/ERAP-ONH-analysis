# ONH FDG-PET: Methods and Results

## Methods

### PET Acquisition

All participants underwent [18F]FDG-PET/CT on a GE Discovery MI 5 scanner at baseline and follow-up. A single static frame was acquired approximately 30-60 minutes post-injection (frame duration: 30 min), reconstructed with motion correction (MoCo) to 1 mm isotropic voxels (384 x 384 x 249 matrix). Native image units were Bq/mL.

### ONH Mask Delineation

Regions of interest (ROIs) over the left and right optic nerve head (ONH) were manually delineated by a trained operator on the reconstructed PET images. Delineation was performed under blinded conditions (randomized session codes) to prevent bias. Masks were intentionally drawn larger than the anatomical ONH (~1.5-2 mm diameter) to ensure complete coverage given the limited PET spatial resolution. Mask volumes ranged from 230 to 797 voxels (mm3).

### Quantification

Because the ONH is considerably smaller than the PET spatial resolution (~5.2 mm FWHM at the ONH location), and mask volumes varied approximately 3.5-fold across delineations, we employed resolution-robust intensity measures that are independent of total mask size:

- **SUVmax**: The single highest-intensity voxel within each mask.
- **SUVpeak**: The mean intensity within a fixed 2 mm radius sphere centered on the maximum voxel, as recommended by PERCIST (1).
- **Top-150**: The mean, median, and 90th percentile of the 150 highest-intensity voxels within each mask. The value of 150 was derived from scanner resolution: at the average ONH distance from the field-of-view center (~75 mm), the estimated FWHM is ~5.2 mm, yielding a resolution element volume of (4/3)pi(5.2/2)^3 = 74 mm3 = 74 voxels. Two resolution elements correspond to ~148 voxels, rounded to 150. Since all masks contained at least 230 voxels, this metric is unbiased by mask size.

Two normalizations were applied:

1. **SUV** (Standardized Uptake Value): PET intensity (Bq/mL) x body weight (kg) / [injected dose (MBq) x 1000]. Body weight and injected dose were obtained from the electronic case report form.
2. **FUR** (Fractional Uptake Rate): PET intensity (Bq/mL) / AUC(0 to scan midpoint) x 60, yielding units of min-1. The input function was constructed by combining an image-derived input function (IDIF) from the descending aorta with manual venous plasma samples, and integrated from time zero to the scan midpoint (ScanStart + FrameDuration/2) using the trapezoidal rule (2).

### Statistical Analysis

Pre-post differences (follow-up minus baseline) were assessed using two-sided paired t-tests for each metric and eye separately, as well as for the bilateral average (mean of left and right eye per subject). Cohen's dz was calculated as the within-subject effect size. 95% confidence intervals for the mean change were computed using the t-distribution. No correction for multiple comparisons was applied given the exploratory nature of this pilot study. All analyses were performed in Python (scipy.stats).

---

## Results

Thirteen subjects contributed complete paired data at both timepoints, yielding 26 eyes per laterality and 13 bilateral averages.

### SUV

SUV increased numerically from baseline to follow-up across all metrics and lateralities. Statistically significant increases were observed for the Top-150 mean and median in the left eye and bilaterally, with medium effect sizes (Table 1).

**Table 1.** SUV at the ONH: baseline vs follow-up (paired t-tests, n = 13).

| Metric | Eye | n | Baseline (SD) | Follow-up (SD) | Δ (95% CI) | % Δ | Cohen's dz | Paired p |
|--------|-----|---|---------------|----------------|------------|-----|------------|----------|
| SUVmax | Left | 13 | 3.00 (0.42) | 3.19 (0.62) | +0.19 (-0.11, +0.49) | +6.3 | 0.39 | 0.19 |
| SUVmax | Right | 13 | 3.08 (0.66) | 3.24 (0.69) | +0.16 (-0.22, +0.54) | +5.3 | 0.26 | 0.37 |
| SUVmax | Bilateral | 13 | 3.04 (0.52) | 3.21 (0.62) | +0.18 (-0.10, +0.45) | +5.8 | 0.39 | 0.18 |
| SUVpeak | Left | 13 | 2.66 (0.37) | 2.82 (0.51) | +0.16 (-0.07, +0.39) | +5.9 | 0.42 | 0.16 |
| SUVpeak | Right | 13 | 2.73 (0.53) | 2.86 (0.55) | +0.13 (-0.16, +0.42) | +4.8 | 0.27 | 0.34 |
| SUVpeak | Bilateral | 13 | 2.69 (0.44) | 2.84 (0.51) | +0.14 (-0.07, +0.36) | +5.4 | 0.41 | 0.17 |
| SUV_top150 mean | Left | 13 | 2.39 (0.34) | 2.58 (0.43) | +0.19 (+0.01, +0.38) | +8.1 | 0.64 | **0.04** |
| SUV_top150 mean | Right | 13 | 2.40 (0.40) | 2.59 (0.43) | +0.18 (-0.01, +0.38) | +7.6 | 0.57 | 0.06 |
| SUV_top150 mean | Bilateral | 13 | 2.39 (0.37) | 2.58 (0.43) | +0.19 (+0.01, +0.36) | +7.9 | 0.66 | **0.04** |
| SUV_top150 median | Left | 13 | 2.34 (0.34) | 2.54 (0.41) | +0.19 (+0.01, +0.37) | +8.1 | 0.65 | **0.04** |
| SUV_top150 median | Right | 13 | 2.36 (0.39) | 2.54 (0.41) | +0.18 (-0.01, +0.36) | +7.5 | 0.58 | 0.06 |
| SUV_top150 median | Bilateral | 13 | 2.35 (0.36) | 2.54 (0.41) | +0.18 (+0.02, +0.35) | +7.8 | 0.66 | **0.03** |
| SUV_top150 p90 | Left | 13 | 2.76 (0.40) | 2.93 (0.53) | +0.17 (-0.09, +0.42) | +6.1 | 0.39 | 0.18 |
| SUV_top150 p90 | Right | 13 | 2.82 (0.55) | 2.98 (0.59) | +0.16 (-0.16, +0.47) | +5.5 | 0.30 | 0.30 |
| SUV_top150 p90 | Bilateral | 13 | 2.79 (0.46) | 2.95 (0.54) | +0.16 (-0.08, +0.40) | +5.8 | 0.41 | 0.16 |

### FUR

The FUR results mirrored the SUV findings. The Top-150 mean and median showed significant increases in the left eye and bilaterally, again with medium effect sizes (Table 2).

**Table 2.** FUR (min-1) at the ONH: baseline vs follow-up (paired t-tests, n = 13).

| Metric | Eye | n | Baseline (SD) ×10⁻³ | Follow-up (SD) ×10⁻³ | Δ ×10⁻³ (95% CI) | % Δ | Cohen's dz | Paired p |
|--------|-----|---|----------------------|-----------------------|-------------------|-----|------------|----------|
| FURmax | Left | 13 | 16.3 (2.5) | 17.4 (3.9) | +1.1 (-0.6, +2.8) | +6.8 | 0.40 | 0.18 |
| FURmax | Right | 13 | 16.7 (3.3) | 17.6 (3.4) | +0.9 (-1.3, +3.2) | +5.6 | 0.25 | 0.38 |
| FURmax | Bilateral | 13 | 16.5 (2.7) | 17.5 (3.5) | +1.0 (-0.6, +2.7) | +6.2 | 0.38 | 0.20 |
| FURpeak | Left | 13 | 14.5 (2.1) | 15.4 (3.1) | +0.9 (-0.4, +2.2) | +6.4 | 0.43 | 0.15 |
| FURpeak | Right | 13 | 14.8 (2.7) | 15.6 (2.8) | +0.8 (-1.0, +2.5) | +5.2 | 0.27 | 0.35 |
| FURpeak | Bilateral | 13 | 14.6 (2.3) | 15.5 (2.8) | +0.8 (-0.4, +2.1) | +5.8 | 0.40 | 0.18 |
| FUR_top150 mean | Left | 13 | 13.0 (1.9) | 14.1 (2.4) | +1.1 (+0.1, +2.1) | +8.5 | 0.66 | **0.03** |
| FUR_top150 mean | Right | 13 | 13.0 (2.1) | 14.1 (2.4) | +1.0 (-0.2, +2.3) | +8.0 | 0.52 | 0.08 |
| FUR_top150 mean | Bilateral | 13 | 13.0 (2.0) | 14.1 (2.3) | +1.1 (+0.0, +2.1) | +8.2 | 0.63 | **0.04** |
| FUR_top150 median | Left | 13 | 12.7 (1.8) | 13.8 (2.2) | +1.1 (+0.1, +2.1) | +8.5 | 0.67 | **0.03** |
| FUR_top150 median | Right | 13 | 12.8 (2.1) | 13.8 (2.3) | +1.0 (-0.1, +2.2) | +8.0 | 0.53 | 0.08 |
| FUR_top150 median | Bilateral | 13 | 12.8 (1.9) | 13.8 (2.2) | +1.1 (+0.0, +2.1) | +8.2 | 0.63 | **0.04** |
| FUR_top150 p90 | Left | 13 | 15.0 (2.2) | 16.0 (3.1) | +1.0 (-0.4, +2.4) | +6.5 | 0.41 | 0.16 |
| FUR_top150 p90 | Right | 13 | 15.3 (2.9) | 16.2 (3.0) | +0.9 (-0.9, +2.7) | +5.9 | 0.30 | 0.31 |
| FUR_top150 p90 | Bilateral | 13 | 15.2 (2.4) | 16.1 (3.0) | +0.9 (-0.4, +2.3) | +6.2 | 0.41 | 0.17 |

### Summary

Across both SUV and FUR, the Top-150 mean and median consistently showed the largest effect sizes and were the only metrics to reach statistical significance. The effects were directionally consistent across left and right eyes, though individually significant only in the left eye and in the bilateral average. Single-voxel (max) and sphere-based (peak) metrics showed smaller, non-significant effects, consistent with their higher susceptibility to noise.

---

## References

1. Wahl RL, et al. From RECIST to PERCIST: Evolving Considerations for PET Response Criteria in Solid Tumors. *J Nucl Med*. 2009;50 Suppl 1:122S-150S.
2. Patlak CS, Blasberg RG. Graphical evaluation of blood-to-brain transfer constants from multiple-time uptake data. *J Cereb Blood Flow Metab*. 1985;5(4):584-90.
