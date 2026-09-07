#!/usr/bin/env python3
"""
QC visualisation for ONH FDG-PET extraction: one PNG per session showing the axial PET slice through the
max voxel of each eye, with the mask contour, the 2 mm sphere and the max voxel.

Revised 2026-09-07: panels are ordered and titled by *anatomical* eye (right eye on screen-left,
radiological convention), with R/L annotations derived from the image affine; the filename label is
shown in the title; colourbar is Bq/mL. The unused single-eye function was removed (review F20).
"""
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def _axial_orientation(affine: np.ndarray):
    """Screen labels for an imshow of pet[:, :, z].T with origin='lower': columns = i axis, rows = j axis."""
    codes = nib.aff2axcodes(affine)
    right_label = "L" if codes[0] == "L" else "R"     # label at the screen-right edge
    left_label = "R" if right_label == "L" else "L"
    top_label = codes[1] if codes[1] in ("A", "P") else "?"
    return left_label, right_label, top_label


def _panel(ax, pet_data, mask_data, metrics, affine, voxel_dims, sphere_radius_mm, zoom, title):
    from utils import create_sphere_mask
    if mask_data is None or metrics is None or "error" in metrics:
        ax.text(0.5, 0.5, f"{title}\nNo data", transform=ax.transAxes, ha="center", va="center", fontsize=14, color="gray")
        ax.set_facecolor("#1a1a1a")
        ax.axis("off")
        return None
    cx, cy, cz = int(metrics["max_voxel_x"]), int(metrics["max_voxel_y"]), int(metrics["max_voxel_z"])
    sphere = create_sphere_mask((cx, cy, cz), sphere_radius_mm, voxel_dims, pet_data.shape)
    x0, x1 = max(0, cx - zoom), min(pet_data.shape[0], cx + zoom)
    y0, y1 = max(0, cy - zoom), min(pet_data.shape[1], cy + zoom)
    pet_z = pet_data[x0:x1, y0:y1, cz].T
    mask_z = mask_data[x0:x1, y0:y1, cz].T
    sph_z = sphere[x0:x1, y0:y1, cz].T
    vmax = np.percentile(pet_z[pet_z > 0], 99) if np.any(pet_z > 0) else 1
    im = ax.imshow(pet_z, cmap="hot", vmin=0, vmax=vmax, interpolation="nearest", origin="lower")
    if np.any(mask_z):
        ax.contour(mask_z, levels=[0.5], colors="cyan", linewidths=2)
    if np.any(sph_z):
        ax.contour(sph_z, levels=[0.5], colors="yellow", linewidths=2, linestyles="dashed")
    ax.plot(cx - x0, cy - y0, "w+", markersize=12, markeredgewidth=2)
    left_label, right_label, top_label = _axial_orientation(affine)
    for x, lab in ((0.02, left_label), (0.95, right_label)):
        ax.text(x, 0.5, lab, transform=ax.transAxes, color="white", fontsize=12, fontweight="bold")
    ax.text(0.5, 0.95, top_label, transform=ax.transAxes, color="white", fontsize=10, ha="center")
    text = (f"z-slice: {cz}\nSUVmax: {metrics.get('SUVmax', np.nan):.2f}\nSUVpeak: {metrics.get('SUVpeak_2mm', np.nan):.2f}"
            f"\nMask: {int(metrics.get('mask_volume_voxels', 0))} vox\nSphere: {int(metrics.get('sphere_voxel_count', 0))} vox")
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=9, va="top", color="white",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))
    ax.axis("off")
    ax.set_title(title, fontsize=12)
    return im


def generate_session_qc_image(pet_data, masks_by_physical_eye: Dict[str, Optional[np.ndarray]],
                              metrics_by_physical_eye: Dict[str, Optional[Dict]], affine, voxel_dims,
                              subject_id: str, session: str, sphere_radius_mm: float = 2.0, zoom_size: int = 30):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, eye in zip(axes, ("right", "left")):   # radiological layout: subject's right on screen-left
        m = metrics_by_physical_eye.get(eye)
        label = m.get("mask_label_in_filename", "?") if m else "?"
        im = _panel(ax, pet_data, masks_by_physical_eye.get(eye), m, affine, voxel_dims, sphere_radius_mm, zoom_size,
                    f"{eye.capitalize()} eye (file: {label})")
        if im is not None:
            cb = plt.colorbar(im, ax=ax, shrink=0.8)
            cb.set_label("Intensity (Bq/mL, decay-corrected to injection)", fontsize=9)
    handles = [mpatches.Patch(facecolor="none", edgecolor="cyan", linewidth=2, label="ONH mask boundary"),
               mpatches.Patch(facecolor="none", edgecolor="yellow", linewidth=2, linestyle="--", label=f"{sphere_radius_mm:g} mm SUVpeak sphere")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(f"{subject_id} - {session} (radiological display: subject's right on the left)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    return fig


def save_session_qc_image(pet_data, masks_by_physical_eye, metrics_by_physical_eye, affine, voxel_dims,
                          subject_id, session, output_dir: Path, sphere_radius_mm: float = 2.0) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = generate_session_qc_image(pet_data, masks_by_physical_eye, metrics_by_physical_eye, affine, voxel_dims,
                                    subject_id, session, sphere_radius_mm)
    out = output_dir / f"{subject_id}_{session}_SUVpeak_QC.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out
