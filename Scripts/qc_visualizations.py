#!/usr/bin/env python3
"""
QC Visualization Module for ONH FDG-PET Extraction

Generates visual QC images showing:
- Axial PET slices at the SUVmax location
- ONH mask overlay
- 2mm SUVpeak sphere overlay
- Side-by-side comparison of left and right eyes

Part of the ERAP retinal rapamycin trial analysis.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/headless use
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches


def create_sphere_mask_for_viz(center: Tuple[int, int, int],
                                radius_mm: float,
                                voxel_dims: np.ndarray,
                                image_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Create a spherical mask centered at given coordinates.

    Args:
        center: (x, y, z) voxel coordinates of sphere center
        radius_mm: Radius of sphere in mm
        voxel_dims: Voxel dimensions (dx, dy, dz) in mm
        image_shape: Shape of the image volume

    Returns:
        Boolean mask array of the same shape as the image
    """
    cx, cy, cz = center
    radius_vox = radius_mm / voxel_dims

    x_range = int(np.ceil(radius_vox[0])) + 1
    y_range = int(np.ceil(radius_vox[1])) + 1
    z_range = int(np.ceil(radius_vox[2])) + 1

    mask = np.zeros(image_shape, dtype=bool)

    for dx in range(-x_range, x_range + 1):
        for dy in range(-y_range, y_range + 1):
            for dz in range(-z_range, z_range + 1):
                x, y, z = cx + dx, cy + dy, cz + dz
                if 0 <= x < image_shape[0] and 0 <= y < image_shape[1] and 0 <= z < image_shape[2]:
                    dist = np.sqrt(
                        (dx * voxel_dims[0])**2 +
                        (dy * voxel_dims[1])**2 +
                        (dz * voxel_dims[2])**2
                    )
                    if dist <= radius_mm:
                        mask[x, y, z] = True

    return mask


def generate_qc_image(pet_data: np.ndarray,
                      mask_data: np.ndarray,
                      voxel_dims: np.ndarray,
                      max_coords: Tuple[int, int, int],
                      sphere_radius_mm: float = 2.0,
                      suv_max: float = None,
                      suv_peak: float = None,
                      zoom_size: int = 40) -> plt.Figure:
    """
    Generate a single QC image for one eye showing axial slice with overlays.

    Args:
        pet_data: 3D PET image array (in SUV units if available)
        mask_data: 3D binary ONH mask array
        voxel_dims: Voxel dimensions in mm
        max_coords: (x, y, z) coordinates of SUVmax voxel
        sphere_radius_mm: Radius of SUVpeak sphere
        suv_max: SUVmax value for annotation
        suv_peak: SUVpeak value for annotation
        zoom_size: Half-width of zoomed region in voxels

    Returns:
        matplotlib Figure object
    """
    # Get the axial slice at max z coordinate
    z_slice = max_coords[2]

    # Extract 2D slices
    pet_slice = pet_data[:, :, z_slice].T  # Transpose for proper orientation
    mask_slice = mask_data[:, :, z_slice].T

    # Create sphere mask and get its slice
    sphere_mask = create_sphere_mask_for_viz(max_coords, sphere_radius_mm, voxel_dims, pet_data.shape)
    sphere_slice = sphere_mask[:, :, z_slice].T

    # Calculate zoom bounds centered on max voxel
    cx, cy = max_coords[0], max_coords[1]
    x_min = max(0, cx - zoom_size)
    x_max = min(pet_data.shape[0], cx + zoom_size)
    y_min = max(0, cy - zoom_size)
    y_max = min(pet_data.shape[1], cy + zoom_size)

    # Extract zoomed regions
    pet_zoomed = pet_slice[y_min:y_max, x_min:x_max]
    mask_zoomed = mask_slice[y_min:y_max, x_min:x_max]
    sphere_zoomed = sphere_slice[y_min:y_max, x_min:x_max]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Display PET image with hot colormap
    vmax = np.percentile(pet_zoomed[pet_zoomed > 0], 99) if np.any(pet_zoomed > 0) else 1
    vmin = 0
    im = ax.imshow(pet_zoomed, cmap='hot', vmin=vmin, vmax=vmax, interpolation='nearest')

    # Overlay mask contour (cyan)
    if np.any(mask_zoomed):
        ax.contour(mask_zoomed, levels=[0.5], colors='cyan', linewidths=2, linestyles='solid')

    # Overlay sphere contour (yellow)
    if np.any(sphere_zoomed):
        ax.contour(sphere_zoomed, levels=[0.5], colors='yellow', linewidths=2, linestyles='dashed')

    # Mark the max voxel with a crosshair
    local_cx = cx - x_min
    local_cy = cy - y_min
    ax.axhline(y=local_cy, color='white', linewidth=0.5, alpha=0.5)
    ax.axvline(x=local_cx, color='white', linewidth=0.5, alpha=0.5)
    ax.plot(local_cx, local_cy, 'w+', markersize=10, markeredgewidth=2)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('SUV', fontsize=10)

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='none', edgecolor='cyan', linewidth=2, label='ONH mask'),
        mpatches.Patch(facecolor='none', edgecolor='yellow', linewidth=2, linestyle='--', label='2mm sphere')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    # Add annotations
    annotation_text = f"z={z_slice}"
    if suv_max is not None:
        annotation_text += f"\nSUVmax: {suv_max:.2f}"
    if suv_peak is not None:
        annotation_text += f"\nSUVpeak: {suv_peak:.2f}"

    ax.text(0.02, 0.98, annotation_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
            color='white')

    ax.axis('off')

    return fig


def generate_session_qc_image(pet_data: np.ndarray,
                               left_mask: Optional[np.ndarray],
                               right_mask: Optional[np.ndarray],
                               voxel_dims: np.ndarray,
                               left_metrics: Optional[Dict],
                               right_metrics: Optional[Dict],
                               subject_id: str,
                               session: str,
                               sphere_radius_mm: float = 2.0,
                               zoom_size: int = 30) -> plt.Figure:
    """
    Generate a combined QC image showing both eyes for a session.

    Args:
        pet_data: 3D PET image array
        left_mask: 3D binary mask for left eye (or None)
        right_mask: 3D binary mask for right eye (or None)
        voxel_dims: Voxel dimensions in mm
        left_metrics: Dictionary with left eye metrics (or None)
        right_metrics: Dictionary with right eye metrics (or None)
        subject_id: Subject ID for title
        session: Session name (Baseline/Followup) for title
        sphere_radius_mm: Radius of SUVpeak sphere
        zoom_size: Half-width of zoomed region in voxels

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Process each eye
    for idx, (mask_data, metrics, eye, ax) in enumerate([
        (left_mask, left_metrics, 'Left', axes[0]),
        (right_mask, right_metrics, 'Right', axes[1])
    ]):
        if mask_data is None or metrics is None or 'error' in metrics:
            # No data for this eye
            ax.text(0.5, 0.5, f'{eye} Eye\nNo data',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=14, color='gray')
            ax.set_facecolor('#1a1a1a')
            ax.axis('off')
            ax.set_title(f'{eye} Eye - Missing', fontsize=12)
            continue

        # Get max coordinates
        max_coords = (metrics['max_voxel_x'], metrics['max_voxel_y'], metrics['max_voxel_z'])
        z_slice = max_coords[2]

        # Extract 2D slices
        pet_slice = pet_data[:, :, z_slice].T
        mask_slice = mask_data[:, :, z_slice].T

        # Create sphere mask
        sphere_mask = create_sphere_mask_for_viz(max_coords, sphere_radius_mm, voxel_dims, pet_data.shape)
        sphere_slice = sphere_mask[:, :, z_slice].T

        # Calculate zoom bounds
        cx, cy = max_coords[0], max_coords[1]
        x_min = max(0, cx - zoom_size)
        x_max = min(pet_data.shape[0], cx + zoom_size)
        y_min = max(0, cy - zoom_size)
        y_max = min(pet_data.shape[1], cy + zoom_size)

        # Extract zoomed regions
        pet_zoomed = pet_slice[y_min:y_max, x_min:x_max]
        mask_zoomed = mask_slice[y_min:y_max, x_min:x_max]
        sphere_zoomed = sphere_slice[y_min:y_max, x_min:x_max]

        # Display PET image
        vmax = np.percentile(pet_zoomed[pet_zoomed > 0], 99) if np.any(pet_zoomed > 0) else 1
        im = ax.imshow(pet_zoomed, cmap='hot', vmin=0, vmax=vmax, interpolation='nearest')

        # Overlay mask contour (cyan)
        if np.any(mask_zoomed):
            ax.contour(mask_zoomed, levels=[0.5], colors='cyan', linewidths=2, linestyles='solid')

        # Overlay sphere contour (yellow)
        if np.any(sphere_zoomed):
            ax.contour(sphere_zoomed, levels=[0.5], colors='yellow', linewidths=2, linestyles='dashed')

        # Mark the max voxel
        local_cx = cx - x_min
        local_cy = cy - y_min
        ax.axhline(y=local_cy, color='white', linewidth=0.5, alpha=0.5)
        ax.axvline(x=local_cx, color='white', linewidth=0.5, alpha=0.5)
        ax.plot(local_cx, local_cy, 'w+', markersize=12, markeredgewidth=2)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Intensity (Bq/mL)', fontsize=9)

        # Get SUV values if available
        suv_max = metrics.get('SUVmax', metrics.get('intensity_max', None))
        suv_peak = metrics.get('SUVpeak', metrics.get('intensity_peak', None))
        mask_vol = metrics.get('mask_volume_voxels', 0)
        sphere_count = metrics.get('sphere_voxel_count', 0)

        # Add annotations
        annotation_text = f"z-slice: {z_slice}"
        if suv_max is not None:
            annotation_text += f"\nSUVmax: {suv_max:.2f}"
        if suv_peak is not None:
            annotation_text += f"\nSUVpeak: {suv_peak:.2f}"
        annotation_text += f"\nMask: {mask_vol} vox"
        annotation_text += f"\nSphere: {sphere_count} vox"

        ax.text(0.02, 0.98, annotation_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                color='white')

        ax.axis('off')
        ax.set_title(f'{eye} Eye', fontsize=12)

    # Add legend at the bottom
    legend_elements = [
        mpatches.Patch(facecolor='none', edgecolor='cyan', linewidth=2, label='ONH mask boundary'),
        mpatches.Patch(facecolor='none', edgecolor='yellow', linewidth=2, linestyle='--', label='2mm SUVpeak sphere')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, 0.02))

    # Add title
    fig.suptitle(f'{subject_id} - {session}', fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    return fig


def save_qc_images_for_subject(pet_data: np.ndarray,
                                masks: Dict[str, Optional[np.ndarray]],
                                voxel_dims: np.ndarray,
                                metrics: Dict[str, Optional[Dict]],
                                subject_id: str,
                                session: str,
                                output_dir: Path,
                                sphere_radius_mm: float = 2.0) -> Path:
    """
    Save QC images for a subject session.

    Args:
        pet_data: 3D PET image array
        masks: Dict with 'left' and 'right' mask arrays (or None)
        voxel_dims: Voxel dimensions in mm
        metrics: Dict with 'left' and 'right' metrics dicts (or None)
        subject_id: Subject ID
        session: Session name (Baseline/Followup)
        output_dir: Directory to save images
        sphere_radius_mm: Radius of SUVpeak sphere

    Returns:
        Path to saved image file
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate combined image for both eyes
    fig = generate_session_qc_image(
        pet_data=pet_data,
        left_mask=masks.get('left'),
        right_mask=masks.get('right'),
        voxel_dims=voxel_dims,
        left_metrics=metrics.get('left'),
        right_metrics=metrics.get('right'),
        subject_id=subject_id,
        session=session,
        sphere_radius_mm=sphere_radius_mm
    )

    # Save figure
    output_file = output_dir / f"{subject_id}_{session}_SUVpeak_QC.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return output_file


def generate_all_qc_visualizations(project_root: Path,
                                    metrics_df,
                                    log_func=None,
                                    analysis_dir: Path = None) -> int:
    """
    Generate QC visualizations for all subjects/sessions in the metrics dataframe.

    This function re-loads the PET and mask data to generate visualizations.

    Args:
        project_root: Path to project root directory (where RawData lives)
        metrics_df: DataFrame with extraction results
        log_func: Optional logging function
        analysis_dir: Path to analysis directory (where QC folder lives).
                      If None, defaults to project_root / "ONH_Analysis"

    Returns:
        Number of images generated
    """
    import sys
    from pathlib import Path

    # Import utilities from the same directory
    script_dir = Path(__file__).parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    from utils import (
        load_blinding_key,
        find_pet_file,
        find_mask_file,
        load_nifti_with_scaling,
        get_voxel_dimensions
    )

    def log(msg):
        if log_func:
            log_func(msg)
        else:
            print(msg)

    # Set analysis_dir if not provided
    if analysis_dir is None:
        analysis_dir = project_root / "ONH_Analysis"

    rawdata_dir = project_root / "RawData"
    qc_viz_dir = analysis_dir / "QC" / "SUVpeak_visualizations"
    qc_viz_dir.mkdir(parents=True, exist_ok=True)

    # Load blinding key
    # Format is: {(subject_id, blinded_session): timepoint}
    try:
        blinding_map = load_blinding_key(project_root)
    except FileNotFoundError:
        log("ERROR: Could not load blinding key for QC visualizations")
        return 0

    # Create reverse mapping: (subject, timepoint) -> blinded session
    reverse_blinding = {}
    for (subj, blinded_sess), timepoint in blinding_map.items():
        reverse_blinding[(subj, timepoint)] = blinded_sess

    # Get unique subject/session combinations
    unique_sessions = metrics_df[['subject_id', 'session_unblinded']].drop_duplicates()

    images_generated = 0

    for _, row in unique_sessions.iterrows():
        subject_id = row['subject_id']
        timepoint = row['session_unblinded']

        # Get blinded session ID
        blinded_session = reverse_blinding.get((subject_id, timepoint))
        if not blinded_session:
            log(f"  WARNING: Could not find blinded session for {subject_id}/{timepoint}")
            continue

        # Find PET file
        subject_dir = rawdata_dir / subject_id / blinded_session
        pet_dir = subject_dir / "pet"

        if not pet_dir.exists():
            log(f"  WARNING: PET directory not found for {subject_id}/{blinded_session}")
            continue

        pet_file = find_pet_file(pet_dir, subject_id, blinded_session)
        if not pet_file:
            log(f"  WARNING: PET file not found for {subject_id}/{blinded_session}")
            continue

        # Load PET data
        try:
            pet_data, pet_img = load_nifti_with_scaling(pet_file)
            voxel_dims = get_voxel_dimensions(pet_img)
        except Exception as e:
            log(f"  WARNING: Could not load PET data for {subject_id}/{blinded_session}: {e}")
            continue

        # Load masks and get metrics for each eye
        masks = {}
        eye_metrics = {}

        for eye in ['left', 'right']:
            mask_file = find_mask_file(pet_dir, subject_id, blinded_session, eye)

            if mask_file and mask_file.exists():
                try:
                    masks[eye], _ = load_nifti_with_scaling(mask_file)
                except Exception as e:
                    log(f"  WARNING: Could not load {eye} mask for {subject_id}/{blinded_session}: {e}")
                    masks[eye] = None
            else:
                masks[eye] = None

            # Get metrics from dataframe
            eye_row = metrics_df[(metrics_df['subject_id'] == subject_id) &
                                 (metrics_df['session_unblinded'] == timepoint) &
                                 (metrics_df['eye'] == eye)]

            if len(eye_row) > 0:
                eye_metrics[eye] = eye_row.iloc[0].to_dict()
            else:
                eye_metrics[eye] = None

        # Generate and save QC image
        try:
            # Create subject subfolder
            subject_output_dir = qc_viz_dir / subject_id

            output_file = save_qc_images_for_subject(
                pet_data=pet_data,
                masks=masks,
                voxel_dims=voxel_dims,
                metrics=eye_metrics,
                subject_id=subject_id,
                session=timepoint,
                output_dir=subject_output_dir
            )

            images_generated += 1
            log(f"  Generated: {output_file.name}")

        except Exception as e:
            log(f"  ERROR generating QC image for {subject_id}/{timepoint}: {e}")

    return images_generated


if __name__ == "__main__":
    # Test/standalone execution
    import sys

    # Script is at: ONH_Analysis/Scripts/qc_visualizations.py
    script_dir = Path(__file__).parent          # ONH_Analysis/Scripts/
    analysis_dir = script_dir.parent            # ONH_Analysis/
    project_root = analysis_dir.parent          # ERAP_FDG_ONH_periodontium_analysis/

    metrics_file = analysis_dir / "Outputs" / "ONH_FDG_metrics.csv"

    if not metrics_file.exists():
        print(f"Metrics file not found: {metrics_file}")
        print("Run extract_onh_metrics.py first to generate metrics.")
        sys.exit(1)

    import pandas as pd
    metrics_df = pd.read_csv(metrics_file)

    print("Generating QC visualizations...")
    n_images = generate_all_qc_visualizations(project_root, metrics_df, analysis_dir=analysis_dir)
    print(f"Generated {n_images} QC images")
    print(f"Output directory: {analysis_dir / 'QC' / 'SUVpeak_visualizations'}")
