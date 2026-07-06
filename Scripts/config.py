"""Configuration for KD-only WSI-level stromal TILs scoring.

Edit this file only if you want to change the default pipeline behavior.
Most users can run ``run_pipeline.py`` and override paths from the command line.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    # Folder layout
    input_dir: Path = REPO_ROOT / "Inputs"
    model_dir: Path = REPO_ROOT / "Models"
    output_dir: Path = REPO_ROOT / "Outputs"

    # KD FastSCNN weight filenames expected inside Models/
    tumor_stroma_weights: Path = REPO_ROOT / "Models" / "tumor_stroma_fastscnn_kd.weights.h5"
    tils_weights: Path = REPO_ROOT / "Models" / "tils_fastscnn_kd.weights.h5"

    # WSI extensions
    wsi_extensions: List[str] = field(default_factory=lambda: [".tif", ".tiff", ".svs", ".ndpi", ".mrxs"])

    # Reproducibility and execution
    seed: int = 42
    overwrite_existing: bool = False
    use_direct_model_call: bool = True
    max_windows_to_evaluate: Optional[int] = None  # set an integer for debugging

    # Model configuration
    ts_n_classes: int = 3       # background, tumor, stroma
    tils_n_classes: int = 2     # background, TILs

    # Resolution and inference
    target_mpp: float = 0.25
    fallback_base_mpp: float = 0.25
    roi_size_target_px: int = 2048
    roi_stride_target_px: int = 2048
    patch_size: int = 256
    patch_stride: int = 192
    inference_batch_size: int = 32
    tils_threshold: float = 0.5

    # ROI selection thresholds
    direct_score_threshold: float = 0.80
    score_threshold_stroma_waiting: float = 0.45
    min_direct_stroma_fraction: float = 0.10
    min_direct_tumor_fraction: float = 0.01
    min_direct_interface_density: float = 0.0003
    min_waiting_stroma_fraction: float = 0.30
    max_waiting_tumor_fraction: float = 0.01
    neighbor_min_tumor_fraction: float = 0.03
    neighbor_min_interface_density: float = 0.0003
    peritumoral_radius_um: float = 150.0

    # ROI selection score weights
    weight_peritumoral_stroma: float = 0.40
    weight_stroma_fraction: float = 0.25
    weight_interface_density: float = 0.20
    weight_tissue_fraction: float = 0.10
    weight_tumor_stroma_balance: float = 0.05
    interface_density_scale: float = 0.01

    # Tissue mask filtering
    thumbnail_max_dim: int = 3000
    tissue_open_close_kernel_size: int = 5
    use_eroded_tissue_mask_for_filter: bool = True
    tissue_erosion_kernel_size: int = 5
    tissue_mask_erosion_iterations: int = 1
    min_tissue_fraction: float = 0.60
    min_center_tissue_fraction: float = 0.50
    grid_size: int = 4
    min_grid_cell_tissue_fraction: float = 0.25
    min_tissue_grid_pass_cells: int = 6
    min_mean_saturation_in_tissue: float = 15.0
    max_mean_value_in_tissue: float = 250.0

    # Area-based stromal TILs scoring
    til_diameter_um: float = 8.0
    max_roi_tils_score: float = 0.90
    min_til_component_area_px: int = 5
    max_til_component_area_px: Optional[int] = None
    apply_tils_mask_opening: bool = False
    tils_opening_kernel_size: int = 2

    # Watershed separation of connected TIL clusters
    apply_tils_watershed: bool = True
    watershed_distance_threshold_rel: float = 0.35
    watershed_background_dilation_iterations: int = 2

    # Saving options
    save_selected_roi_images: bool = True
    save_selected_masks: bool = True
    save_overlays: bool = True
