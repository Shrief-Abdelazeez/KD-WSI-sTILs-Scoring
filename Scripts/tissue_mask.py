"""Thumbnail-level tissue mask creation and candidate ROI filtering."""

import cv2
import numpy as np
from scipy.ndimage import binary_erosion


def create_tissue_mask_thumbnail(slide, output_dir, config):
    w0, h0 = slide.dimensions
    scale = config.thumbnail_max_dim / max(w0, h0)
    thumb_w = int(round(w0 * scale))
    thumb_h = int(round(h0 * scale))
    thumbnail = slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB")
    thumb_rgb = np.array(thumbnail)

    hsv = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1]
    val = hsv[..., 2]
    otsu_thr, _ = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sat_threshold = max(20, int(0.8 * otsu_thr))

    tissue_mask = ((sat > sat_threshold) & (val < 245)).astype(np.uint8)
    kernel = np.ones((config.tissue_open_close_kernel_size, config.tissue_open_close_kernel_size), dtype=np.uint8)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(tissue_mask, connectivity=8)
    min_area = int(0.0005 * tissue_mask.shape[0] * tissue_mask.shape[1])
    clean = np.zeros_like(tissue_mask, dtype=np.uint8)
    for lab in range(1, num_labels):
        if stats[lab, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == lab] = 1

    if config.use_eroded_tissue_mask_for_filter:
        erosion_structure = np.ones((config.tissue_erosion_kernel_size, config.tissue_erosion_kernel_size), dtype=bool)
        eroded = binary_erosion(clean.astype(bool), structure=erosion_structure,
                                iterations=config.tissue_mask_erosion_iterations).astype(np.uint8)
    else:
        eroded = clean.copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "thumbnail_rgb.png"), cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "tissue_mask_thumbnail.png"), clean * 255)
    cv2.imwrite(str(output_dir / "tissue_mask_thumbnail_eroded.png"), eroded * 255)
    return clean, eroded, thumb_rgb, thumb_rgb.shape[1], thumb_rgb.shape[0]


def level0_rect_to_thumbnail_rect(slide_w0, slide_h0, mask_w, mask_h, x0, y0, w0, h0):
    x1 = x0 + w0
    y1 = y0 + h0
    mx0 = int(np.floor(x0 / slide_w0 * mask_w))
    my0 = int(np.floor(y0 / slide_h0 * mask_h))
    mx1 = int(np.ceil(x1 / slide_w0 * mask_w))
    my1 = int(np.ceil(y1 / slide_h0 * mask_h))
    mx0 = int(np.clip(mx0, 0, mask_w - 1))
    my0 = int(np.clip(my0, 0, mask_h - 1))
    mx1 = int(np.clip(mx1, mx0 + 1, mask_w))
    my1 = int(np.clip(my1, my0 + 1, mask_h))
    return mx0, my0, mx1, my1


def strong_tissue_filter_for_level0_rect(tissue_mask, eroded_tissue_mask, thumb_rgb, mask_w, mask_h,
                                         slide_w0, slide_h0, x0, y0, w0, h0, config):
    mx0, my0, mx1, my1 = level0_rect_to_thumbnail_rect(slide_w0, slide_h0, mask_w, mask_h, x0, y0, w0, h0)
    patch_mask = tissue_mask[my0:my1, mx0:mx1]
    patch_eroded = eroded_tissue_mask[my0:my1, mx0:mx1]
    patch_rgb = thumb_rgb[my0:my1, mx0:mx1]

    if patch_mask.size == 0:
        return False, {"filter_global_tissue_fraction": 0.0, "filter_reason": "empty_thumbnail_patch"}

    filter_mask = patch_eroded if config.use_eroded_tissue_mask_for_filter else patch_mask
    global_tissue_fraction = float(np.mean(filter_mask > 0))

    ph, pw = filter_mask.shape[:2]
    center_patch = filter_mask[int(ph * 0.25):int(ph * 0.75), int(pw * 0.25):int(pw * 0.75)]
    center_tissue_fraction = float(np.mean(center_patch > 0)) if center_patch.size > 0 else 0.0

    grid_fractions = []
    for gy in range(config.grid_size):
        for gx in range(config.grid_size):
            y_start = int(round(gy * ph / config.grid_size))
            y_end = int(round((gy + 1) * ph / config.grid_size))
            x_start = int(round(gx * pw / config.grid_size))
            x_end = int(round((gx + 1) * pw / config.grid_size))
            cell = filter_mask[y_start:y_end, x_start:x_end]
            grid_fractions.append(float(np.mean(cell > 0)) if cell.size > 0 else 0.0)
    grid_fractions = np.array(grid_fractions, dtype=np.float32)
    grid_pass_cells = int(np.sum(grid_fractions >= config.min_grid_cell_tissue_fraction))
    min_grid_tissue_fraction = float(np.min(grid_fractions)) if len(grid_fractions) else 0.0

    tissue_pixels = patch_rgb[patch_mask > 0]
    if tissue_pixels.size > 0:
        tissue_hsv = cv2.cvtColor(tissue_pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV)
        mean_sat = float(np.mean(tissue_hsv[..., 1]))
        mean_val = float(np.mean(tissue_hsv[..., 2]))
    else:
        mean_sat = 0.0
        mean_val = 255.0

    passed = True
    reason = "pass"
    if global_tissue_fraction < config.min_tissue_fraction:
        passed, reason = False, "low_global_tissue_fraction"
    elif center_tissue_fraction < config.min_center_tissue_fraction:
        passed, reason = False, "low_center_tissue_fraction"
    elif grid_pass_cells < config.min_tissue_grid_pass_cells:
        passed, reason = False, "insufficient_grid_tissue_coverage"
    elif mean_sat < config.min_mean_saturation_in_tissue:
        passed, reason = False, "low_mean_saturation"
    elif mean_val > config.max_mean_value_in_tissue:
        passed, reason = False, "high_mean_value_background_like"

    info = {
        "filter_global_tissue_fraction": global_tissue_fraction,
        "filter_center_tissue_fraction": center_tissue_fraction,
        "filter_grid_pass_cells": grid_pass_cells,
        "filter_min_grid_tissue_fraction": min_grid_tissue_fraction,
        "filter_mean_saturation_tissue": mean_sat,
        "filter_mean_value_tissue": mean_val,
        "filter_reason": reason,
    }
    return passed, info
