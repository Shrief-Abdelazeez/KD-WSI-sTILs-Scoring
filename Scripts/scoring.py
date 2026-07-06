"""Area-based ROI and WSI-level stromal TILs scoring utilities."""

import cv2
import numpy as np


def count_binary_components(mask):
    mask = (mask > 0).astype(np.uint8)
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return int(max(0, num_labels - 1))


def apply_watershed_to_tils_mask(tils_mask, config):
    mask = (tils_mask > 0).astype(np.uint8)
    stats = {
        "tils_watershed_applied": bool(config.apply_tils_watershed),
        "tils_raw_components_before_watershed": count_binary_components(mask),
        "tils_components_after_watershed": count_binary_components(mask),
    }
    if not config.apply_tils_watershed or int(mask.sum()) == 0:
        return mask, stats
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_dist = float(dist.max()) if dist.size > 0 else 0.0
    if max_dist <= 0.0:
        return mask, stats
    _, sure_fg = cv2.threshold(dist, config.watershed_distance_threshold_rel * max_dist, 1, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    if int(sure_fg.sum()) == 0:
        return mask, stats
    kernel = np.ones((3, 3), dtype=np.uint8)
    sure_bg = cv2.dilate(mask, kernel, iterations=config.watershed_background_dilation_iterations)
    unknown = ((sure_bg > 0) & (sure_fg == 0)).astype(np.uint8)
    num_markers, markers = cv2.connectedComponents(sure_fg, connectivity=8)
    if num_markers <= 1:
        return mask, stats
    markers = markers.astype(np.int32) + 1
    markers[unknown > 0] = 0
    ws_img = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(ws_img, markers)
    separated = (markers > 1).astype(np.uint8)
    stats["tils_components_after_watershed"] = count_binary_components(separated)
    return separated, stats


def postprocess_tils_inside_stroma_mask(mask, config):
    mask = mask.astype(np.uint8)
    if config.apply_tils_mask_opening:
        kernel = np.ones((config.tils_opening_kernel_size, config.tils_opening_kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask.astype(np.uint8)


def count_tils_components(tils_inside_stroma, config):
    mask = postprocess_tils_inside_stroma_mask(tils_inside_stroma, config)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    count, kept_area_px, removed_small, removed_large = 0, 0, 0, 0
    clean_mask = np.zeros_like(mask, dtype=np.uint8)
    for lab in range(1, num_labels):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < config.min_til_component_area_px:
            removed_small += 1
            continue
        if config.max_til_component_area_px is not None and area > config.max_til_component_area_px:
            removed_large += 1
            continue
        count += 1
        kept_area_px += area
        clean_mask[labels == lab] = 1
    return {
        "num_tils_components": int(count),
        "tils_predicted_area_px_after_filtering": int(kept_area_px),
        "removed_small_components": int(removed_small),
        "removed_large_components": int(removed_large),
        "clean_tils_inside_stroma_mask": clean_mask,
    }


def compute_area_based_roi_tils_score(stroma_binary, tils_inside_stroma, effective_mpp_x, effective_mpp_y, config):
    eps = 1e-7
    stroma_area_px = int(np.sum(stroma_binary > 0))
    component_info = count_tils_components(tils_inside_stroma, config)
    n_tils = int(component_info["num_tils_components"])
    til_bbox_width_px = float(config.til_diameter_um / effective_mpp_x)
    til_bbox_height_px = float(config.til_diameter_um / effective_mpp_y)
    area_one_til_px = float(til_bbox_width_px * til_bbox_height_px)
    estimated_all_tils_area_px = float(n_tils * area_one_til_px)
    raw_score = float(estimated_all_tils_area_px / (stroma_area_px + eps))
    capped_score = float(min(raw_score, config.max_roi_tils_score))
    info = {
        "stroma_area_px": stroma_area_px,
        "num_tils_inside_stroma": n_tils,
        "effective_mpp_x_for_scoring": float(effective_mpp_x),
        "effective_mpp_y_for_scoring": float(effective_mpp_y),
        "til_diameter_um": float(config.til_diameter_um),
        "til_bbox_width_px": til_bbox_width_px,
        "til_bbox_height_px": til_bbox_height_px,
        "area_one_til_px": area_one_til_px,
        "estimated_all_tils_area_px": estimated_all_tils_area_px,
        "roi_tils_score_raw_fraction": raw_score,
        "roi_tils_score_fraction": capped_score,
        "roi_tils_score_percent": capped_score * 100.0,
        "roi_tils_score_was_capped": bool(raw_score > config.max_roi_tils_score),
        "tils_predicted_area_px_after_filtering": component_info["tils_predicted_area_px_after_filtering"],
        "removed_small_components": component_info["removed_small_components"],
        "removed_large_components": component_info["removed_large_components"],
        "clean_tils_inside_stroma_mask": component_info["clean_tils_inside_stroma_mask"],
    }
    return info


def safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def safe_weighted_mean(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return safe_mean(values)
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


def aggregate_roi_scores_to_wsi(df_roi_scores, config):
    if df_roi_scores is None or len(df_roi_scores) == 0:
        return {
            "score_mean_fraction": 0.0,
            "score_stroma_weighted_mean_fraction": 0.0,
            "score_median_fraction": 0.0,
        }
    scores = df_roi_scores["roi_tils_score_fraction"].fillna(0).to_numpy(dtype=np.float64)
    scores = np.clip(scores, 0.0, config.max_roi_tils_score)
    if "stroma_area_px" in df_roi_scores.columns:
        weights = df_roi_scores["stroma_area_px"].fillna(0).to_numpy(dtype=np.float64)
    else:
        weights = np.ones_like(scores)
    mean_score = safe_mean(scores)
    swmean_score = safe_weighted_mean(scores, weights)
    median_score = float(np.median(scores)) if scores.size else 0.0
    return {
        "score_mean_fraction": mean_score,
        "score_mean_percent": mean_score * 100.0,
        "score_stroma_weighted_mean_fraction": swmean_score,
        "score_stroma_weighted_mean_percent": swmean_score * 100.0,
        "score_median_fraction": median_score,
        "score_median_percent": median_score * 100.0,
    }
