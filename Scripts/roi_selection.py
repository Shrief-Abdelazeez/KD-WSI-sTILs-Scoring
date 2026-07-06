"""Tumor/stroma-based ROI selection utilities."""

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, distance_transform_edt, generate_binary_structure


def compute_tumor_stroma_interface(pred_mask):
    tumor = pred_mask == 1
    stroma = pred_mask == 2
    if not tumor.any() or not stroma.any():
        return np.zeros_like(pred_mask, dtype=bool)
    struct = generate_binary_structure(2, 2)
    tumor_dil = binary_dilation(tumor, structure=struct, iterations=1)
    stroma_dil = binary_dilation(stroma, structure=struct, iterations=1)
    return (tumor & stroma_dil) | (stroma & tumor_dil)


def compute_roi_features(pred_mask, tissue_fraction, effective_mpp_mean, config):
    total_pixels = pred_mask.shape[0] * pred_mask.shape[1]
    eps = 1e-7
    tumor = pred_mask == 1
    stroma = pred_mask == 2
    background = pred_mask == 0
    tumor_pixels = int(tumor.sum())
    stroma_pixels = int(stroma.sum())
    background_pixels = int(background.sum())
    tumor_fraction = tumor_pixels / (total_pixels + eps)
    stroma_fraction = stroma_pixels / (total_pixels + eps)
    background_fraction = background_pixels / (total_pixels + eps)
    interface = compute_tumor_stroma_interface(pred_mask)
    interface_pixels = int(interface.sum())
    interface_density = interface_pixels / (total_pixels + eps)

    if tumor.any():
        dist_to_tumor_px = distance_transform_edt(~tumor)
        dist_to_tumor_um = dist_to_tumor_px * effective_mpp_mean
        peritumoral_zone = (dist_to_tumor_um <= config.peritumoral_radius_um) & (~tumor)
        peritumoral_stroma = peritumoral_zone & stroma
        peritumoral_stroma_pixels = int(peritumoral_stroma.sum())
        peritumoral_stroma_fraction = peritumoral_stroma_pixels / (stroma_pixels + eps)
    else:
        peritumoral_stroma_pixels = 0
        peritumoral_stroma_fraction = 0.0

    tumor_stroma_balance = 2.0 * min(tumor_fraction, stroma_fraction)
    return {
        "tissue_fraction": float(tissue_fraction),
        "tumor_fraction": float(tumor_fraction),
        "stroma_fraction": float(stroma_fraction),
        "background_fraction": float(background_fraction),
        "tumor_pixels": tumor_pixels,
        "stroma_pixels": stroma_pixels,
        "background_pixels": background_pixels,
        "interface_pixels": interface_pixels,
        "interface_density": float(interface_density),
        "peritumoral_stroma_pixels": peritumoral_stroma_pixels,
        "peritumoral_stroma_fraction": float(peritumoral_stroma_fraction),
        "tumor_stroma_balance": float(tumor_stroma_balance),
    }


def compute_raw_roi_score(features, config):
    interface_scaled = min(features["interface_density"] / config.interface_density_scale, 1.0)
    score = (
        config.weight_peritumoral_stroma * features["peritumoral_stroma_fraction"]
        + config.weight_stroma_fraction * features["stroma_fraction"]
        + config.weight_interface_density * interface_scaled
        + config.weight_tissue_fraction * features["tissue_fraction"]
        + config.weight_tumor_stroma_balance * features["tumor_stroma_balance"]
    )
    return float(np.clip(score, 0.0, 1.0))


def get_neighbor_keys(grid_x, grid_y):
    keys = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            keys.append((grid_x + dx, grid_y + dy))
    return keys


def classify_selected_rois(df, config):
    out = df.copy()
    out["initial_decision"] = "discard"
    out["final_decision"] = "discard"
    out["best_neighbor_tumor_fraction"] = 0.0
    out["best_neighbor_interface_density"] = 0.0

    direct_mask = (
        (out["tissue_fraction"] >= config.min_tissue_fraction)
        & (out["stroma_fraction"] >= config.min_direct_stroma_fraction)
        & (out["tumor_fraction"] >= config.min_direct_tumor_fraction)
        & (out["interface_density"] >= config.min_direct_interface_density)
        & (out["roi_score"] >= config.direct_score_threshold)
    )
    waiting_mask = (
        (out["tissue_fraction"] >= config.min_tissue_fraction)
        & (out["stroma_fraction"] >= config.min_waiting_stroma_fraction)
        & (out["tumor_fraction"] < config.max_waiting_tumor_fraction)
        & (out["roi_score"] >= config.score_threshold_stroma_waiting)
        & (~direct_mask)
    )
    out.loc[direct_mask, "initial_decision"] = "direct_accept"
    out.loc[direct_mask, "final_decision"] = "direct_accept"
    out.loc[waiting_mask, "initial_decision"] = "stroma_waiting_for_neighbor"

    roi_lookup = {(int(row["grid_x"]), int(row["grid_y"])): idx for idx, row in out.iterrows()}
    waiting_indices = out.index[out["initial_decision"] == "stroma_waiting_for_neighbor"].tolist()

    for idx in waiting_indices:
        row = out.loc[idx]
        neighbor_support = False
        best_neighbor_tumor_fraction = 0.0
        best_neighbor_interface_density = 0.0
        for key in get_neighbor_keys(int(row["grid_x"]), int(row["grid_y"])):
            if key not in roi_lookup:
                continue
            nrow = out.loc[roi_lookup[key]]
            n_tumor = float(nrow.get("tumor_fraction", 0.0))
            n_interface = float(nrow.get("interface_density", 0.0))
            best_neighbor_tumor_fraction = max(best_neighbor_tumor_fraction, n_tumor)
            best_neighbor_interface_density = max(best_neighbor_interface_density, n_interface)
            if n_tumor >= config.neighbor_min_tumor_fraction or n_interface >= config.neighbor_min_interface_density:
                neighbor_support = True
        out.loc[idx, "best_neighbor_tumor_fraction"] = best_neighbor_tumor_fraction
        out.loc[idx, "best_neighbor_interface_density"] = best_neighbor_interface_density
        if neighbor_support:
            out.loc[idx, "final_decision"] = "neighbor_associated_stroma"
    return out
