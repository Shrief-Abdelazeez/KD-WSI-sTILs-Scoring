"""Saving masks, overlays, and QuPath-compatible ROI GeoJSON outputs."""

import json
from pathlib import Path
import cv2
import numpy as np


def colorize_tumor_stroma_mask(mask):
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color[mask == 1] = (255, 0, 0)   # tumor: red
    color[mask == 2] = (0, 255, 0)   # stroma: green
    return color


def colorize_tils_mask(mask):
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color[mask == 1] = (255, 255, 0) # TILs: yellow
    return color


def colorize_binary_mask(mask, color_rgb):
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color[mask == 1] = color_rgb
    return color


def overlay_mask_on_image(image_rgb, mask_color, alpha=0.40):
    return cv2.addWeighted(image_rgb.astype(np.uint8), 1.0 - alpha, mask_color.astype(np.uint8), alpha, 0)


def save_selected_roi_outputs(roi_rgb, ts_mask, tils_mask, stroma_binary, tils_inside_stroma,
                              clean_tils_inside_stroma, roi_rank, row, dirs, config):
    if not (config.save_selected_roi_images or config.save_selected_masks or config.save_overlays):
        return
    decision_tag = "D" if str(row.get("final_decision", "")) == "direct_accept" else "N"
    base = f"r{roi_rank:03d}_{decision_tag}_gx{int(row['grid_x'])}_gy{int(row['grid_y'])}"

    if config.save_selected_roi_images:
        cv2.imwrite(str(dirs["roi_images"] / f"{base}_img.png"), cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR))

    if config.save_selected_masks:
        ts_color = colorize_tumor_stroma_mask(ts_mask)
        tils_color = colorize_tils_mask(tils_mask)
        cv2.imwrite(str(dirs["ts_masks"] / f"{base}_ts_label.png"), ts_mask.astype(np.uint8))
        cv2.imwrite(str(dirs["ts_masks"] / f"{base}_ts_color.png"), cv2.cvtColor(ts_color, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(dirs["tils_masks"] / f"{base}_tils.png"), tils_mask.astype(np.uint8) * 255)
        cv2.imwrite(str(dirs["tils_masks"] / f"{base}_tils_color.png"), cv2.cvtColor(tils_color, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(dirs["stroma_masks"] / f"{base}_stroma.png"), stroma_binary.astype(np.uint8) * 255)
        cv2.imwrite(str(dirs["tils_inside_stroma"] / f"{base}_tis_raw.png"), tils_inside_stroma.astype(np.uint8) * 255)
        cv2.imwrite(str(dirs["tils_inside_stroma"] / f"{base}_tis_clean.png"), clean_tils_inside_stroma.astype(np.uint8) * 255)

    if config.save_overlays:
        ts_color = colorize_tumor_stroma_mask(ts_mask)
        clean_tis_color = colorize_binary_mask(clean_tils_inside_stroma, (255, 255, 0))
        combined_overlay = overlay_mask_on_image(roi_rgb, ts_color, alpha=0.30)
        combined_overlay = overlay_mask_on_image(combined_overlay, clean_tis_color, alpha=0.45)
        cv2.imwrite(str(dirs["overlays"] / f"{base}_overlay.png"), cv2.cvtColor(combined_overlay, cv2.COLOR_RGB2BGR))


def rectangle_polygon(x, y, w, h):
    x, y, w, h = float(x), float(y), float(w), float(h)
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]]


def make_qupath_geojson_feature(row, rank, class_name, color_rgb):
    measurement_keys = [
        "roi_score", "tumor_fraction", "stroma_fraction", "interface_density",
        "stroma_area_px", "num_tils_inside_stroma", "roi_tils_score_fraction",
        "roi_tils_score_percent", "effective_mpp_x_for_scoring", "effective_mpp_y_for_scoring",
    ]
    measurements = []
    for key in measurement_keys:
        if key in row and row[key] == row[key]:
            try:
                measurements.append({"name": key, "value": float(row[key])})
            except Exception:
                pass
    return {
        "type": "Feature",
        "id": f"{class_name}_{rank:03d}",
        "geometry": {"type": "Polygon", "coordinates": [rectangle_polygon(row["x0_level0"], row["y0_level0"], row["w_level0"], row["h_level0"])]},
        "properties": {
            "objectType": "annotation",
            "name": f"ROI_{rank:03d}",
            "classification": {"name": class_name, "color": color_rgb},
            "measurements": measurements,
            "isLocked": False,
            "rank": int(rank),
            "grid_x": int(row["grid_x"]),
            "grid_y": int(row["grid_y"]),
            "final_decision": str(row.get("final_decision", "")),
        },
    }


def save_geojson(df, geojson_path: Path, class_name="selected_TILs_scoring_ROI", color_rgb=(0, 0, 255)):
    geojson_path = Path(geojson_path)
    geojson_path.parent.mkdir(parents=True, exist_ok=True)
    features = [make_qupath_geojson_feature(row, rank, class_name, list(color_rgb))
                for rank, (_, row) in enumerate(df.iterrows(), start=1)]
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)
