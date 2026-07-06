"""KD-only WSI-level stromal TILs scoring pipeline."""

from pathlib import Path
import traceback
import time

import cv2
import numpy as np
import pandas as pd
import openslide

from config import Config
from utils import Timer, clear_memory_soft, progress_iter, safe_slide_id, write_csv_and_excel, append_or_update_summary, list_wsi_files
from wsi_io import get_base_mpp, read_region_at_target_mpp, compute_effective_mpp, generate_candidate_windows
from tissue_mask import create_tissue_mask_thumbnail, strong_tissue_filter_for_level0_rect
from inference import predict_roi_multiclass, predict_roi_tils_mask
from roi_selection import compute_roi_features, compute_raw_roi_score, classify_selected_rois
from scoring import apply_watershed_to_tils_mask, compute_area_based_roi_tils_score, aggregate_roi_scores_to_wsi
from visualization import save_selected_roi_outputs, save_geojson


def make_slide_dirs(output_dir: Path, slide_id: str):
    slide_root = Path(output_dir) / "PerSlideResults" / slide_id
    dirs = {
        "slide_root": slide_root,
        "debug": slide_root / "debug",
        "tables": slide_root / "tables",
        "geojson": slide_root / "geojson",
        "roi_images": slide_root / "ROIs",
        "ts_masks": slide_root / "tumor_stroma_masks",
        "tils_masks": slide_root / "tils_masks",
        "stroma_masks": slide_root / "stroma_masks",
        "tils_inside_stroma": slide_root / "tils_in_stroma",
        "overlays": slide_root / "overlays",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def save_timing_report(timer: Timer, summary: dict, dirs: dict) -> None:
    rows = [{"stage": k, "seconds": v, "formatted": Timer.fmt(v)} for k, v in timer.records.items()]
    rows.append({"stage": "total_pipeline_time", "seconds": timer.total(), "formatted": Timer.fmt(timer.total())})
    pd.DataFrame(rows).to_csv(dirs["tables"] / "timing.csv", index=False)
    lines = ["KD WSI stromal TILs scoring timing report", "=" * 80, ""]
    for row in rows:
        lines.append(f"{row['stage']}: {row['formatted']}")
    lines += ["", "Summary", "=" * 80]
    for k, v in summary.items():
        lines.append(f"{k}: {v}")
    with open(dirs["tables"] / "timing_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def empty_summary(slide_id, wsi_path, status, reason, timer, dirs):
    return {
        "slide_id": slide_id,
        "wsi_path": str(wsi_path),
        "status": status,
        "reason": reason,
        "num_selected_rois": 0,
        "score_mean_fraction": 0.0,
        "score_mean_percent": 0.0,
        "score_stroma_weighted_mean_fraction": 0.0,
        "score_stroma_weighted_mean_percent": 0.0,
        "score_median_fraction": 0.0,
        "score_median_percent": 0.0,
        "total_pipeline_time_seconds": timer.total(),
        "total_pipeline_time_minutes": timer.total() / 60.0,
        "total_pipeline_time_formatted": Timer.fmt(timer.total()),
        "output_folder": str(dirs["slide_root"]),
    }


def process_one_wsi(wsi_path: Path, ts_model, tils_model, config: Config):
    slide_id = safe_slide_id(wsi_path)
    dirs = make_slide_dirs(config.output_dir, slide_id)
    timer = Timer()

    print("\n" + "=" * 120)
    print(f"Processing slide: {slide_id}")
    print("WSI path:", wsi_path)
    print("=" * 120)

    t = timer.now()
    slide = openslide.OpenSlide(str(wsi_path))
    slide_w0, slide_h0 = slide.dimensions
    base_mpp_x, base_mpp_y = get_base_mpp(slide, config.fallback_base_mpp)
    timer.mark("slide_opening", t)

    t = timer.now()
    tissue_mask, eroded_tissue_mask, thumb_rgb, mask_w, mask_h = create_tissue_mask_thumbnail(slide, dirs["debug"], config)
    timer.mark("tissue_mask_creation", t)

    t = timer.now()
    windows = list(generate_candidate_windows(slide, base_mpp_x, base_mpp_y, config))
    if config.max_windows_to_evaluate is not None:
        windows = windows[:config.max_windows_to_evaluate]
    timer.mark("roi_grid_generation", t)
    print("Candidate ROI windows:", len(windows))

    raw_rows = []
    tissue_rejected_rows = []
    t_tissue = 0.0
    t_ts = 0.0

    for win in progress_iter(windows, total=len(windows), desc=f"{slide_id} ROI selection"):
        x0, y0, w0, h0 = int(win["x0_level0"]), int(win["y0_level0"]), int(win["w_level0"]), int(win["h_level0"])
        base_row = {
            "slide_id": slide_id,
            "wsi_path": str(wsi_path),
            "grid_x": int(win["grid_x"]),
            "grid_y": int(win["grid_y"]),
            "x0_level0": x0,
            "y0_level0": y0,
            "w_level0": w0,
            "h_level0": h0,
            "target_mpp_for_model_inference": config.target_mpp,
            "base_mpp_x": base_mpp_x,
            "base_mpp_y": base_mpp_y,
        }

        t0 = timer.now()
        passed, tissue_info = strong_tissue_filter_for_level0_rect(
            tissue_mask, eroded_tissue_mask, thumb_rgb, mask_w, mask_h,
            slide_w0, slide_h0, x0, y0, w0, h0, config,
        )
        t_tissue += timer.now() - t0
        if not passed:
            tissue_rejected_rows.append({**base_row, **tissue_info, "roi_score": np.nan, "final_decision": "discard_tissue_filter"})
            continue

        t0 = timer.now()
        roi_rgb, actual_w0, actual_h0 = read_region_at_target_mpp(
            slide, x0, y0, config.roi_size_target_px, config.roi_size_target_px,
            config.target_mpp, base_mpp_x, base_mpp_y,
        )
        effective_mpp_x, effective_mpp_y = compute_effective_mpp(
            actual_w0, actual_h0, base_mpp_x, base_mpp_y,
            config.roi_size_target_px, config.roi_size_target_px,
        )
        ts_mask = predict_roi_multiclass(ts_model, roi_rgb, config.ts_n_classes, config)
        features = compute_roi_features(ts_mask, tissue_info["filter_global_tissue_fraction"],
                                        (effective_mpp_x + effective_mpp_y) / 2.0, config)
        roi_score = compute_raw_roi_score(features, config)
        t_ts += timer.now() - t0
        raw_rows.append({
            **base_row,
            **tissue_info,
            **features,
            "roi_score": roi_score,
            "effective_mpp_x_for_selection": effective_mpp_x,
            "effective_mpp_y_for_selection": effective_mpp_y,
        })
        del roi_rgb, ts_mask
        clear_memory_soft()

    timer.records["tissue_filtering"] = t_tissue
    timer.records["tumor_stroma_inference_for_roi_selection"] = t_ts

    if tissue_rejected_rows:
        write_csv_and_excel(pd.DataFrame(tissue_rejected_rows), dirs["tables"] / "rejected_by_tissue_filter.csv",
                            dirs["tables"] / "rejected_by_tissue_filter.xlsx")
    if not raw_rows:
        summary = empty_summary(slide_id, wsi_path, "NO_TISSUE_ROIS", "No candidate ROI passed tissue filtering.", timer, dirs)
        write_csv_and_excel(pd.DataFrame([summary]), dirs["tables"] / "final_summary.csv", dirs["tables"] / "final_summary.xlsx")
        save_timing_report(timer, summary, dirs)
        slide.close()
        return summary

    df_raw = pd.DataFrame(raw_rows).sort_values("roi_score", ascending=False).reset_index(drop=True)
    write_csv_and_excel(df_raw, dirs["tables"] / "raw_roi_selection_features.csv", dirs["tables"] / "raw_roi_selection_features.xlsx")

    t = timer.now()
    df_selection = classify_selected_rois(df_raw, config)
    selected_df = df_selection[df_selection["final_decision"].isin(["direct_accept", "neighbor_associated_stroma"])].copy()
    selected_df = selected_df.sort_values("roi_score", ascending=False).reset_index(drop=True)
    selected_df["rank"] = np.arange(1, len(selected_df) + 1)
    timer.mark("roi_selection_and_neighbor_rescue", t)

    write_csv_and_excel(df_selection, dirs["tables"] / "all_roi_selection_decisions.csv", dirs["tables"] / "all_roi_selection_decisions.xlsx")
    write_csv_and_excel(selected_df, dirs["tables"] / "selected_rois_before_tils_scoring.csv", dirs["tables"] / "selected_rois_before_tils_scoring.xlsx")
    print("Selected ROIs:", len(selected_df))

    if len(selected_df) == 0:
        summary = empty_summary(slide_id, wsi_path, "NO_SELECTED_ROIS", "No ROI selected after tumor/stroma ROI selection.", timer, dirs)
        write_csv_and_excel(pd.DataFrame([summary]), dirs["tables"] / "final_summary.csv", dirs["tables"] / "final_summary.xlsx")
        save_timing_report(timer, summary, dirs)
        slide.close()
        return summary

    roi_score_rows = []
    t_score = timer.now()
    for _, row in progress_iter(selected_df.iterrows(), total=len(selected_df), desc=f"{slide_id} ROI scoring"):
        roi_rank = int(row["rank"])
        x0, y0 = int(row["x0_level0"]), int(row["y0_level0"])
        roi_rgb, actual_w0, actual_h0 = read_region_at_target_mpp(
            slide, x0, y0, config.roi_size_target_px, config.roi_size_target_px,
            config.target_mpp, base_mpp_x, base_mpp_y,
        )
        effective_mpp_x, effective_mpp_y = compute_effective_mpp(
            actual_w0, actual_h0, base_mpp_x, base_mpp_y,
            config.roi_size_target_px, config.roi_size_target_px,
        )
        ts_mask = predict_roi_multiclass(ts_model, roi_rgb, config.ts_n_classes, config)
        tils_raw = predict_roi_tils_mask(tils_model, roi_rgb, config)
        tils_mask, watershed_info = apply_watershed_to_tils_mask(tils_raw, config)
        stroma_binary = (ts_mask == 2).astype(np.uint8)
        tils_inside_stroma = ((tils_mask == 1) & (stroma_binary == 1)).astype(np.uint8)
        score_info = compute_area_based_roi_tils_score(stroma_binary, tils_inside_stroma, effective_mpp_x, effective_mpp_y, config)
        clean_tis = score_info.pop("clean_tils_inside_stroma_mask")
        roi_score_row = row.to_dict()
        roi_score_row.update(score_info)
        roi_score_row.update(watershed_info)
        roi_score_row.update({"roi_rank": roi_rank, "actual_w0_for_scoring": actual_w0, "actual_h0_for_scoring": actual_h0})
        roi_score_rows.append(roi_score_row)
        save_selected_roi_outputs(roi_rgb, ts_mask, tils_mask, stroma_binary, tils_inside_stroma, clean_tis, roi_rank, roi_score_row, dirs, config)
        del roi_rgb, ts_mask, tils_raw, tils_mask, stroma_binary, tils_inside_stroma, clean_tis
        clear_memory_soft()
    timer.records["selected_roi_tils_inference_and_area_based_scoring"] = timer.now() - t_score

    df_roi_scores = pd.DataFrame(roi_score_rows)
    write_csv_and_excel(df_roi_scores, dirs["tables"] / "roi_scores.csv", dirs["tables"] / "roi_scores.xlsx")
    save_geojson(df_roi_scores.sort_values("roi_tils_score_fraction", ascending=False), dirs["geojson"] / "all_selected_rois.geojson")

    wsi_scores = aggregate_roi_scores_to_wsi(df_roi_scores, config)
    summary = {
        "slide_id": slide_id,
        "wsi_path": str(wsi_path),
        "status": "OK",
        "num_candidate_windows": len(windows),
        "num_selected_rois": int(len(df_roi_scores)),
        "num_direct_selected_rois": int((df_roi_scores["final_decision"] == "direct_accept").sum()),
        "num_neighbor_selected_rois": int((df_roi_scores["final_decision"] == "neighbor_associated_stroma").sum()),
        "til_diameter_um": config.til_diameter_um,
        "total_selected_stroma_area_px": int(df_roi_scores["stroma_area_px"].sum()),
        "total_tils_inside_stroma_count": int(df_roi_scores["num_tils_inside_stroma"].sum()),
        **wsi_scores,
        "total_pipeline_time_seconds": timer.total(),
        "total_pipeline_time_minutes": timer.total() / 60.0,
        "total_pipeline_time_formatted": Timer.fmt(timer.total()),
        "output_folder": str(dirs["slide_root"]),
    }
    write_csv_and_excel(pd.DataFrame([summary]), dirs["tables"] / "final_summary.csv", dirs["tables"] / "final_summary.xlsx")
    save_timing_report(timer, summary, dirs)
    slide.close()
    print("Finished", slide_id, "median score (%) =", f"{summary['score_median_percent']:.2f}", "time =", summary["total_pipeline_time_formatted"])
    return summary


def run_batch(ts_model, tils_model, config: Config):
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "summary").mkdir(parents=True, exist_ok=True)
    wsi_files = list_wsi_files(config.input_dir, config.wsi_extensions)
    if not wsi_files:
        raise RuntimeError(f"No WSI files found in: {config.input_dir}")
    print("Total WSIs found:", len(wsi_files))

    batch_start = time.perf_counter()
    for i, wsi_path in enumerate(wsi_files, start=1):
        slide_id = safe_slide_id(wsi_path)
        final_summary_path = config.output_dir / "PerSlideResults" / slide_id / "tables" / "final_summary.csv"
        if final_summary_path.exists() and not config.overwrite_existing:
            print(f"Skipping already processed slide {i}/{len(wsi_files)}: {slide_id}")
            old = pd.read_csv(final_summary_path).iloc[0].to_dict()
            append_or_update_summary(old, config.output_dir / "summary" / "summary.csv", config.output_dir / "summary" / "summary.xlsx")
            continue
        try:
            row = process_one_wsi(wsi_path, ts_model, tils_model, config)
            append_or_update_summary(row, config.output_dir / "summary" / "summary.csv", config.output_dir / "summary" / "summary.xlsx")
        except Exception as exc:
            error_text = traceback.format_exc()
            print("ERROR while processing", slide_id)
            print(error_text)
            fail_row = {
                "slide_id": slide_id,
                "wsi_path": str(wsi_path),
                "status": "FAILED",
                "error_message": str(exc),
                "traceback": error_text,
                "score_mean_percent": 0.0,
                "score_stroma_weighted_mean_percent": 0.0,
                "score_median_percent": 0.0,
            }
            slide_dir = config.output_dir / "PerSlideResults" / slide_id / "tables"
            slide_dir.mkdir(parents=True, exist_ok=True)
            with open(slide_dir / "error_log.txt", "w", encoding="utf-8") as f:
                f.write(error_text)
            append_or_update_summary(fail_row, config.output_dir / "summary" / "summary.csv", config.output_dir / "summary" / "summary.xlsx")
            append_or_update_summary(fail_row, config.output_dir / "summary" / "failed_wsi_processing.csv", config.output_dir / "summary" / "failed_wsi_processing.xlsx")

    batch_time = time.perf_counter() - batch_start
    with open(config.output_dir / "summary" / "batch_processing_report.txt", "w", encoding="utf-8") as f:
        f.write("KD WSI stromal TILs scoring batch report\n")
        f.write("=" * 80 + "\n")
        f.write(f"Input directory: {config.input_dir}\n")
        f.write(f"Output directory: {config.output_dir}\n")
        f.write(f"Total WSIs found: {len(wsi_files)}\n")
        f.write(f"Batch time seconds: {batch_time:.2f}\n")
        f.write(f"Batch time formatted: {Timer.fmt(batch_time)}\n")
