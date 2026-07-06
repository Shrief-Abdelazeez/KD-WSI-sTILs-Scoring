"""WSI reading and ROI grid generation utilities."""

import math
import numpy as np
import cv2

try:
    import openslide
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Please install openslide-python first. On Windows, OpenSlide binaries "
        "must also be available in PATH."
    ) from exc


def get_base_mpp(slide, fallback_base_mpp: float):
    mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X, None)
    mpp_y = slide.properties.get(openslide.PROPERTY_NAME_MPP_Y, None)
    if mpp_x is None or mpp_y is None:
        print(f"Warning: WSI MPP missing. Using fallback MPP = {fallback_base_mpp}")
        return float(fallback_base_mpp), float(fallback_base_mpp)
    return float(mpp_x), float(mpp_y)


def choose_best_level_for_mpp(slide, base_mpp_x: float, target_mpp: float) -> int:
    target_downsample = target_mpp / base_mpp_x
    downsamples = np.array(slide.level_downsamples, dtype=np.float32)
    return int(np.argmin(np.abs(downsamples - target_downsample)))


def read_region_at_target_mpp(slide, x0_level0, y0_level0, out_w_px, out_h_px, target_mpp, base_mpp_x, base_mpp_y):
    region_w0 = int(round(out_w_px * target_mpp / base_mpp_x))
    region_h0 = int(round(out_h_px * target_mpp / base_mpp_y))
    level = choose_best_level_for_mpp(slide, base_mpp_x, target_mpp)
    level_down = float(slide.level_downsamples[level])
    read_w = int(math.ceil(region_w0 / level_down))
    read_h = int(math.ceil(region_h0 / level_down))
    rgba = slide.read_region((int(x0_level0), int(y0_level0)), level, (read_w, read_h))
    rgb = np.array(rgba.convert("RGB"))
    if rgb.shape[1] != out_w_px or rgb.shape[0] != out_h_px:
        rgb = cv2.resize(rgb, (out_w_px, out_h_px), interpolation=cv2.INTER_LINEAR)
    return rgb, region_w0, region_h0


def compute_effective_mpp(actual_w0, actual_h0, base_mpp_x, base_mpp_y, out_w_px, out_h_px):
    effective_mpp_x = (float(actual_w0) * float(base_mpp_x)) / float(out_w_px)
    effective_mpp_y = (float(actual_h0) * float(base_mpp_y)) / float(out_h_px)
    return float(effective_mpp_x), float(effective_mpp_y)


def generate_candidate_windows(slide, base_mpp_x, base_mpp_y, config):
    slide_w0, slide_h0 = slide.dimensions
    roi_w0 = int(round(config.roi_size_target_px * config.target_mpp / base_mpp_x))
    roi_h0 = int(round(config.roi_size_target_px * config.target_mpp / base_mpp_y))
    stride_x0 = int(round(config.roi_stride_target_px * config.target_mpp / base_mpp_x))
    stride_y0 = int(round(config.roi_stride_target_px * config.target_mpp / base_mpp_y))

    x_positions = list(range(0, max(1, slide_w0 - roi_w0 + 1), stride_x0)) or [0]
    y_positions = list(range(0, max(1, slide_h0 - roi_h0 + 1), stride_y0)) or [0]

    last_x = max(0, slide_w0 - roi_w0)
    last_y = max(0, slide_h0 - roi_h0)
    if x_positions[-1] != last_x:
        x_positions.append(last_x)
    if y_positions[-1] != last_y:
        y_positions.append(last_y)

    for grid_y, y in enumerate(y_positions):
        for grid_x, x in enumerate(x_positions):
            yield {
                "grid_x": grid_x,
                "grid_y": grid_y,
                "x0_level0": int(x),
                "y0_level0": int(y),
                "w_level0": int(roi_w0),
                "h_level0": int(roi_h0),
            }
