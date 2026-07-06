"""Patch-wise model inference utilities."""

import numpy as np
import tensorflow as tf

from utils import clear_memory_soft


def preprocess_model_input(img_rgb):
    return img_rgb.astype(np.float32) / 255.0


def run_model_inference(model, x, use_direct_model_call=True, batch_size=None):
    if use_direct_model_call:
        y = model(tf.convert_to_tensor(x, dtype=tf.float32), training=False)
        if isinstance(y, (list, tuple)):
            y = y[0]
        return y.numpy()
    return model.predict(x, batch_size=batch_size or len(x), verbose=0)


def compute_start_positions_covering_boundary(length, patch_size, stride):
    if length <= patch_size:
        return [0]
    positions = list(range(0, length - patch_size + 1, stride))
    last_start = length - patch_size
    if positions[-1] != last_start:
        positions.append(last_start)
    return positions


def generate_padded_tiles(image_rgb, patch_size=256, stride=192):
    h, w = image_rgb.shape[:2]
    y_positions = compute_start_positions_covering_boundary(h, patch_size, stride)
    x_positions = compute_start_positions_covering_boundary(w, patch_size, stride)
    for y in y_positions:
        for x in x_positions:
            y_end = min(y + patch_size, h)
            x_end = min(x + patch_size, w)
            crop = image_rgb[y:y_end, x:x_end]
            valid_h = y_end - y
            valid_w = x_end - x
            padded = np.zeros((patch_size, patch_size, 3), dtype=image_rgb.dtype)
            padded[:valid_h, :valid_w] = crop
            yield padded, y, x, valid_h, valid_w


def predict_roi_multiclass(model, roi_rgb, num_classes, config):
    h, w = roi_rgb.shape[:2]
    full_probs = np.zeros((h, w, num_classes), dtype=np.float16)
    batch_tiles, batch_infos = [], []

    def flush_batch():
        nonlocal batch_tiles, batch_infos, full_probs
        if not batch_tiles:
            return
        x = np.stack([preprocess_model_input(t) for t in batch_tiles], axis=0).astype(np.float32)
        probs = run_model_inference(model, x, config.use_direct_model_call, batch_size=len(batch_tiles))
        for prob_tile, info in zip(probs, batch_infos):
            y, x0, valid_h, valid_w = info
            full_probs[y:y + valid_h, x0:x0 + valid_w, :] = np.maximum(
                full_probs[y:y + valid_h, x0:x0 + valid_w, :],
                prob_tile[:valid_h, :valid_w, :].astype(np.float16),
            )
        batch_tiles, batch_infos = [], []
        clear_memory_soft()

    for tile, y, x0, valid_h, valid_w in generate_padded_tiles(roi_rgb, config.patch_size, config.patch_stride):
        batch_tiles.append(tile)
        batch_infos.append((y, x0, valid_h, valid_w))
        if len(batch_tiles) >= config.inference_batch_size:
            flush_batch()
    flush_batch()
    return np.argmax(full_probs, axis=-1).astype(np.uint8)


def extract_tils_foreground_prob(model_output):
    out = np.asarray(model_output)
    if out.ndim != 4:
        raise ValueError(f"Unexpected TILs model output shape: {out.shape}")
    if out.shape[-1] == 1:
        return out[..., 0]
    if out.shape[-1] >= 2:
        return out[..., 1]
    raise ValueError(f"Unexpected TILs model output channels: {out.shape}")


def predict_roi_tils_mask(model, roi_rgb, config):
    h, w = roi_rgb.shape[:2]
    full_fg_prob = np.zeros((h, w), dtype=np.float16)
    batch_tiles, batch_infos = [], []

    def flush_batch():
        nonlocal batch_tiles, batch_infos, full_fg_prob
        if not batch_tiles:
            return
        x = np.stack([preprocess_model_input(t) for t in batch_tiles], axis=0).astype(np.float32)
        output = run_model_inference(model, x, config.use_direct_model_call, batch_size=len(batch_tiles))
        fg_probs = extract_tils_foreground_prob(output)
        for prob_tile, info in zip(fg_probs, batch_infos):
            y, x0, valid_h, valid_w = info
            full_fg_prob[y:y + valid_h, x0:x0 + valid_w] = np.maximum(
                full_fg_prob[y:y + valid_h, x0:x0 + valid_w],
                prob_tile[:valid_h, :valid_w].astype(np.float16),
            )
        batch_tiles, batch_infos = [], []
        clear_memory_soft()

    for tile, y, x0, valid_h, valid_w in generate_padded_tiles(roi_rgb, config.patch_size, config.patch_stride):
        batch_tiles.append(tile)
        batch_infos.append((y, x0, valid_h, valid_w))
        if len(batch_tiles) >= config.inference_batch_size:
            flush_batch()
    flush_batch()
    return (full_fg_prob >= config.tils_threshold).astype(np.uint8)
