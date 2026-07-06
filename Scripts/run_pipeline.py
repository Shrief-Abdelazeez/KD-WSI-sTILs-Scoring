"""Command-line entry point for KD-only WSI-level stromal TILs scoring."""

from pathlib import Path
import argparse
import numpy as np

from utils import setup_tensorflow_environment
setup_tensorflow_environment()

import tensorflow as tf

from config import Config
from utils import setup_gpu
from model_utils import load_kd_models
from pipeline import run_batch


def parse_args():
    parser = argparse.ArgumentParser(description="KD-only WSI-level stromal TILs scoring")
    parser.add_argument("--input_dir", type=Path, default=None, help="Folder containing input WSIs")
    parser.add_argument("--model_dir", type=Path, default=None, help="Folder containing KD model weights")
    parser.add_argument("--output_dir", type=Path, default=None, help="Folder where outputs will be saved")
    parser.add_argument("--tumor_stroma_weights", type=Path, default=None, help="Tumor/stroma KD FastSCNN weights")
    parser.add_argument("--tils_weights", type=Path, default=None, help="TILs KD FastSCNN weights")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess slides even if final_summary.csv exists")
    parser.add_argument("--batch_size", type=int, default=None, help="Inference batch size")
    parser.add_argument("--max_windows", type=int, default=None, help="Debug option: process only the first N ROI windows")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()
    if args.input_dir is not None:
        config.input_dir = args.input_dir
    if args.model_dir is not None:
        config.model_dir = args.model_dir
        config.tumor_stroma_weights = args.model_dir / "tumor_stroma_fastscnn_kd.weights.h5"
        config.tils_weights = args.model_dir / "tils_fastscnn_kd.weights.h5"
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.tumor_stroma_weights is not None:
        config.tumor_stroma_weights = args.tumor_stroma_weights
    if args.tils_weights is not None:
        config.tils_weights = args.tils_weights
    if args.overwrite:
        config.overwrite_existing = True
    if args.batch_size is not None:
        config.inference_batch_size = args.batch_size
    if args.max_windows is not None:
        config.max_windows_to_evaluate = args.max_windows

    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)
    setup_gpu(tf)
    ts_model, tils_model = load_kd_models(config)
    run_batch(ts_model, tils_model, config)


if __name__ == "__main__":
    main()
