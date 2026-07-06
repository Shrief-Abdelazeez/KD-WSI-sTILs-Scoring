"""General utilities for the KD WSI scoring pipeline."""

from pathlib import Path
import gc
import os
import time
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def setup_tensorflow_environment() -> None:
    """Set TensorFlow memory behavior before TensorFlow is imported."""
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")


class Timer:
    """Simple stage-wise timer."""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.records = {}

    def now(self) -> float:
        return time.perf_counter()

    def mark(self, name: str, start_time: float) -> None:
        self.records[name] = time.perf_counter() - start_time

    def total(self) -> float:
        return time.perf_counter() - self.t0

    @staticmethod
    def fmt(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.2f} sec"
        if seconds < 3600:
            return f"{seconds / 60:.2f} min"
        return f"{seconds / 3600:.2f} hr"


def setup_gpu(tf) -> None:
    """Enable TensorFlow GPU memory growth when GPUs are available."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU available: {len(gpus)}")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
    else:
        print("No GPU found. Running on CPU.")


def clear_memory_soft() -> None:
    gc.collect()


def progress_iter(iterable: Iterable, total: Optional[int] = None, desc: str = "Processing"):
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc)
    return iterable


def resolve_path(path: Path, description: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find {description}: {path}")
    return path


def list_wsi_files(input_dir: Path, extensions) -> list:
    input_dir = Path(input_dir)
    files = []
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in extensions:
            files.append(p)
    return files


def safe_slide_id(wsi_path: Path) -> str:
    """Return a folder-safe slide identifier from a WSI path."""
    slide_id = Path(wsi_path).stem
    invalid = '<>:"/\\|?*'
    for ch in invalid:
        slide_id = slide_id.replace(ch, "_")
    return slide_id or "unnamed_slide"


def write_csv_and_excel(df: pd.DataFrame, csv_path: Path, xlsx_path: Path) -> None:
    csv_path = Path(csv_path)
    xlsx_path = Path(xlsx_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)


def append_or_update_summary(row: dict, summary_csv: Path, summary_xlsx: Path, key: str = "slide_id") -> None:
    summary_csv = Path(summary_csv)
    summary_xlsx = Path(summary_xlsx)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([row])
    if summary_csv.exists() and summary_csv.stat().st_size > 0:
        old_df = pd.read_csv(summary_csv)
        if key in old_df.columns and key in row_df.columns:
            old_df = old_df[old_df[key].astype(str) != str(row_df.iloc[0][key])]
        new_df = pd.concat([old_df, row_df], ignore_index=True, sort=False)
    else:
        new_df = row_df
    new_df.to_csv(summary_csv, index=False)
    new_df.to_excel(summary_xlsx, index=False)
