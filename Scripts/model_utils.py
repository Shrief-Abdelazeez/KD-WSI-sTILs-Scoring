"""Model loading utilities.

The repository expects the FastSCNN architecture file used during training to be
available as ``Scripts/Fast_SCNN_Model.py`` with a function named
``build_fast_scnn``. This keeps the inference architecture identical to training.
"""

from pathlib import Path


def build_fastscnn_model(num_classes: int, patch_size: int):
    try:
        from Fast_SCNN_Model import build_fast_scnn
    except ImportError as exc:
        raise ImportError(
            "Fast_SCNN_Model.py was not found. Copy the training architecture file "
            "containing build_fast_scnn() into the Scripts/ folder."
        ) from exc

    return build_fast_scnn(
        input_shape=(patch_size, patch_size, 3),
        num_classes=num_classes,
        down_channels=(32, 48, 64),
        block_channels=(64, 96, 128),
        bottleneck_counts=(2, 2, 2),
        expansion=4,
        ppm_channels=64,
        fusion_channels=64,
        classifier_channels=64,
    )


def load_kd_models(config):
    """Load KD tumor/stroma and KD TILs FastSCNN models."""
    ts_weights = Path(config.tumor_stroma_weights)
    tils_weights = Path(config.tils_weights)

    if not ts_weights.exists():
        raise FileNotFoundError(
            f"Tumor/stroma KD weights not found: {ts_weights}\n"
            "Place the file in Models/ or pass --tumor_stroma_weights."
        )
    if not tils_weights.exists():
        raise FileNotFoundError(
            f"TILs KD weights not found: {tils_weights}\n"
            "Place the file in Models/ or pass --tils_weights."
        )

    print("Loading KD tumor/stroma FastSCNN...")
    print("Weights:", ts_weights)
    ts_model = build_fastscnn_model(config.ts_n_classes, config.patch_size)
    ts_model.load_weights(str(ts_weights))

    print("Loading KD TILs FastSCNN...")
    print("Weights:", tils_weights)
    tils_model = build_fastscnn_model(config.tils_n_classes, config.patch_size)
    tils_model.load_weights(str(tils_weights))

    return ts_model, tils_model
