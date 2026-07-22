"""High-level tests for the four waste-grid pipeline stages.

The toy data is deliberately easy: bright grid cells are waste and dark cells
are background.  Together, these tests describe the normal pipeline flow:
split data, train a classifier, evaluate it, then export and use it.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np


def _train_toy_model(
    toy_chips: Path,
    toy_labels: Path,
    pretrained_weights: str,
    hyperparameters: dict[str, Any],
) -> tuple[dict[str, Any], bytes]:
    """Run the shared split-and-train setup for a pipeline-stage test."""
    from models.yolo_swag_waste_grid_segmentation.pipeline import CLASS_NAMES, split_dataset, train_model

    split_info = split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=hyperparameters,
    )

    with patch("models.yolo_swag_waste_grid_segmentation.pipeline._log_yolo_loss_history"):
        model_bytes = train_model.entrypoint(
            dataset_chips=str(toy_chips),
            dataset_labels=str(toy_labels),
            base_model_weights=pretrained_weights,
            hyperparameters=hyperparameters,
            split_info=split_info,
            num_classes=len(CLASS_NAMES),
        )
    return split_info, model_bytes


def test_split_dataset(
    toy_chips: Path,
    toy_labels: Path,
    base_hyperparameters: dict[str, Any],
) -> None:
    """Split the labelled grid cells into train and validation class folders."""
    from models.yolo_swag_waste_grid_segmentation.pipeline import CLASS_NAMES, split_dataset

    info = split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=base_hyperparameters,
    )

    assert info["strategy"] == "grid_5m_stratified_random"
    assert info["train_count"] > 0
    assert info["val_count"] > 0
    yolo_dir = Path(info["_yolo_dir"])
    for split in ("train", "val"):
        for class_name in CLASS_NAMES:
            assert list((yolo_dir / split / class_name).glob("cell_*.png"))
    train_cells = {image.stem for image in (yolo_dir / "train").rglob("cell_*.png")}
    val_cells = {image.stem for image in (yolo_dir / "val").rglob("cell_*.png")}
    assert train_cells.isdisjoint(val_cells)


def test_training_rebuilds_missing_split_dataset(
    toy_chips: Path,
    toy_labels: Path,
    base_hyperparameters: dict[str, Any],
) -> None:
    """Training can reconstruct its files when a prior step's workspace is absent."""
    from models.yolo_swag_waste_grid_segmentation.pipeline import (
        CLASS_NAMES,
        _resolve_yolo_dir_for_step,
        split_dataset,
    )

    split_info = split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=base_hyperparameters,
    )
    shutil.rmtree(split_info["_yolo_dir"])

    yolo_dir = _resolve_yolo_dir_for_step(
        str(toy_chips),
        str(toy_labels),
        base_hyperparameters,
        split_info,
    )

    for split in ("train", "val"):
        for class_name in CLASS_NAMES:
            assert list((yolo_dir / split / class_name).glob("cell_*.png"))


def test_train_model(
    toy_chips: Path,
    toy_labels: Path,
    base_hyperparameters: dict[str, Any],
    pretrained_weights: str,
) -> None:
    """Train from the public checkpoint and return a reloadable classifier."""
    from models.yolo_swag_waste_grid_segmentation.pipeline import CLASS_NAMES, _restore_checkpoint

    _, model_bytes = _train_toy_model(
        toy_chips,
        toy_labels,
        pretrained_weights,
        base_hyperparameters,
    )

    assert model_bytes
    model = _restore_checkpoint(model_bytes)
    assert model.task == "classify"
    assert [model.names[index] for index in range(len(CLASS_NAMES))] == list(CLASS_NAMES)


def test_evaluate_model(
    toy_chips: Path,
    toy_labels: Path,
    base_hyperparameters: dict[str, Any],
    pretrained_weights: str,
) -> None:
    """The simple, balanced toy grid should converge to high validation accuracy."""
    from models.yolo_swag_waste_grid_segmentation.pipeline import evaluate_model

    hyperparameters = {**base_hyperparameters, "epochs": 5}
    split_info, model_bytes = _train_toy_model(
        toy_chips,
        toy_labels,
        pretrained_weights,
        hyperparameters,
    )
    metrics = evaluate_model.entrypoint(
        trained_model=model_bytes,
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=hyperparameters,
        split_info=split_info,
    )

    assert set(metrics) == {"accuracy"}
    assert metrics["accuracy"] >= 0.9


def test_export_onnx(
    toy_chips: Path,
    toy_labels: Path,
    base_hyperparameters: dict[str, Any],
    pretrained_weights: str,
) -> None:
    """Export the trained classifier and use its ONNX session for grid predictions."""
    import onnx
    import onnxruntime as ort

    from models.yolo_swag_waste_grid_segmentation.pipeline import export_onnx

    _, model_bytes = _train_toy_model(
        toy_chips,
        toy_labels,
        pretrained_weights,
        base_hyperparameters,
    )
    onnx_bytes = export_onnx.entrypoint(trained_model=model_bytes)

    assert onnx_bytes
    model = onnx.load_from_string(onnx_bytes)
    assert len(model.graph.input) == 1
    assert len(model.graph.output) == 1

    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    output = session.run(
        None,
        {session.get_inputs()[0].name: np.zeros((1, 3, 128, 128), dtype=np.float32)},
    )[0]
    assert isinstance(output, np.ndarray)
    assert output.shape == (1, 2)
    assert np.isfinite(output).all()
