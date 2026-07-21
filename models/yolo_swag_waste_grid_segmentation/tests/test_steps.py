"""Tests that explain split, training, export, and grid-cell prediction."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

_PRETRAINED_URL = (
    "https://github.com/GIScience/Open-access_Model_for_Solid_Waste_Detection_on_Crowdsourced_UAV_Imagery_in_Sub-Saharan_Africa/"
    "raw/refs/heads/main/data/models/train/weights/best.pt"
)


@pytest.fixture(scope="session")
def pretrained_weights(tmp_path_factory: pytest.TempPathFactory) -> str:
    from upath import UPath

    cache = tmp_path_factory.mktemp("yolo11xcls_weights") / "best.pt"
    cache.write_bytes(UPath(_PRETRAINED_URL).read_bytes())
    return str(cache)


def _train_toy_model(
    toy_chips: Path,
    toy_labels: Path,
    pretrained_weights: str,
    hyperparameters: dict[str, Any],
) -> tuple[dict[str, Any], bytes]:
    """Run the common split-and-train setup needed by model-step tests."""
    from models.yolo_swag_waste_grid_segmentation.pipeline import CLASS_NAMES, split_dataset, train_model

    split_info = split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=hyperparameters,
    )
    model_bytes = train_model.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        base_model_weights=pretrained_weights,
        hyperparameters=hyperparameters,
        split_info=split_info,
        num_classes=len(CLASS_NAMES),
    )
    return split_info, model_bytes


def test_split_dataset_writes_disjoint_class_folders(
    toy_chips: Path,
    toy_labels: Path,
    base_hyperparameters: dict[str, Any],
) -> None:
    from models.yolo_swag_waste_grid_segmentation.pipeline import CLASS_NAMES, split_dataset

    info = split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=base_hyperparameters,
    )

    assert info["strategy"] == "grid_5m_stratified_random"
    yolo_dir = Path(info["_yolo_dir"])
    seen_train_cells: set[str] = set()
    seen_val_cells: set[str] = set()
    for split in ("train", "val"):
        for cls in CLASS_NAMES:
            images = {image.stem for image in (yolo_dir / split / cls).glob("cell_*.png")}
            assert images
            expected_count = info[f"{split}_counts_per_class"][cls]
            assert len(images) == expected_count
            if split == "train":
                seen_train_cells.update(images)
            else:
                seen_val_cells.update(images)

    assert seen_train_cells.isdisjoint(seen_val_cells)
    assert len(seen_train_cells) == info["train_count"]
    assert len(seen_val_cells) == info["val_count"]


def test_step_rebuilds_dataset_when_split_container_is_gone(
    toy_chips: Path,
    toy_labels: Path,
    base_hyperparameters: dict[str, Any],
) -> None:
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

    rebuilt = _resolve_yolo_dir_for_step(
        str(toy_chips),
        str(toy_labels),
        base_hyperparameters,
        split_info,
    )

    for split in ("train", "val"):
        for cls in CLASS_NAMES:
            assert list((rebuilt / split / cls).glob("cell_*.png"))


def test_pretrained_checkpoint_loads_as_the_expected_classifier(pretrained_weights: str) -> None:
    from ultralytics import YOLO

    from models.yolo_swag_waste_grid_segmentation.pipeline import CLASS_NAMES

    model = YOLO(pretrained_weights, task="classify")

    assert model.task == "classify"
    assert [model.names[index] for index in range(len(CLASS_NAMES))] == list(CLASS_NAMES)


def test_train_model_returns_a_reloadable_classifier_checkpoint(
    toy_chips: Path,
    toy_labels: Path,
    base_hyperparameters: dict[str, Any],
    pretrained_weights: str,
) -> None:
    from models.yolo_swag_waste_grid_segmentation.pipeline import (
        CLASS_NAMES,
        _restore_checkpoint,
    )

    _, model_bytes = _train_toy_model(
        toy_chips,
        toy_labels,
        pretrained_weights,
        base_hyperparameters,
    )

    assert isinstance(model_bytes, bytes)
    assert len(model_bytes) > 0
    restored_model = _restore_checkpoint(model_bytes)
    assert restored_model.task == "classify"
    assert [restored_model.names[index] for index in range(len(CLASS_NAMES))] == list(CLASS_NAMES)


def test_evaluate_model_reports_accuracy(
    toy_chips: Path,
    toy_labels: Path,
    base_hyperparameters: dict[str, Any],
    pretrained_weights: str,
) -> None:
    from models.yolo_swag_waste_grid_segmentation.pipeline import evaluate_model

    split_info, model_bytes = _train_toy_model(
        toy_chips,
        toy_labels,
        pretrained_weights,
        base_hyperparameters,
    )
    metrics = evaluate_model.entrypoint(
        trained_model=model_bytes,
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=base_hyperparameters,
        split_info=split_info,
    )

    assert set(metrics) == {"accuracy"}
    assert 0.0 <= metrics["accuracy"] <= 1.0


@pytest.mark.slow
def test_deterministic_toy_data_converges(
    toy_chips: Path,
    toy_labels: Path,
    base_hyperparameters: dict[str, Any],
    pretrained_weights: str,
) -> None:
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

    assert metrics["accuracy"] >= 0.9


def test_export_onnx_runs_with_onnxruntime(
    toy_chips: Path,
    toy_labels: Path,
    base_hyperparameters: dict[str, Any],
    pretrained_weights: str,
) -> None:
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

    assert isinstance(onnx_bytes, bytes)
    loaded = onnx.load_from_string(onnx_bytes)
    assert len(loaded.graph.input) == 1
    assert len(loaded.graph.output) == 1

    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    model_input = np.zeros((1, 3, 128, 128), dtype=np.float32)
    output = session.run(None, {session.get_inputs()[0].name: model_input})[0]
    assert isinstance(output, np.ndarray)
    assert output.shape == (1, 2)
    assert np.isfinite(output).all()


class _StubSession:
    """Mimics the onnxruntime InferenceSession surface predict() relies on."""

    def __init__(self, probs: list[float]) -> None:
        self._probs = np.asarray([probs], dtype=np.float32)
        self.batches: list[np.ndarray] = []

    def get_inputs(self) -> list[Any]:
        return [SimpleNamespace(name="input")]

    def run(self, _output_names: Any, _feeds: dict[str, Any]) -> list[Any]:
        self.batches.append(_feeds["input"])
        return [self._probs]


def _probabilities_for_waste_confidence(waste_confidence: float) -> list[float]:
    from models.yolo_swag_waste_grid_segmentation.pipeline import CLASS_NAMES

    waste_index = CLASS_NAMES.index("waste")
    probabilities = [waste_confidence, waste_confidence]
    probabilities[waste_index] = waste_confidence
    probabilities[1 - waste_index] = 1.0 - waste_confidence
    return probabilities


def test_predict_returns_unique_wgs84_polygons(toy_chips: Path) -> None:
    from models.yolo_swag_waste_grid_segmentation.pipeline import predict

    session = _StubSession(_probabilities_for_waste_confidence(0.9))

    fc = predict(session, str(toy_chips), {"confidence_threshold": 0.5})

    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) > 0
    cell_ids = [f["properties"]["cell_id"] for f in fc["features"]]
    assert len(cell_ids) == len(set(cell_ids))
    feature = fc["features"][0]
    ring = feature["geometry"]["coordinates"][0]
    assert feature["geometry"]["type"] == "Polygon"
    assert len(ring) == 5
    assert ring[0] == ring[-1]
    assert session.batches[0].shape == (1, 3, 128, 128)
    assert session.batches[0].dtype == np.float32


@pytest.mark.parametrize(
    ("waste_confidence", "expected_label"),
    [(0.1, "background"), (0.5, "waste"), (0.9, "waste")],
)
def test_predict_applies_the_waste_confidence_threshold(
    toy_chips: Path,
    waste_confidence: float,
    expected_label: str,
) -> None:
    from models.yolo_swag_waste_grid_segmentation.pipeline import predict

    session = _StubSession(_probabilities_for_waste_confidence(waste_confidence))

    fc = predict(session, str(toy_chips), {"confidence_threshold": 0.5})

    assert {feature["properties"]["label"] for feature in fc["features"]} == {expected_label}


def test_predict_requires_a_confidence_threshold(toy_chips: Path) -> None:
    from models.yolo_swag_waste_grid_segmentation.pipeline import predict

    with pytest.raises(ValueError, match="confidence_threshold"):
        predict(_StubSession(_probabilities_for_waste_confidence(0.9)), str(toy_chips), {})
