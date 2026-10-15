"""Step tests for dinov3s-road-surface-classification.

Runs the real @step entrypoints against toy OAM chips + labelled road lines (tests/conftest.py).
Telemetry sinks (zenml/mlflow) are no-ops via models/conftest.py.
"""

from pathlib import Path
from typing import Any

import pytest

from models.dinov3s_road_surface_classification.tests.conftest import ROAD_LABELS


def test_split_dataset(toy_chips: Path, toy_labels: Path, base_hyperparameters: dict) -> None:
    from models.dinov3s_road_surface_classification.pipeline import split_dataset

    def split(**overrides: Any) -> dict:
        return split_dataset.entrypoint(
            dataset_chips=str(toy_chips),
            dataset_labels=str(toy_labels),
            hyperparameters={**base_hyperparameters, **overrides},
        )

    # 4 equal clusters of the 8 parallel toy roads: two neighbouring roads each, one cluster each for test and val.
    info = split(n_clusters=4)
    assert info["strategy"] == "spatial"
    assert info["n_clusters"] == 4
    assert (info["train_count"], info["val_count"], info["test_count"]) == (4, 2, 2)
    train, val, test = (set(info[f"{name}_ids"]) for name in ("train", "val", "test"))
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    assert split(n_clusters=4)["test_ids"] == info["test_ids"]

    # More clusters than ids: every id is a cluster of its own, test and val get 20% of them.
    single = split(n_clusters=100)
    assert single["n_clusters"] == len(ROAD_LABELS)
    assert single["val_count"] == round(0.2 * len(ROAD_LABELS))
    assert single["test_count"] == round(0.2 * len(ROAD_LABELS))


def test_split_dataset_custom_id_column(
    toy_chips: Path, toy_labels: Path, tmp_path: Path, base_hyperparameters: dict
) -> None:
    """Labels without osm_id name their own id column; roads sharing an id stay on one side of the split."""
    import geopandas as gpd

    from models.dinov3s_road_surface_classification.pipeline import split_dataset

    labels = gpd.read_file(toy_labels / "labels.geojson").drop(columns="osm_id")
    labels["way"] = [i // 2 for i in range(len(labels))]  # pairs of neighbouring roads share an id
    (tmp_path / "labels").mkdir()
    labels.to_file(tmp_path / "labels" / "labels.geojson", driver="GeoJSON")

    info = split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(tmp_path / "labels"),
        hyperparameters={**base_hyperparameters, "id_column": "way", "n_clusters": 4},
    )
    assert info["id_column"] == "way"
    ids = info["train_ids"] + info["val_ids"] + info["test_ids"]
    assert sorted(ids) == sorted(set(labels.dropna(subset=["label"])["way"]))
    # Every labelled road is counted exactly once, in the split of its id.
    counted = sum(sum(info[f"{name}_counts_per_class"].values()) for name in ("train", "val", "test"))
    assert counted == len(ROAD_LABELS)


def test_materialize_and_load(toy_chips: Path, toy_labels: Path, tmp_path: Path) -> None:
    from road_surface_classifier.data.dataset import PatchDataset

    from models.dinov3s_road_surface_classification.pipeline import (
        CLASS_NAMES,
        PATCH_CONFIG_KWARGS,
        materialize,
        road_subset,
    )

    dataset_dir = materialize(str(toy_chips), str(toy_labels), tmp_path)
    dataset = PatchDataset(dataset_dir, {i: i for i in range(len(CLASS_NAMES))}, seed=0, mode="eval")
    # Only the labelled roads are cut, not the unlabelled cross street.
    assert dataset.roads["road_id"].tolist() == list(range(len(ROAD_LABELS)))
    assert dataset.labels == ROAD_LABELS

    images, label = dataset[0]
    size = PATCH_CONFIG_KWARGS["patch_size_px"]
    assert images.shape[1:] == (3, size, size)
    assert 1 <= images.shape[0] <= PATCH_CONFIG_KWARGS["max_patches_per_road"]
    assert images.min() >= 0.0
    assert images.max() <= 1.0
    assert label.item() == ROAD_LABELS[0]

    subset = road_subset(dataset, [1, 3])
    assert len(subset) == 2
    assert [subset[i][1].item() for i in range(2)] == [1, 1]


def test_materialize_inference_config(toy_chips: Path, toy_labels: Path, tmp_path: Path) -> None:
    import pandas as pd
    from road_surface_classifier.data.materialize import PATCHES_FILENAME

    from models.dinov3s_road_surface_classification.pipeline import (
        INFERENCE_PATCH_CONFIG_KWARGS,
        SCENE_ID,
        materialize,
    )

    dataset_dir = materialize(str(toy_chips), str(toy_labels), tmp_path, INFERENCE_PATCH_CONFIG_KWARGS)
    patches = pd.read_parquet(dataset_dir / SCENE_ID / PATCHES_FILENAME, columns=["road_id", "bag"])
    # A single bag per road, within the patch budget.
    assert set(patches["bag"]) == {0}
    assert patches.groupby("road_id").size().max() <= INFERENCE_PATCH_CONFIG_KWARGS["max_patches_per_road"]


def test_checkpoints_match_replay_bank(example_pair: dict) -> None:
    import torch
    from road_surface_classifier.model import load_model

    from models.dinov3s_road_surface_classification.pipeline import replay_bags

    weights = example_pair["weights"]
    model = load_model(weights / "head.pt", weights / "encoder.pt").eval()
    features, labels = replay_bags(torch.load(weights / "replay_bank_2000_features.pt"))
    with torch.no_grad():
        logits = model.classifier(torch.stack([model.pool(bag)[0] for bag in features]))
    accuracy = (logits.argmax(1) == labels).float().mean().item()
    assert accuracy >= 0.85


@pytest.fixture()
def example_inputs(example_pair: dict, base_hyperparameters: dict, tmp_path: Path) -> dict:
    """Step inputs for the example pair, with the real weights (encoder zipped like the released dinov3_encoder.zip)
    and a 5-step fine-tune."""
    import zipfile

    from models.dinov3s_road_surface_classification.pipeline import DINOV3_LICENSE, split_dataset

    weights = example_pair["weights"]
    encoder_zip = tmp_path / "dinov3_encoder.zip"
    with zipfile.ZipFile(encoder_zip, "w") as archive:
        archive.write(weights / "encoder.pt", "encoder.pt")
        archive.write(DINOV3_LICENSE, "LICENSE-DINOv3.md")
    hyperparameters = {
        **base_hyperparameters,
        "encoder_uri": str(encoder_zip),
        "replay_bank_uri": str(weights / "replay_bank_2000_features.pt"),
        "steps": 5,
        "val_every": 2,
    }
    data = {"dataset_chips": str(example_pair["chips"]), "dataset_labels": str(example_pair["labels"])}
    return {
        **data,
        "hyperparameters": hyperparameters,
        "split_info": split_dataset.entrypoint(**data, hyperparameters=hyperparameters),
    }


def test_train_model(example_pair: dict, example_inputs: dict) -> None:
    import torch
    from road_surface_classifier.model import load_model

    from models.dinov3s_road_surface_classification.pipeline import train_model

    weights = example_pair["weights"]
    hyperparameters = example_inputs["hyperparameters"]
    inputs = {**example_inputs, "base_model_weights": str(weights / "head.pt")}
    base = load_model(weights / "head.pt", weights / "encoder.pt").state_dict()

    # Gate passes: the base model comes back unchanged.
    skipped = train_model.entrypoint(**{**inputs, "hyperparameters": {**hyperparameters, "gate_accuracy": 0.0}})
    assert all(torch.equal(value, skipped.state_dict()[key]) for key, value in base.items())

    # Gate forced to fail: the full model comes back, with only the classifier changed.
    tuned = train_model.entrypoint(**{**inputs, "hyperparameters": {**hyperparameters, "gate_accuracy": 1.01}})
    assert tuned.hparams.trainable == "classifier"
    for key, value in base.items():
        assert torch.equal(value, tuned.state_dict()[key]) != key.startswith("classifier."), key


def test_evaluate_model(example_pair: dict, example_inputs: dict) -> None:
    from road_surface_classifier.model import load_model

    from models.dinov3s_road_surface_classification.pipeline import evaluate_model

    weights = example_pair["weights"]
    model = load_model(weights / "head.pt", weights / "encoder.pt")
    metrics = evaluate_model.entrypoint(trained_model=model, **example_inputs)
    assert set(metrics) == {"accuracy", "loss"}
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["loss"] >= 0.0


def test_export_onnx(example_pair: dict) -> None:
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from road_surface_classifier.model import load_model

    from models.dinov3s_road_surface_classification.pipeline import (
        DINOV3_LICENSE,
        INFERENCE_PATCH_CONFIG_KWARGS,
        export_onnx,
    )

    weights = example_pair["weights"]
    model = load_model(weights / "head.pt", weights / "encoder.pt").eval()
    onnx_bytes = export_onnx.entrypoint(trained_model=model)
    assert isinstance(onnx_bytes, bytes)
    loaded = onnx.load_from_string(onnx_bytes)
    assert len(loaded.graph.input) == 1
    assert len(loaded.graph.output) == 1

    # The DINOv3 license is bundled in the model file.
    session = ort.InferenceSession(onnx_bytes)
    metadata = session.get_modelmeta().custom_metadata_map
    assert metadata["license"] == DINOV3_LICENSE.read_text(encoding="utf-8")
    assert metadata["notice"] == "Built with DINOv3"

    # One bag of any size up to the patch budget gives the same logits as the torch model.
    size = INFERENCE_PATCH_CONFIG_KWARGS["patch_size_px"]
    for n_patches in range(1, INFERENCE_PATCH_CONFIG_KWARGS["max_patches_per_road"] + 1):
        images = torch.rand(n_patches, 3, size, size)
        with torch.no_grad():
            expected = model(images, [n_patches])[0].numpy()
        (logits,) = session.run(None, {"image": images.numpy()})
        np.testing.assert_allclose(logits, expected, atol=1e-4)
