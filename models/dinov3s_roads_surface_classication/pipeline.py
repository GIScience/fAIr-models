import tempfile
from pathlib import Path
from typing import Annotated, Any

from zenml import log_metadata, pipeline, step

from fair.zenml.materializers import ONNXMaterializer

CLASS_NAMES = ("unpaved", "paved")

LABEL_COLUMN = "label"
SCENE_ID = "scene"
MATERIALIZE_SEED = 0

PATCH_CONFIG_KWARGS = {
    "patch_size_m": 15.0,
    "patch_size_px": 256,
    "meters_per_patch": 20.0,
    "min_patches_per_road": 1,
    "max_patches_per_road": 4,
    "n_bags": 5,
    "endpoint_margin_fraction": 0.1,
    "junction_clearance_m": 7.5,
    "lateral_jitter_m": 1.5,
    "bearing_jitter_deg": 10.0,
    "max_nodata_fraction": 0.1,
    "max_attempts_per_patch": 10,
    "fill_tolerance": 1,
    "junction_exclude_highway_values": [
        "footway",
        "path",
        "cycleway",
        "steps",
        "pedestrian",
        "corridor",
        "platform",
        "bus_stop",
        "crossing",
        "give_way",
        "stop",
        "traffic_signals",
        "traffic_mirror",
        "street_lamp",
        "milestone",
        "emergency_access_point",
        "construction",
    ],
}


def build_mosaic(chip_paths: list, out_dir: Path) -> Path:
    import rasterio
    from rasterio.merge import merge

    out_path = out_dir / "mosaic.tif"
    with rasterio.open(chip_paths[0]) as ref:
        profile = ref.profile

    merge(
        [str(p) for p in chip_paths],
        dst_path=str(out_path),
        dst_kwds={
            **profile,
            "driver": "GTiff",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "compress": "deflate",
            "photometric": "rgb",
        },
        mem_limit=512,
    )
    return out_path


def load_labelled_roads(dataset_labels: str) -> Any:
    import geopandas as gpd

    from fair.utils.data import resolve_directory

    lines = gpd.read_file(resolve_directory(dataset_labels, "*.geojson") / "labels.geojson")
    roads = lines.dropna(subset=[LABEL_COLUMN]).rename_axis("road_key").reset_index()
    return roads.astype({LABEL_COLUMN: int})


def materialize(dataset_chips: str, dataset_labels: str, work_dir: Path) -> Path:
    """Cut bags along every labelled road into the source repo's dataset layout; deterministic, so each step can
    rebuild it locally. Lines without a label only count as junctions."""
    from road_surface_classifier.data.materialize import PatchConfig, materialize_dataset

    from fair.utils.data import resolve_directory

    chips = sorted(resolve_directory(dataset_chips, "*.tif").glob("*.tif"))
    raster = chips[0] if len(chips) == 1 else build_mosaic(chips, work_dir)

    dataset_dir = work_dir / "dataset"
    materialize_dataset(
        [(SCENE_ID, raster, resolve_directory(dataset_labels, "*.geojson") / "labels.geojson")],
        dataset_dir,
        LABEL_COLUMN,
        PatchConfig(**PATCH_CONFIG_KWARGS),
        seed=MATERIALIZE_SEED,
    )
    return dataset_dir


def road_subset(dataset: Any, road_keys: list) -> Any:
    """The bags of `dataset` (a PatchDataset) whose road came from one of `road_keys`."""
    from torch.utils.data import Subset

    keys = set(road_keys)
    return Subset(dataset, [i for i, key in enumerate(dataset.roads["source_row"]) if key in keys])


def split_roads(roads: Any, val_ratio: float, test_ratio: float, seed: int, block_size_m: float) -> tuple:
    import numpy as np

    centroids = roads.to_crs(roads.estimate_utm_crs()).geometry.centroid
    blocks = {}
    for key, x, y in zip(roads["road_key"], centroids.x, centroids.y, strict=True):
        blocks.setdefault((int(x // block_size_m), int(y // block_size_m)), []).append(int(key))
    if len(blocks) < 3:
        blocks = {key: [key] for key in roads["road_key"].astype(int)}

    groups = [blocks[block] for block in sorted(blocks)]
    test_target = max(1, round(test_ratio * len(roads)))
    val_target = max(1, round(val_ratio * len(roads)))
    train_keys, val_keys, test_keys = [], [], []
    for assigned, i in enumerate(np.random.default_rng(seed).permutation(len(groups))):
        remaining = len(groups) - assigned  # blocks not yet assigned, this one included
        if len(test_keys) < test_target and remaining > 2:
            test_keys += groups[i]
        elif len(val_keys) < val_target and remaining > 1:
            val_keys += groups[i]
        else:
            train_keys += groups[i]
    return sorted(train_keys), sorted(val_keys), sorted(test_keys), len(groups)


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def replay_bags(bank: dict) -> tuple:
    """The replay bank's bags: per-patch backbone features [n_patches, 384] of each bag, and its label. Label 2
    (paved_segmental) is paved."""
    import torch

    patches = bank["patches"]
    bags = {}
    for row, key in enumerate(zip(patches["scene_id"], patches["road_id"], patches["bag"], strict=True)):
        bags.setdefault(key, []).append(row)
    features = bank["features"].float()
    labels = torch.tensor([min(int(patches["label"][rows[0]]), 1) for rows in bags.values()])
    return [features[rows] for rows in bags.values()], labels


def load_dataset(dataset_dir: Path, road_keys: list, seed: int, augment: bool = False) -> Any:
    from road_surface_classifier.data.augmentation import build_bag_augmentations, build_patch_augmentations
    from road_surface_classifier.data.dataset import PatchDataset

    label_map = {i: i for i in range(len(CLASS_NAMES))}
    if augment:
        dataset = PatchDataset(
            dataset_dir,
            label_map,
            seed=seed,
            bag_transform=build_bag_augmentations(),
            patch_transform=build_patch_augmentations(PATCH_CONFIG_KWARGS["patch_size_px"]),
        )
    else:
        dataset = PatchDataset(dataset_dir, label_map, seed=seed, mode="eval")
    return road_subset(dataset, road_keys)


def load_base_model(head_path: Path, encoder_path: Path) -> Any:
    from road_surface_classifier.model import load_model

    model = load_model(head_path, encoder_path, trainable="classifier")
    model.eval().requires_grad_(False)
    model.classifier.requires_grad_(True)
    return model


def evaluate_dataset(model: Any, dataset: Any, device: str) -> tuple:
    """Loss and accuracy of the model on `dataset`, all its bags in one batch."""
    import torch
    import torch.nn.functional as F
    from road_surface_classifier.data.dataset import collate_bags

    images, sizes, labels = collate_bags([dataset[i] for i in range(len(dataset))])
    with torch.no_grad():
        logits = model(images.to(device), sizes)[0].cpu()
    loss = F.cross_entropy(logits, labels, weight=model.class_weights.cpu()).item()
    return loss, (logits.argmax(1) == labels).float().mean().item()


def train_classifier(
    model: Any,
    train_dataset: Any,
    val_dataset: Any,
    replay_features: list,
    replay_labels: Any,
    steps: int,
    replay_bags_per_step: int,
    learning_rate: float,
    val_every: int,
    device: str,
) -> None:
    import mlflow
    import torch
    import torch.nn.functional as F
    from road_surface_classifier.data.dataset import collate_bags

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=learning_rate, weight_decay=model.hparams.weight_decay
    )
    for step_index in range(1, steps + 1):
        images, sizes, labels = collate_bags([train_dataset[i] for i in range(len(train_dataset))])
        replay = torch.randint(len(replay_labels), (replay_bags_per_step,))
        replay_pooled = torch.stack([model.pool(replay_features[i].to(device))[0] for i in replay.tolist()])
        logits = torch.cat([model(images.to(device), sizes)[0], model.classifier(replay_pooled)])
        labels = torch.cat([labels, replay_labels[replay]]).to(device)
        loss = F.cross_entropy(
            logits, labels, weight=model.class_weights, label_smoothing=model.hparams.label_smoothing
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), 1.0)
        optimizer.step()
        mlflow.log_metric("train_loss", loss.item(), step=step_index)  # ty: ignore[possibly-missing-attribute]

        if step_index % val_every == 0 or step_index == steps:
            val_loss, val_accuracy = evaluate_dataset(model, val_dataset, device)
            mlflow.log_metric("val_loss", val_loss, step=step_index)  # ty: ignore[possibly-missing-attribute]
            mlflow.log_metric("val_accuracy", val_accuracy, step=step_index)  # ty: ignore[possibly-missing-attribute]


def predict(session: Any, input_images: str, params: dict) -> dict:
    raise NotImplementedError("road inference is step 3")


def preprocess(image_path: Any) -> Any:
    raise NotImplementedError("road inference is step 3")


def postprocess(logits: Any) -> Any:
    raise NotImplementedError("road inference is step 3")


@step
def split_dataset(
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict,
) -> dict:
    roads = load_labelled_roads(dataset_labels)

    val_ratio = hyperparameters.get("val_ratio", 0.2)
    test_ratio = hyperparameters.get("test_ratio", 0.2)
    seed = hyperparameters.get("split_seed", 42)
    block_size_m = hyperparameters.get("block_size_m", 100.0)
    train_keys, val_keys, test_keys, n_blocks = split_roads(roads, val_ratio, test_ratio, seed, block_size_m)

    def counts_per_class(keys: list) -> dict:
        labels = roads.set_index("road_key").loc[keys, LABEL_COLUMN].tolist()
        return {name: labels.count(i) for i, name in enumerate(CLASS_NAMES)}

    split_info = {
        "strategy": "spatial",
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "block_size_m": block_size_m,
        "block_count": n_blocks,
        "train_count": len(train_keys),
        "val_count": len(val_keys),
        "test_count": len(test_keys),
        "train_counts_per_class": counts_per_class(train_keys),
        "val_counts_per_class": counts_per_class(val_keys),
        "test_counts_per_class": counts_per_class(test_keys),
        "train_road_keys": train_keys,
        "val_road_keys": val_keys,
        "test_road_keys": test_keys,
        "class_names": list(CLASS_NAMES),
        "description": (
            f"Roads grouped into {block_size_m:g} m grid blocks by their centroid; in seeded block order, test gets "
            f"about {test_ratio:.0%} and val about {val_ratio:.0%} of the roads, the rest is train. With fewer than "
            "3 blocks, roads are split at random. All bags of a road stay in one split; not stratified by class."
        ),
    }
    log_metadata(metadata={"fair/split": split_info})
    return split_info


@step
def train_model(
    dataset_chips: str,
    dataset_labels: str,
    base_model_weights: str,
    hyperparameters: dict,
    split_info: dict,
    num_classes: int = 2,
    model_name: str | None = None,
    base_model_id: str | None = None,
    dataset_id: str | None = None,
) -> Annotated[Any, "trained_model_artifact"]:
    import mlflow
    import torch

    from fair.utils.data import resolve_path
    from fair.zenml.instrumentation import mlflow_training_context

    device = _device()
    dataset_dir = materialize(dataset_chips, dataset_labels, Path(tempfile.mkdtemp()))
    train_dataset = load_dataset(dataset_dir, split_info["train_road_keys"], split_info["seed"], augment=True)
    val_dataset = load_dataset(dataset_dir, split_info["val_road_keys"], split_info["seed"])
    head_path = resolve_path(base_model_weights)
    encoder_path = resolve_path(hyperparameters["encoder_uri"])
    model = load_base_model(head_path, encoder_path)
    bank = torch.load(resolve_path(hyperparameters["replay_bank_uri"]), map_location="cpu")
    replay_features, replay_labels = replay_bags(bank)
    model.to(device)

    # Gate: keep the base model if it already gets the scene's labelled roads (all splits) right.
    all_road_keys = split_info["train_road_keys"] + split_info["val_road_keys"] + split_info["test_road_keys"]
    _, base_accuracy = evaluate_dataset(model, load_dataset(dataset_dir, all_road_keys, split_info["seed"]), device)
    mlflow.log_metric("base_accuracy", base_accuracy)  # ty: ignore[possibly-missing-attribute]
    if base_accuracy >= hyperparameters.get("gate_accuracy", 0.9):
        return model.cpu()

    with mlflow_training_context(hyperparameters, model_name, base_model_id, dataset_id):
        train_classifier(
            model,
            train_dataset,
            val_dataset,
            replay_features,
            replay_labels,
            steps=hyperparameters.get("steps", 200),
            replay_bags_per_step=hyperparameters.get("replay_bags_per_step", 20),
            learning_rate=hyperparameters.get("learning_rate", 1e-3),
            val_every=hyperparameters.get("val_every", 20),
            device=device,
        )

    return model.cpu()


@step
def evaluate_model(
    trained_model: Any,
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict,
    split_info: dict,
    class_names: list | None = None,
) -> dict:
    from fair.zenml.instrumentation import log_evaluation_results

    device = _device()
    model = trained_model.to(device)
    dataset_dir = materialize(dataset_chips, dataset_labels, Path(tempfile.mkdtemp()))
    loss, accuracy = evaluate_dataset(
        model, load_dataset(dataset_dir, split_info["test_road_keys"], split_info["seed"]), device
    )

    metrics = {"accuracy": accuracy, "loss": loss}
    log_evaluation_results(metrics)
    return metrics


@step(output_materializers={"onnx_model": ONNXMaterializer})
def export_onnx(trained_model: Any) -> Annotated[bytes, "onnx_model"]:
    raise NotImplementedError("ONNX export comes in a later step")


@step
def run_inference(
    model_uri: str,
    input_images: str,
    inference_params: dict,
) -> Annotated[dict, "predictions"]:
    from fair.serve.base import load_session

    session = load_session(model_uri)
    return predict(session, input_images, inference_params)


@pipeline
def training_pipeline(
    base_model_weights: str,
    dataset_chips: str,
    dataset_labels: str,
    num_classes: int,
    hyperparameters: dict,
) -> None:
    split_info = split_dataset(
        dataset_chips=dataset_chips,
        dataset_labels=dataset_labels,
        hyperparameters=hyperparameters,
    )
    trained_model = train_model(
        dataset_chips=dataset_chips,
        dataset_labels=dataset_labels,
        base_model_weights=base_model_weights,
        hyperparameters=hyperparameters,
        split_info=split_info,
        num_classes=num_classes,
    )
    evaluate_model(
        trained_model=trained_model,
        dataset_chips=dataset_chips,
        dataset_labels=dataset_labels,
        hyperparameters=hyperparameters,
        split_info=split_info,
    )
    export_onnx(trained_model=trained_model)


@pipeline
def inference_pipeline(
    model_uri: str,
    input_images: str,
    inference_params: dict,
) -> None:
    run_inference(
        model_uri=model_uri,
        input_images=input_images,
        inference_params=inference_params,
    )
