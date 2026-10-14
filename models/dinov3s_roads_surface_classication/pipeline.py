import tempfile
from pathlib import Path
from typing import Annotated, Any

from zenml import log_metadata, pipeline, step

from fair.zenml.materializers import CheckpointBytesMaterializer, ONNXMaterializer

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


def _download_checkpoint(url: str) -> Path:
    from upath import UPath

    local_path = Path(tempfile.mkdtemp()) / UPath(url).name
    local_path.write_bytes(UPath(url).read_bytes())
    return local_path


def build_mosaic(chip_paths: list, out_dir: Path) -> Path:
    import rasterio
    from rasterio.merge import merge

    out_path = out_dir / "mosaic.tif"
    with rasterio.open(chip_paths[0]) as ref:
        profile = ref.profile

    merge(
        [str(p) for p in chip_paths],
        indexes=[1, 2, 3],
        dst_path=str(out_path),
        dst_kwds={
            **profile,
            "driver": "GTiff",
            "count": 3,
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

    work_dir.mkdir(parents=True, exist_ok=True)
    mosaic = build_mosaic(sorted(resolve_directory(dataset_chips, "*.tif").glob("*.tif")), work_dir)

    dataset_dir = work_dir / "dataset"
    materialize_dataset(
        [(SCENE_ID, mosaic, resolve_directory(dataset_labels, "*.geojson") / "labels.geojson")],
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


def split_roads(roads: Any, val_ratio: float, seed: int, block_size_m: float) -> tuple:
    """Spatial block split: roads are grouped by the grid block (block_size_m, in UTM metres) holding their centroid,
    and whole blocks go to val in seeded order until it holds val_ratio of the roads; at least one block stays in
    train. With a single block every road becomes its own block, i.e. a random split by road."""
    import numpy as np

    centroids = roads.to_crs(roads.estimate_utm_crs()).geometry.centroid
    blocks = {}
    for key, x, y in zip(roads["road_key"], centroids.x, centroids.y, strict=True):
        blocks.setdefault((int(x // block_size_m), int(y // block_size_m)), []).append(int(key))
    if len(blocks) < 2:
        blocks = {key: [key] for key in roads["road_key"].astype(int)}

    groups = [blocks[block] for block in sorted(blocks)]
    order = np.random.default_rng(seed).permutation(len(groups))
    target = max(1, round(val_ratio * len(roads)))
    val_keys = []
    for i in order[:-1]:
        if len(val_keys) >= target:
            break
        val_keys += groups[i]
    train_keys = [key for group in groups for key in group if key not in val_keys]
    return sorted(train_keys), sorted(val_keys), len(groups)


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
    seed = hyperparameters.get("split_seed", 42)
    block_size_m = hyperparameters.get("block_size_m", 100.0)
    train_keys, val_keys, n_blocks = split_roads(roads, val_ratio, seed, block_size_m)

    def counts_per_class(keys: list) -> dict:
        labels = roads.set_index("road_key").loc[keys, LABEL_COLUMN].tolist()
        return {name: labels.count(i) for i, name in enumerate(CLASS_NAMES)}

    split_info = {
        "strategy": "spatial",
        "val_ratio": val_ratio,
        "seed": seed,
        "block_size_m": block_size_m,
        "block_count": n_blocks,
        "train_count": len(train_keys),
        "val_count": len(val_keys),
        "train_counts_per_class": counts_per_class(train_keys),
        "val_counts_per_class": counts_per_class(val_keys),
        "train_road_keys": train_keys,
        "val_road_keys": val_keys,
        "class_names": list(CLASS_NAMES),
        "description": (
            f"Roads grouped into {block_size_m:g} m grid blocks by their centroid; whole blocks go to val in seeded "
            f"order until it holds about {val_ratio:.0%} of the roads. With a single block, roads are split at random. "
            "All bags of a road stay on one side; the split is not stratified by class."
        ),
    }
    log_metadata(metadata={"fair/split": split_info})
    return split_info


@step(output_materializers={"trained_model_artifact": CheckpointBytesMaterializer})
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
) -> Annotated[bytes, "trained_model_artifact"]:
    raise NotImplementedError("training is step 2")


@step
def evaluate_model(
    trained_model: bytes,
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict,
    split_info: dict,
    class_names: list | None = None,
) -> dict:
    raise NotImplementedError("evaluation is step 2")


@step(output_materializers={"onnx_model": ONNXMaterializer})
def export_onnx(trained_model: bytes) -> Annotated[bytes, "onnx_model"]:
    raise NotImplementedError("ONNX export is step 3")


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
