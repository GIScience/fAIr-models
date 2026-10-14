import tempfile
from pathlib import Path
from typing import Annotated, Any

from zenml import log_metadata, pipeline, step

from fair.zenml.instrumentation import log_evaluation_results
from fair.zenml.materializers import CheckpointBytesMaterializer, ONNXMaterializer

CLASS_NAMES = ("unpaved", "paved")


def _get_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _download_checkpoint(url: str) -> Path:
    from upath import UPath

    local_path = Path(tempfile.mkdtemp()) / UPath(url).name
    local_path.write_bytes(UPath(url).read_bytes())
    return local_path


def _log_yolo_loss_history(model: Any) -> None:
    import csv

    from fair.zenml.metrics import log_loss_history

    save_dir = getattr(model.trainer, "save_dir", None) if hasattr(model, "trainer") else None
    if save_dir is None:
        return
    results_csv = Path(save_dir) / "results.csv"
    if not results_csv.exists():
        return

    train_losses: list[float] = []
    val_losses: list[float] = []
    with results_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            stripped = {k.strip(): v.strip() for k, v in row.items()}
            train_loss = stripped.get("train/loss")
            val_loss = stripped.get("val/loss")
            if train_loss is not None and val_loss is not None:
                train_losses.append(float(train_loss))
                val_losses.append(float(val_loss))

    if train_losses:
        import mlflow

        for epoch, (tl, vl) in enumerate(zip(train_losses, val_losses, strict=True)):
            mlflow.log_metric("train_loss", tl, step=epoch)  # ty: ignore[possibly-missing-attribute]
            mlflow.log_metric("val_loss", vl, step=epoch)  # ty: ignore[possibly-missing-attribute]
        log_loss_history(train_losses, val_losses)


def predict(session: Any, input_images: str, params: dict[str, Any], bbox: list[float] | None = None) -> dict[str, Any]:
    pass


def pick_utm_crs(lon: float, lat: float) -> Any:
    import geopandas as gpd

    gdf = gpd.GeoSeries(gpd.points_from_xy([lon], [lat]), crs="EPSG:4326")
    return gdf.estimate_utm_crs()


def build_mosaic(chip_paths: list[Path]) -> Path:
    import rasterio
    from rasterio.merge import merge

    out_dir = Path(tempfile.mkdtemp(prefix="yolo_cls_mosaic_"))
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
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "compress": "deflate",
            "photometric": "rgb",
        },
        mem_limit=512,
    )
    return out_path


def resolve_chip_paths(chips_path: str) -> list[Path]:
    from fair.utils.data import resolve_directory

    return sorted(resolve_directory(chips_path).rglob("*.tif"))


def resolve_label_file(labels_path: str) -> Path:
    from fair.utils.data import resolve_directory

    labels_dir = resolve_directory(labels_path)
    if labels_dir.is_file():
        return labels_dir
    return sorted([*labels_dir.rglob("*.geojson"), *labels_dir.rglob("*.gpkg")])[0]


@step
def split_dataset(
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
) -> Annotated[dict[str, Any], "split_info_artifact"]:

    split_info = {
        "strategy": "grid_5m_stratified_grouped",
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "train_counts_per_class": train_counts,
        "val_counts_per_class": val_counts,
        "test_counts_per_class": test_counts,
        "waste_overlap_threshold": waste_overlap_threshold,
        "cell_size_m": cell_size_m,
        "class_names": list(CLASS_NAMES),
        "description": (
            f"5 m x 5 m grid over the chip mosaic; cells with >= "
            f"{waste_overlap_threshold:.0%} label coverage are 'waste', "
            f"else 'background'. Waste and background cells are grouped by their source "
            f"label polygon. Validation and test target "
            f"{val_ratio:.0%} and {test_ratio:.0%} of source-polygon groups."
        ),
        "_yolo_dir": str(yolo_dir),
    }
    log_metadata(metadata={"fair/split": {k: v for k, v in split_info.items() if not k.startswith("_")}})
    return split_info


@step(output_materializers={"trained_model_artifact": CheckpointBytesMaterializer})
def train_model(
    dataset_chips: str,
    dataset_labels: str,
    base_model_weights: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    num_classes: int = 2,
    model_name: str | None = None,
    base_model_id: str | None = None,
    dataset_id: str | None = None,
) -> Annotated[Any, "trained_model_artifact"]:

    local_weights = _download_checkpoint(base_model_weights)
    device = _get_device()

    model.save(str(saved_path))
    return saved_path.read_bytes()


@step
def evaluate_model(
    trained_model: Any,
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    class_names: list[str] | None = None,
) -> Annotated[dict[str, Any], "metrics"]:
    chip_size = hyperparameters.get("chip_size", MODEL_INPUT_SIZE)
    metrics_dict: dict[str, Any] = {"accuracy": float(accuracy)}
    log_evaluation_results(metrics_dict)
    return metrics_dict


@step(output_materializers={"onnx_model": ONNXMaterializer})
def export_onnx(trained_model: Any) -> Annotated[bytes, "onnx_model"]:
    import onnx

    model = _restore_checkpoint(trained_model)
    onnx_path = model.export(format="onnx")
    proto = onnx.load(onnx_path)
    onnx.save_model(proto, onnx_path, save_as_external_data=False)
    onnx.checker.check_model(onnx_path)
    try:
        return Path(onnx_path).read_bytes()
    finally:
        Path(onnx_path).unlink(missing_ok=True)


@step
def run_inference(
    model_uri: str,
    input_images: str,
    inference_params: dict[str, Any],
) -> Annotated[dict[str, Any], "predictions"]:
    from fair.serve.base import load_session

    session = load_session(model_uri)
    return predict(session, input_images, inference_params)


@pipeline
def training_pipeline(
    base_model_weights: str,
    dataset_chips: str,
    dataset_labels: str,
    num_classes: int,
    hyperparameters: dict[str, Any],
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
    inference_params: dict[str, Any],
) -> None:
    run_inference(
        model_uri=model_uri,
        input_images=input_images,
        inference_params=inference_params,
    )
