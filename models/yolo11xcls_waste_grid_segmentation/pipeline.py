import hashlib
import random
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Any

from zenml import log_metadata, pipeline, step

from fair.zenml.instrumentation import log_evaluation_results, mlflow_training_context
from fair.zenml.materializers import CheckpointBytesMaterializer, ONNXMaterializer

MODEL_INPUT_SIZE = 128
CELL_SIZE_M = 5.0
CLASS_NAMES = ("background", "waste")
CHIP_SIZE = MODEL_INPUT_SIZE


def _get_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _download_checkpoint(url: str) -> Path:
    from upath import UPath

    print(url)
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


def _restore_checkpoint(trained_model: Any):
    from ultralytics import YOLO

    if isinstance(trained_model, YOLO):
        return trained_model
    if isinstance(trained_model, bytes):
        checkpoint = Path(tempfile.mkdtemp()) / "best.pt"
        checkpoint.write_bytes(trained_model)
        return YOLO(str(checkpoint))
    return YOLO(trained_model)


def preprocess(image_path: Any, chip_size: int = 640) -> Any:
    import numpy as np
    import rasterio
    import torch
    import torch.nn.functional as F

    with rasterio.open(image_path) as src:
        arr = src.read([1, 2, 3]).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0)
    if tensor.shape[-2:] != (chip_size, chip_size):
        tensor = F.interpolate(tensor, size=(chip_size, chip_size), mode="bilinear", align_corners=False)
    return tensor


def postprocess(results: Any) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for result in results:
        for box in result.boxes:
            detections.append(
                {
                    "bbox": box.xyxy[0].tolist(),
                    "confidence": box.conf.item(),
                    "class": int(box.cls.item()),
                }
            )
    return detections


def _preprocess_onnx_image(img_path: Any) -> tuple[Any, Any, Any]:
    import numpy as np
    import rasterio
    from PIL import Image

    with rasterio.open(img_path) as src:
        arr = src.read([1, 2, 3]).astype(np.float32) / 255.0
        transform = src.transform
        crs = src.crs

    resized = [
        np.asarray(Image.fromarray(arr[c]).resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.Resampling.BILINEAR))
        for c in range(arr.shape[0])
    ]
    batch = np.stack(resized, axis=0)[np.newaxis, ...].astype(np.float32)
    return batch, transform, crs


def _nms(boxes: Any, scores: Any, iou_threshold: float) -> list[int]:
    import numpy as np

    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_threshold]
    return keep


def _decode_yolo_output(
    output: Any,
    confidence_threshold: float,
    iou_threshold: float,
) -> list[dict[str, Any]]:
    """Decode ultralytics YOLO ONNX output: shape (1, 4+nc, num_anchors)."""
    import numpy as np

    preds = np.squeeze(output, axis=0)
    if preds.shape[0] < preds.shape[1]:
        preds = preds.transpose(1, 0)

    boxes_cxcywh = preds[:, :4]
    class_scores = preds[:, 4:]
    if class_scores.shape[1] == 0:
        return []
    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores.max(axis=1)

    keep_mask = confidences >= confidence_threshold
    if not keep_mask.any():
        return []
    boxes_cxcywh = boxes_cxcywh[keep_mask]
    confidences = confidences[keep_mask]
    class_ids = class_ids[keep_mask]

    cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    boxes_xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    keep_idx = _nms(boxes_xyxy, confidences, iou_threshold)

    scale = CHIP_SIZE / MODEL_INPUT_SIZE
    detections: list[dict[str, Any]] = []
    for idx in keep_idx:
        x1, y1, x2, y2 = boxes_xyxy[idx] * scale
        detections.append(
            {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(confidences[idx]),
                "class": int(class_ids[idx]),
            }
        )
    return detections


def predict(session: Any, input_images: str, params: dict[str, Any]) -> dict[str, Any]:
    from fair.utils.data import resolve_directory

    if "confidence_threshold" not in params:
        raise ValueError("params['confidence_threshold'] is required")
    confidence_threshold = float(params["confidence_threshold"])
    iou_threshold = float(params.get("iou_threshold", 0.45))
    input_name = session.get_inputs()[0].name

    input_dir = resolve_directory(input_images)
    patterns = ("*.png", "*.tif", "*.tiff", "*.jpg")
    img_paths = sorted(p for pat in patterns for p in input_dir.glob(pat))
    if not img_paths:
        msg = f"No input images found in {input_dir}"
        raise FileNotFoundError(msg)

    features: list[dict[str, Any]] = []
    for img_path in img_paths:
        batch, transform, crs = _preprocess_onnx_image(img_path)
        output = session.run(None, {input_name: batch})[0]
        for det in _decode_yolo_output(output, confidence_threshold, iou_threshold):
            feature = _pixel_bbox_to_geo_feature(
                det["bbox"],
                transform,
                crs,
                {
                    "confidence": round(det["confidence"], 4),
                    "class": det["class"],
                    "source": img_path.name,
                },
            )
            features.append(feature)
    return _build_feature_collection(features)


def dataset_cache_dir(chips_path: str, labels_path: str, threshold: float, cell_size_m: float) -> Path:
    key = hashlib.sha256(f"{chips_path}|{labels_path}|{threshold}|{cell_size_m}".encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"yolo_cls_dataset_{key}"


def _subset_chips_dir(chips_path: str, fraction: float) -> str:
    if fraction >= 1.0:
        return chips_path
    from fair.utils.data import resolve_directory

    chips = sorted(resolve_directory(chips_path).rglob("*.tif"))
    step = max(1, round(1 / fraction))
    subset = Path(tempfile.mkdtemp(prefix="yolo_chips_subset_"))
    for chip in chips[::step]:
        (subset / chip.name).symlink_to(chip)
        sidecar = chip.with_name(chip.name + ".aux.xml")
        (subset / sidecar.name).symlink_to(sidecar)
    return str(subset)


def pick_utm_crs(lon: float, lat: float):
    from rasterio.crs import CRS

    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


def build_mosaic(chip_paths: list[Path]) -> Path:
    import subprocess

    out_dir = Path(tempfile.mkdtemp(prefix="yolo_cls_mosaic_"))
    vrt_path = out_dir / "mosaic.vrt"
    filelist = out_dir / "chips.txt"
    filelist.write_text("\n".join(str(p) for p in chip_paths))
    subprocess.run(
        ["gdalbuildvrt", "-input_file_list", str(filelist), str(vrt_path)],
        check=True,
        capture_output=True,
    )
    return vrt_path


def load_labels_merged(labels_dir: Path, target_crs):
    import geopandas as gpd
    import pandas as pd

    files = sorted(labels_dir.rglob("*.geojson"))
    if not files:
        msg = f"No .geojson files under {labels_dir}"
        raise FileNotFoundError(msg)
    frames = [gpd.read_file(f) for f in files]
    src_crs = next((f.crs for f in frames if f.crs is not None), "EPSG:4326")
    combined = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=src_crs)
    combined = combined.to_crs(target_crs)
    return combined.geometry.union_all()


def build_grid_gdf(bounds_proj, cell_size: float, crs):
    import geopandas as gpd
    from shapely.geometry import box

    minx, miny, maxx, maxy = bounds_proj
    minx = (minx // cell_size) * cell_size
    miny = (miny // cell_size) * cell_size
    cols = max(1, int((maxx - minx) // cell_size) + 1)
    rows = max(1, int((maxy - miny) // cell_size) + 1)

    cells = []
    cell_id = 0
    for r in range(rows):
        for c in range(cols):
            x = minx + c * cell_size
            y = miny + r * cell_size
            cells.append({"cell_id": cell_id, "geometry": box(x, y, x + cell_size, y + cell_size)})
            cell_id += 1
    return gpd.GeoDataFrame(cells, crs=crs)


def classify_cells(grid_gdf, labels_union, threshold: float):
    grid_gdf = grid_gdf.copy()
    intersections = grid_gdf.geometry.intersection(labels_union)
    grid_gdf["overlap_fraction"] = (intersections.area / grid_gdf.geometry.area).fillna(0.0)
    grid_gdf["label"] = (grid_gdf["overlap_fraction"] >= threshold).astype(int)
    return grid_gdf


def read_cell_array_from_mosaic(mosaic_ds, cell_geom, utm_to_mosaic, nodata) -> Any:
    from rasterio.windows import from_bounds

    minx, miny, maxx, maxy = cell_geom.bounds
    xs = [minx, minx, maxx, maxx]
    ys = [miny, maxy, miny, maxy]
    lons, lats = utm_to_mosaic.transform(xs, ys)
    west, east = min(lons), max(lons)
    south, north = min(lats), max(lats)

    window = from_bounds(west, south, east, north, transform=mosaic_ds.transform)

    arr = mosaic_ds.read(
        [1, 2, 3],
        window=window,
        boundless=True,
        fill_value=0,
        out_dtype="uint8",
    )
    if arr.size == 0:
        return None

    # ToDO check for NoData
    if nodata is not None:
        mask = (arr == nodata).all(axis=0)
        if mask.mean() > 0.8:  # min covered area, same as min threshold as for waste intersetcion maybe?
            return None
    return arr


def save_arr_as_png(arr: Any, out_path: Path) -> None:
    import numpy as np
    from PIL import Image

    rgb = np.transpose(arr, (1, 2, 0)).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(out_path, format="PNG")


def _prepare_yolo_classification_dataset(
    chips_path: str,
    labels_path: str,
    waste_overlap_threshold: float,
    cell_size_m: float = CELL_SIZE_M,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[Path, dict[str, int], dict[str, int]]:
    import rasterio
    from pyproj import Transformer

    from fair.utils.data import resolve_directory

    yolo_dir = dataset_cache_dir(chips_path, labels_path, waste_overlap_threshold, cell_size_m)

    local_chips = resolve_directory(chips_path)
    chip_paths = sorted(local_chips.rglob("*.tif"))
    if not chip_paths:
        msg = f"No chips under {chips_path}"
        raise FileNotFoundError(msg)

    labels_dir = resolve_directory(labels_path, "*.geojson")

    mosaic_path = build_mosaic(chip_paths)

    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)
    for split in ("train", "val"):
        for cls in CLASS_NAMES:
            (yolo_dir / split / cls).mkdir(parents=True)

    train_counts: dict[str, int] = {cls: 0 for cls in CLASS_NAMES}
    val_counts: dict[str, int] = {cls: 0 for cls in CLASS_NAMES}

    with rasterio.open(mosaic_path) as mosaic:
        mosaic_crs = mosaic.crs
        bounds = mosaic.bounds
        nodata = mosaic.nodata

        centroid_lon = (bounds.left + bounds.right) / 2
        centroid_lat = (bounds.bottom + bounds.top) / 2
        target_crs = pick_utm_crs(centroid_lon, centroid_lat) if mosaic_crs.is_geographic else mosaic_crs

        to_proj = Transformer.from_crs(mosaic_crs, target_crs, always_xy=True)
        xs = [bounds.left, bounds.left, bounds.right, bounds.right]
        ys = [bounds.bottom, bounds.top, bounds.bottom, bounds.top]
        px, py = to_proj.transform(xs, ys)
        bounds_proj = (min(px), min(py), max(px), max(py))

        grid = build_grid_gdf(bounds_proj, cell_size_m, target_crs)
        labels_union = load_labels_merged(labels_dir, target_crs)
        grid = classify_cells(grid, labels_union, waste_overlap_threshold)

        utm_to_mosaic = Transformer.from_crs(target_crs, mosaic_crs, always_xy=True)
        rng = random.Random(seed)

        for label_value, cls_name in enumerate(CLASS_NAMES):
            class_cells = grid[grid["label"] == label_value]
            cell_ids = list(class_cells.index)
            rng.shuffle(cell_ids)
            n_val = int(round(len(cell_ids) * val_ratio)) if cell_ids else 0
            val_set = set(cell_ids[:n_val])

            for idx in cell_ids:
                arr = read_cell_array_from_mosaic(
                    mosaic,
                    grid.loc[idx, "geometry"],
                    utm_to_mosaic,
                    nodata,
                )
                if arr is None or arr.size == 0:
                    continue
                split = "val" if idx in val_set else "train"
                out_path = yolo_dir / split / cls_name / f"cell_{int(grid.loc[idx, 'cell_id']):08d}.png"
                save_arr_as_png(arr, out_path)
                if split == "val":
                    val_counts[cls_name] += 1
                else:
                    train_counts[cls_name] += 1

    return yolo_dir, train_counts, val_counts


@step
def split_dataset(
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
) -> Annotated[dict[str, Any], "split_info_artifact"]:
    val_ratio = hyperparameters.get("val_ratio", 0.2)
    seed = hyperparameters.get("split_seed", 42)
    waste_overlap_threshold = hyperparameters.get("waste_overlap_threshold", 0.8)
    cell_size_m = hyperparameters.get("cell_size_m", CELL_SIZE_M)

    chips_path = _subset_chips_dir(dataset_chips, hyperparameters.get("sample_fraction", 1.0))
    yolo_dir, train_counts, val_counts = _prepare_yolo_classification_dataset(
        chips_path,
        dataset_labels,
        waste_overlap_threshold=waste_overlap_threshold,
        cell_size_m=cell_size_m,
        val_ratio=val_ratio,
        seed=seed,
    )

    train_count = sum(train_counts.values())
    val_count = sum(val_counts.values())
    split_info = {
        "strategy": "grid_5m_stratified_random",
        "val_ratio": val_ratio,
        "seed": seed,
        "train_count": train_count,
        "val_count": val_count,
        "train_counts_per_class": train_counts,
        "val_counts_per_class": val_counts,
        "waste_overlap_threshold": waste_overlap_threshold,
        "cell_size_m": cell_size_m,
        "class_names": list(CLASS_NAMES),
        "description": (
            f"5 m x 5 m grid over the chip mosaic; cells with >= "
            f"{waste_overlap_threshold:.0%} label coverage are 'waste', "
            f"else 'background'. Stratified random split holds out {val_ratio:.0%} per class."
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
    from ultralytics import YOLO
    from ultralytics import settings as yolo_settings

    epochs = hyperparameters["epochs"]
    batch_size = hyperparameters.get("batch_size", 8)
    chip_size = hyperparameters.get("chip_size", MODEL_INPUT_SIZE)
    learning_rate = hyperparameters.get("learning_rate", 0.01)
    freeze_encoder = hyperparameters.get("freeze_encoder", True)

    yolo_dir = Path(split_info["_yolo_dir"])
    if not yolo_dir.exists() or not all(
        (yolo_dir / split / cls).exists() for split in ("train", "val") for cls in CLASS_NAMES
    ):
        chips_path = _subset_chips_dir(dataset_chips, hyperparameters.get("sample_fraction", 1.0))
        yolo_dir, _, _ = _prepare_yolo_classification_dataset(
            chips_path,
            dataset_labels,
            waste_overlap_threshold=split_info.get("waste_overlap_threshold", 0.8),
            cell_size_m=split_info.get("cell_size_m", CELL_SIZE_M),
            val_ratio=split_info["val_ratio"],
            seed=split_info.get("seed", 42),
        )

    yolo_settings.update({"mlflow": False})
    local_weights = _download_checkpoint(base_model_weights)
    device = _get_device()

    with mlflow_training_context(
        hyperparameters,
        model_name,
        base_model_id,
        dataset_id,
    ):
        model = YOLO(str(local_weights), task="classify")
        results = model.train(
            data=str(yolo_dir),
            epochs=epochs,
            batch=batch_size,
            imgsz=chip_size,
            device=device,
            lr0=learning_rate,
            freeze=10 if freeze_encoder else 0,
            cos_lr=True,
            verbose=False,
        )
        if results and hasattr(results, "results_dict"):
            top1 = results.results_dict.get("metrics/accuracy_top1", 0.0)
            log_metadata(metadata={"accuracy_top1": float(top1), "epoch": epochs})

        _log_yolo_loss_history(model)

    saved_path = Path(tempfile.mkdtemp()) / "best.pt"
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

    yolo_dir = Path(split_info["_yolo_dir"])
    if not yolo_dir.exists() or not all(
        (yolo_dir / split / cls).exists() for split in ("train", "val") for cls in CLASS_NAMES
    ):
        chips_path = _subset_chips_dir(dataset_chips, hyperparameters.get("sample_fraction", 1.0))
        yolo_dir, _, _ = _prepare_yolo_classification_dataset(
            chips_path,
            dataset_labels,
            waste_overlap_threshold=split_info.get("waste_overlap_threshold", 0.8),
            cell_size_m=split_info.get("cell_size_m", CELL_SIZE_M),
            val_ratio=split_info["val_ratio"],
            seed=split_info.get("seed", 42),
        )

    model = _restore_checkpoint(trained_model)
    results = model.val(data=str(yolo_dir), imgsz=chip_size, verbose=False)

    if not hasattr(results, "results_dict") or not results.results_dict:
        msg = "YOLO validation produced no results"
        raise RuntimeError(msg)

    metrics_dict: dict[str, Any] = {
        "accuracy": float(results.results_dict.get("metrics/accuracy_top1", 0.0)),
    }
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
