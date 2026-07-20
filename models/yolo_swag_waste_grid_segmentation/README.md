# YOLO Solid Waste Grid Segmentation (SWAG)

Model for detecting solid waste piles from UAV imagery using grid-based classification.

## Architecture

- **Model**: YOLO26x-cls
- **Framework**: PyTorch 2.10.0
- **Task**: Semantic segmentation (grid-based)
- **Input**: RGB chips (128x128, float32)
- **Classes**: 2 (background, waste)

## Dataset

- **Source**: OpenAerialMap (OAM) scenes
- **Coverage**: 60 globally distributed scenes
- **Labels**: Polygon annotations for solid waste piles
- **Grid Strategy**: 5m x 5m stratified random grid cells

## Pipeline

Training and inference pipeline defined in `models.yolo_swag_waste_grid_segmentation.pipeline`:

- **Preprocessing**: `preprocess` – Normalizes RGB chips for input.
- **Training**: Fine-tunes YOLO26x-cls on grid-labeled chips.
- **Postprocessing**: `postprocess` – Converts model output to waste/no-waste grid classifications.
- **Evaluation**: Computes accuracy for per-cell classification.

Inference produces 5m x 5m grid classifications for waste as polygons from the input imagery.
