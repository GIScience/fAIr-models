# YOLO Solid Waste Grid Segmentation (SWAG)

SWAG maps openly dumped solid waste in georeferenced, very-high-resolution RGB aerial or UAV imagery. It divides the input mosaic into 5 m × 5 m cells and classifies each cell as `waste` or `background`; inference returns those cells as GeoJSON polygons in EPSG:4326. It is intended for screening and prioritising areas for review, rather than for tracing the exact boundary or volume of a waste pile.

For more information please checkout the training repository: [YOLO Solid Waste Assessment on Grids (SWAG)
](https://github.com/GIScience/solid-waste-detection-for-fAIr/tree/main)
## Summary

| | |
| --- | --- |
| Task | Grid-based solid-waste mapping |
| Input | Georeferenced three-band RGB GeoTIFF chips (`.tif` or `.tiff`) |
| Output | GeoJSON `FeatureCollection` of 5 m × 5 m grid-cell polygons with `cell_id`, `label`, and `confidence` |
| Classes | `background`, `waste` |
| Coverage | 60 globally distributed OpenAerialMap (OAM) scenes |
| Licence | MIT |

## Architecture and data preparation

The model is a two-class YOLO26x classification model. RGB cell chips are resized to 128 × 128 pixels for the model; the spatial grid and GeoJSON polygons are created by the pipeline before and after classification.

Fine-tuning starts with a mosaic of the supplied imagery and a GeoJSON or GeoPackage label file. Features with `label = 1` mark waste. Features with `label = 0` may explicitly mark background; when they are absent, the pipeline deterministically samples an equal number of no-waste cells. A cell is labelled `waste` when at least 80% of its area is covered by a waste label, then each class is split into train and validation sets.

The pretrained checkpoint, annotation utilities, OAM collection script, grid-generation script, cropper, and dataset merge/re-split scripts originate in the [SWAG source repository](https://github.com/GIScience/solid-waste-detection-for-fAIr). The source repository is MIT-licensed; its [checkpoint](https://media.githubusercontent.com/media/GIScience/solid-waste-detection-for-fAIr/main/data/checkpoint/checkpoint_v1.pt) is the base model used by this pipeline.

## Intended use

Use SWAG with north-up, georeferenced RGB imagery where a 5 m ground grid is meaningful. Typical uses are rapid, area-wide screening of urban or peri-urban imagery, producing candidate waste cells for human review, and fine-tuning from local vector annotations when imagery or waste appearance differs from the pretrained scenes.

The `confidence` property is the model's waste-class probability for a grid cell. A `waste` cell identifies a location requiring review; it is not a precise waste-pile footprint, a material classification, or an estimate of mass or volume.

## Limitations

- The model only supports three-band RGB imagery. Multispectral, SAR, DEM, grayscale, and non-georeferenced inputs are outside its supported input contract.
- Results depend on image quality, ground sampling distance, illumination, occlusion, and whether local waste appearance resembles the OAM training scenes. Local fine-tuning is recommended before operational use in a new geography or sensor setting.
- The fixed 5 m grid trades boundary detail for consistent area coverage. A positive cell can include both waste and non-waste land, and small piles can be missed when they do not cover the configured threshold of a cell.
- Tile zoom is not a fixed ground resolution: metres per pixel vary with latitude, and an OAM service can resample imagery beyond its native resolution. The serving runtime currently accepts its global zoom range without enforcing this model's STAC zoom metadata. Check the source imagery's native ground sampling distance and the number of native pixels per 5 m cell before relying on a prediction.
- Background sampling and the validation split are stratified by cell, not by independent field campaign. Reported accuracy should therefore not be interpreted as a guarantee of performance in another area.
- Predictions should be reviewed by a domain expert before publication, enforcement, or resource-allocation decisions.

## How to use

For the full local fAIr workflow, run `just example yolo_swag_waste_grid_segmentation` after setting up the compose stack; this registers the model, fine-tunes it, promotes the resulting ONNX model, and runs prediction. Then run `just test-serve yolo_swag_waste_grid_segmentation` to build the inference image and make a live prediction request against the OAM test area.

Direct inference accepts one GeoTIFF or a directory of GeoTIFFs plus the required `confidence_threshold` parameter (default `0.5`). The optional `cell_size_m` parameter defaults to `5.0`. The result is a GeoJSON `FeatureCollection`; every emitted feature has a WGS84 polygon and the `cell_id`, `label`, and `confidence` properties described above.

## Citation

If you use this model or its pretrained weights, cite [*Open-access model for detecting openly dumped dispersed municipal solid waste from crowdsourced UAV imagery in Sub-Saharan Africa*](https://doi.org/10.48550/arXiv.2605.02316).

## Licence

SWAG is released under the [MIT License](https://spdx.org/licenses/MIT.html). The accompanying data-preparation source repository is also MIT-licensed by the GIScience Research Group and HeiGIT.
