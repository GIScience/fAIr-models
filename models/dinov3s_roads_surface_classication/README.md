# dinov3s-road-surface

Binary road-surface classification (**paved** vs. **unpaved**) for aerial/drone imagery, built on a frozen DINOv3-**ViT-S/16** encoder with attention pooling and a linear classifier head, trained with a multi-instance-learning (bag-of-patches) objective. Unlike a segmentation model, this model does not find roads itself: it classifies a surface type onto road geometries (OSM ways) that are supplied alongside the imagery.

This fAIr model packages [SUSformer](https://github.com/GIScience/Road-surface-classification-for-FAIR) ("Surface from UAV Scenery") by HeiGIT, fine-tuned from Meta's [DINOv3](https://arxiv.org/abs/2508.10104) ViT-S/16 backbone.

## Summary

|           |                                                                                                                 |
| --------- | --------------------------------------------------------------------------------------------------------------- |
| Task      | Road surface classification (paved / unpaved) from imagery + road geometry                                      |
| Input     | 3-band RGB imagery chips plus OSM road `LineString`s (as a `labels.geojson`) covering the same scene             |
| Output    | One prediction per road: predicted `class`/`label`, `confidence`, carrying through `osm_id` and `highway`         |
| Coverage  | Global; base model trained on a worldwide catalogue of OpenAerialMap drone/satellite scenes                       |
| Use cases | Road/route-planning decision support where surface type affects vehicle accessibility                            |
| License   | Code: MIT. Encoder weights: [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)     |

## Classes

Upstream, SUSformer labels roads against a 3-class taxonomy, but this model trains and predicts on the 2-class collapse of it:

| id | name      | meaning                                                                                                                                            |
| -- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0  | `unpaved` | No continuous pavement or segmented pavement e.g. dirt, gravel, ground               |
| 1  | `paved`   | Continuous pavement (asphalt, concrete, tarmac) **or** segmented pavement (sett, cobblestone, paving stones) — collapses the upstream `paved_continuous` and `paved_segmental` classes |

## Intended use

- A decision-support signal for road/route planning where surface type affects vehicle accessibility.
- Classifies roads that are **already mapped**; it does not detect new roads. If an area of interest isn't well mapped, that's a gap to close upstream, not something this model solves.
- Works best on OAM-style drone and high-resolution satellite imagery; low-resolution imagery is harder for it to read reliably.

## How the bagging approach works

Rather than judging a road from a single photo, the model looks at several close-up patches taken at different points along the same road segment and combines what it sees across all of them into one decision (multi-instance learning over a "bag" of patches), weighting the clearest patches more heavily via attention pooling. A single patch can be misleading — a tree or building can cover the road, or the image can be blurred — so a handful of patches per road gives a more reliable signal than any one patch alone.

In this fAIr packaging, bags are cut by `road_surface_classifier.data.materialize.materialize_dataset` per labelled road, configured via `PATCH_CONFIG_KWARGS` in `pipeline.py`:

- Patch size: 15 m along the road, rendered at 256x256 px.
- Patches spaced roughly every 20 m along the road, 1-4 patches per road.
- 5 differently-sampled bags generated per labelled road during materialization, with lateral jitter (1.5 m) and bearing jitter (10°) so the model isn't sensitive to exactly where along a road a patch is taken.
- Patches too close to junctions (7.5 m clearance) or to a road's endpoints (10% margin) are avoided; junction clearance ignores minor highway values (`footway`, `path`, `cycleway`, `steps`, etc.).

## Training data

The base model is trained on SUSformer's labelled catalogue:

- **Scale**: 131 labelled scenes, 4,649 labelled road segments, globally distributed across OpenAerialMap (mostly UAV/drone captures, with a smaller share of high-resolution satellite imagery).
- **Class balance** (3-class, as hand-labelled): `unpaved` 1,784 (38.4%), `paved_continuous` 2,600 (55.9%), `paved_segmental` 265 (5.7%). Collapsed to the 2 classes this model trains on: 1,784 (38.4%) unpaved vs. 2,865 (61.6%) paved — a real imbalance the training loss compensates for (class-weighted cross-entropy). If you fine-tune on a new region, expect to re-balance for your own label distribution rather than assuming this one transfers.

For a local fAIr fine-tune, `train_model` instead takes your own `dataset_chips` (imagery) and `dataset_labels` (a `labels.geojson` of roads, with a `label` column of 0/1/2 — 2 is folded into paved) and re-materializes bags from those.

## Reliability under degraded conditions

Upstream macro-balanced accuracy, mean ± std over 5-fold cross-validation, each fold scored on its held-out validation scenes:

| condition                        | accuracy     |
| --------------------------------- | ------------ |
| normal use (full set of patches)  | 96.5% ± 0.7% |
| only 1 patch per road              | 92.9% ± 0.8% |
| only 2 patches per road             | 94.9% ± 0.6% |
| only 4 patches per road             | 96.1% ± 0.7% |
| imagery downsampled 2x             | 96.3% ± 1.1% |
| imagery downsampled 4x             | 94.4% ± 1.9% |
| imagery downsampled 8x             | 91.8% ± 1.7% |
| imagery downsampled 16x            | 86.0% ± 2.8% |

Macro AUROC under normal use: 99.5% ± 0.3%.

Takeaway: the model is fairly robust to having fewer patches per road (even a single patch only costs ~3.6pp of accuracy) and to moderate resolution loss, but accuracy degrades noticeably once imagery is downsampled 8x or more — treat predictions with more caution on very low-resolution imagery.

## How to use

### Fine-tuning (`training_pipeline`)

1. `split_dataset` — splits labelled roads into train/val/test by spatial grid block (roads bucketed by `road_key` into `block_size_m` x `block_size_m` blocks on their UTM centroid, whole blocks assigned to a split in seeded random order; not stratified by class). Defaults: `val_ratio=0.2`, `test_ratio=0.2`, `split_seed=42`, `block_size_m=100.0`.
2. `train_model` — loads the base model (frozen encoder + attention pool, trainable linear classifier), then **gates**: if the base model already scores at or above `gate_accuracy` (default 0.9) on all labelled roads in the scene, fine-tuning is skipped and the base model is returned unchanged. Otherwise it fine-tunes the classifier head for `steps` (default 200) with AdamW at `learning_rate` (default 1e-3), mixing in `replay_bags_per_step` (default 20) bags of cached backbone features from the base model's own training scenes each step so the head doesn't drift off the original distribution, and validates every `val_every` (default 20) steps.
3. `evaluate_model` — reports class-weighted cross-entropy loss and accuracy on the held-out test roads.
4. `export_onnx` — not implemented yet (raises `NotImplementedError`).

## Hyperparameters

| Parameter               |                         Default | Meaning                                                                                                   |
| ------------------------ | -------------------------------: | ----------------------------------------------------------------------------------------------------------- |
| `epochs`                 |                                1 | Fixed at 1; fine-tuning length is actually controlled by `steps` (fAIr requires this field)                 |
| `val_ratio`              |                              0.2 | Share of labelled roads held out for validation                                                             |
| `test_ratio`             |                              0.2 | Share of labelled roads held out for test (used only by `evaluate_model`)                                   |
| `split_seed`             |                               42 | Seed for the spatial block split order                                                                      |
| `id_column`              |                         `osm_id` | Label column identifying a road; all patches of one id stay on one side of the split                        |
| `n_clusters`             |                               10 | Target number of spatially compact road-id clusters for the split                                           |
| `steps`                  |                              200 | Fine-tuning steps of the classifier head; each step sees every train road plus replay bags                  |
| `learning_rate`          |                            0.001 | AdamW learning rate for the classifier head (backbone and attention pool stay frozen)                       |
| `gate_accuracy`          |                              0.9 | Base-model accuracy on all labelled roads at/above which fine-tuning is skipped                             |
| `replay_bags_per_step`   |                               20 | Replay-bank bags mixed into every step so the head keeps working on the base training scenes                 |
| `val_every`              |                               20 | Steps between validation runs during fine-tuning                                                            |
| `encoder_uri`            | *(TODO — not yet published)*     | Fine-tuned DINOv3 ViT-S/16 encoder `state_dict` (DINOv3 License); loaded with the base head                 |
| `replay_bank_uri`        | *(TODO — not yet published)*     | Replay bank of per-patch encoder features from the base model's training scenes                             |
| `confidence_threshold`   |                              0.5 | Below this softmax confidence a road is classed `unknown`                                                   |

## Architecture

- **Encoder**: DINOv3-ViT-S/16 (`vit_small_patch16_dinov3.lvd1689m`, pretrained on LVD-1689M, ~22M params), frozen.
- **Pooling**: attention pooling over a road's bag of patch encodings, weighting clearer/more informative patches more heavily.
- **Head**: linear classifier on the pooled representation, the only trainable part during fAIr fine-tuning.
- **Native patch input**: 256x256 RGB, ~15 m of road per patch.

## License

Code (this package and the upstream [SUSformer](https://github.com/GIScience/Road-surface-classification-for-FAIR) source) is MIT licensed. The DINOv3-ViT-S/16 encoder weights are fine-tuned from Meta's [DINOv3](https://github.com/facebookresearch/dinov3), and as a derivative of the DINOv3 Materials are **not** MIT licensed — they are distributed under, and subject to, the [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md). Among other things, that license requires that any redistribution of the weights includes a copy of the license, that use complies with applicable trade-control laws and excludes the license's prohibited end uses (e.g. military/warfare, nuclear, espionage, weapons, ITAR-controlled activities), and that publications using the weights acknowledge DINOv3. Read the full DINOv3 License before using or redistributing the model weights.

## Citation

If you use this model or its weights, please cite the DINOv3 paper it is based on:

> Siméoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C.,
> Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., Massa, F.,
> Haziza, D., Wehrstedt, L., Wang, J., Darcet, T., Moutakanni, T.,
> Sentana, L., Roberts, C., Vedaldi, A., Tolan, J., Brandt, J., Couprie, C.,
> Mairal, J., Jégou, H., Labatut, P., & Bojanowski, P. (2025).
> *DINOv3*. arXiv:2508.10104. https://arxiv.org/abs/2508.10104

Source: [GIScience/Road-surface-classification-for-FAIR](https://github.com/GIScience/Road-surface-classification-for-FAIR) (SUSformer), [doi:10.5281/zenodo.23396797](https://doi.org/10.5281/zenodo.23396797).
