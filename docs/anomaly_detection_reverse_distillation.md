# Anomaly Detection

This repository now includes a reverse-distillation anomaly detector built
around a lightweight encoder-decoder path.

## Purpose

The goal is image-level anomaly detection on a fixed trash-bin ROI without
training a large segmentation model. The main path is:

1. crop a fixed ROI
2. resize to `128x128`
3. convert to grayscale and duplicate to 3 channels
4. run a frozen pretrained `mobilenet_v3_small` teacher encoder
5. reconstruct normal multi-scale features with a compact decoder
6. measure feature reconstruction error as an anomaly map
7. reduce the anomaly map to an image score with top-k mean pooling

The training set should contain only normal samples. Validation for threshold
selection is also normal-only. The test split should contain both normal and
anomalous samples.

## Dataset layout

Expected layout:

```text
data/anomaly_detection_dataset/
|-- images/
|   |-- normal/
|   `-- anomaly/
`-- labels/
    |-- train.csv
    |-- validation.csv
    `-- test.csv
```

Each CSV must contain a `file_path` column. Optional label columns that the
dataset reader understands are:

- `label_id`
- `label`
- `target`
- `class_name`
- `class`
- `is_anomaly`

If no explicit label column is present, the loader falls back to the image path
and treats paths containing `anomaly` as anomalous.

## Files

Core files:

- `configs/anomaly_detection_reverse_distillation.py`
- `src/data/anomaly_dataset.py`
- `src/models/anomaly_reverse_distillation.py`
- `src/trainers/anomaly_trainer.py`
- `src/testers/anomaly_evaluator.py`
- `scripts/train_anomaly_detector.py`
- `scripts/eval_anomaly_detector.py`

## Training

Run training:

```bash
uv run -m scripts.train_anomaly_detector --config anomaly_detection_reverse_distillation --exp_name anomaly_rd_lite_v1
```

Artifacts are saved to:

```text
experiments/anomaly_rd_lite_v1/
|-- config.json
|-- weights/
`-- artifacts/
```

Important outputs:

- `weights/best_model.pth`
- `artifacts/threshold.json`
- `artifacts/loss_history.png`
- `artifacts/val_score_histogram.png`

## Evaluation

Run evaluation on the test split:

```bash
uv run -m scripts.eval_anomaly_detector --config anomaly_detection_reverse_distillation --exp_name anomaly_rd_lite_v1 --split test --save_maps
```

This writes:

- image-level metrics as JSON
- score histogram
- optional anomaly map visualizations

## Configuration notes

Main config fields:

- `encoder_name`: currently `mobilenet_v3_small`
- `input_size`: default `128`
- `roi`: fixed crop used before model input
- `freeze_epochs`: number of epochs with the encoder frozen
- `feature_loss_weights`: weights for the 3 feature scales
- `pixel_loss_weight`: enables the optional reconstruction branch when `> 0`
- `feature_map_weight` and `pixel_map_weight`: combine feature and pixel maps
- `threshold_mode`: currently `val_quantile`
- `threshold_quantile`: quantile over validation-normal scores

The default setup is feature-only reverse distillation. The image reconstruction
branch is disabled unless pixel-based losses or maps are explicitly enabled.

## What to collect for the real dataset

For the first usable iteration:

- 600 to 1000 images total is enough to start
- normal training images should cover lighting and pose variability
- validation-normal should be clean and representative
- test should include real anomalies, hard negatives, and borderline but still
  normal cases

## Recommended workflow

1. collect the dataset and generate the CSV files
2. set the ROI in the config
3. train on normal-only train split
4. derive the threshold from validation-normal scores
5. evaluate on the mixed test split
6. inspect saved anomaly maps for failure modes
