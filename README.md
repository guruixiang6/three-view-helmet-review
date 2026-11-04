# PPE2286 helmet compliance reproduction

Train YOLOv8s, YOLO11s and YOLO26s on the fixed PPE2286 split, then compare single-view and three-view image-level helmet compliance. All inference runs locally. The repository contains no checkpoints, hosted review client, or API credentials.

## 1. Install

Clone this repository and enter its directory. Use Python 3.10 or newer and install a PyTorch build suitable for your GPU (or CPU) before installing the pinned Ultralytics version.

```sh
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -r requirements.txt
```

## 2. Prepare the dataset

Download `ppe2286-resplit-v1.zip` from [Release v1.0.0](../../releases/tag/v1.0.0). Then run:

```sh
python prepare_data.py --archive ppe2286-resplit-v1.zip
```

Alternatively, pass `--original /path/to/roboflow-export` to reconstruct the exact split from the original v1 YOLO export. Both paths verify SHA-256 for all 4,572 image/label files. Run `python prepare_data.py --verify` to recheck an existing copy. The archive contains the 2,286 original images and the experiment's labels; see [dataset attribution and changes](DATASET.md).

## 3. Train

```sh
python train.py --device 0
```

The script trains the three detectors sequentially from YAML, with 100 epochs, batch 16, 640-pixel input, SGD, and seed 42 by default. It uses the validation split for model selection and early stopping (patience 30). Use `--models yolo26s` to train one model, or `--device cpu --workers 0` for a CPU run. If enabling `--amp`, Ultralytics may fetch its own auxiliary AMP-check assets. Training produces local checkpoints under `runs/`.

## 4. Evaluate

```sh
python evaluate.py --model runs/yolov8s_e100_b16_640/weights/best.pt --method single --out runs/eval_v8_single
python evaluate.py --model runs/yolov8s_e100_b16_640/weights/best.pt --method three-view --out runs/eval_v8_three_view
```

Repeat for `yolo11s` and `yolo26s`, using each actual checkpoint path and a new output directory. To predict a single image with the three-view pipeline, run `python helmet_pipeline.py --model PATH_TO_BEST_PT --image PATH_TO_IMAGE --device 0`.

Each evaluation writes per-image decisions, the uncertain-image queue, and counts and scores. The fixed test split has 227 images: 117 compliant, 62 violation and 48 excluded because their labels contain neither `Helmet` nor `NoHelmet`. Violation is the positive class. Accuracy counts uncertain outcomes as incorrect over all evaluable images. Precision, recall and F1 are macro averages over compliant and violation; an uncertain outcome is a missed instance of its true class.

With `--method single`, the default strict evaluation treats images with no predicted `Helmet` or `NoHelmet` as uncertain. Add `--force-binary` for a separate two-outcome baseline: any predicted `NoHelmet` means violation, and all other images are predicted compliant, even if no helmet-related box was detected. Report the policy with the scores; absence of a detection does not establish compliance.

## Optional: offline review

The three-view pipeline retains its original uncertainty routing. To evaluate a separate review, copy `review_queue.csv`, fill its `decision` column with `ok`, `violation`, or `uncertain`, and pass the copy as `--reviews` in a new evaluation. Only uncertain predictions may be replaced. Reviewers should not see ground-truth labels. No review responses are included, so the previous cloud-reviewed results are not reproduced by the local-only commands. New training runs can also differ numerically from the historical runs.

Code: [AGPL-3.0](LICENSE). Dataset: [CC BY 4.0](DATASET.md).
