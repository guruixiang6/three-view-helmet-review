"""Train the three PPE2286 detectors from their architecture definitions."""

import argparse
import gc
from pathlib import Path

from prepare_data import DATA, ROOT
from local_runtime import configure

MODELS = ("yolov8s", "yolo11s", "yolo26s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--project", type=Path, default=ROOT / "runs")
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--amp", action="store_true", help="Enable AMP and the upstream AMP check.")
    args = parser.parse_args()
    if not all((args.data / s / k).is_dir() for s in ("train", "valid", "test") for k in ("images", "labels")):
        parser.error("Prepare the dataset first with prepare_data.py.")

    configure()
    from ultralytics import YOLO, settings
    import torch
    import yaml

    settings.update({"sync": False})
    args.project.mkdir(parents=True, exist_ok=True)
    config = args.project / "ppe2286.yaml"
    config.write_text(yaml.safe_dump({
        "path": args.data.resolve().as_posix(), "train": "train/images",
        "val": "valid/images", "test": "test/images",
        "names": ["Helmet", "NoHelmet", "NoVest", "Vest"],
    }), encoding="utf-8")
    for name in args.models:
        run = f"{name}_e{args.epochs}_b{args.batch}_{args.imgsz}"
        model = YOLO(f"{name}.yaml", task="detect")
        metrics = model.train(
            data=str(config.resolve()), epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
            workers=args.workers, device=args.device, project=str(args.project.resolve()), name=run,
            pretrained=False, optimizer="SGD", lr0=0.01, lrf=0.01, momentum=0.937,
            weight_decay=0.0005, seed=42, deterministic=True, patience=args.patience,
            amp=args.amp, cache=False, plots=True,
        )
        print(f"Checkpoint: {Path(metrics.save_dir) / 'weights' / 'best.pt'}")
        del model, metrics
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
