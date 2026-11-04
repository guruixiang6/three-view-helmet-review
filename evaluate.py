"""Evaluate one checkpoint on the fixed helmet compliance test split."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from prepare_data import DATA, inventory
from local_runtime import configure

CLASSES = ("ok", "violation")
DECISIONS = (*CLASSES, "uncertain")


def ground_truth(path):
    ids = {int(line.split()[0]) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()}
    return "violation" if 1 in ids else "ok" if 0 in ids else "excluded"


def safe_div(n, d):
    return n / d if d else None


def mean(values):
    return sum(v or 0 for v in values) / len(values)


def metrics(rows, prediction):
    matrix = Counter((r["truth"], r[prediction]) for r in rows if r["truth"] in CLASSES)
    for (_, pred) in matrix:
        if pred not in DECISIONS:
            raise ValueError(f"Unexpected prediction: {pred}")
    per_class = {}
    for cls in CLASSES:
        correct = matrix[cls, cls]
        predicted = sum(matrix[t, cls] for t in CLASSES)
        actual = sum(matrix[cls, p] for p in DECISIONS)
        per_class[cls] = dict(support=actual, precision=safe_div(correct, predicted),
                              recall=safe_div(correct, actual), f1=safe_div(2 * correct, actual + predicted))
    tp, tn = matrix["violation", "violation"], matrix["ok", "ok"]
    fp, fn = matrix["ok", "violation"], matrix["violation", "ok"]
    u_ok, u_violation = matrix["ok", "uncertain"], matrix["violation", "uncertain"]
    n = sum(matrix.values())
    return dict(total=len(rows), evaluated=n, excluded=len(rows) - n, tp=tp, tn=tn, fp=fp, fn=fn,
                uncertain=u_ok + u_violation, uncertain_on_ok=u_ok, uncertain_on_violation=u_violation,
                accuracy=safe_div(tp + tn, n), coverage=safe_div(tp + tn + fp + fn, n),
                precision_macro=mean([v["precision"] for v in per_class.values()]),
                recall_macro=mean([v["recall"] for v in per_class.values()]),
                f1_macro=mean([v["f1"] for v in per_class.values()]), per_class=per_class)


def apply_reviews(rows, reviews):
    candidates = {r["image"]: r for r in rows if r["truth"] in CLASSES and r["prediction"] == "uncertain"}
    seen = set()
    for review in reviews:
        image, decision = review["image"], review["decision"]
        if image in seen or image not in candidates or decision not in DECISIONS:
            raise ValueError(f"Invalid, duplicate or out-of-scope review: {image}")
        seen.add(image)
        candidates[image]["final_prediction"] = decision
    return len(seen)


def infer(args, rows, images, reviewer=None):
    configure()
    if args.method == "single":
        from ultralytics import YOLO

        model = YOLO(str(args.model.resolve()))
        by_name = {r["image"]: r for r in rows}
        seen = set()
        for offset in range(0, len(images), args.batch):
            batch = [str(p) for p in images[offset:offset + args.batch]]
            results = model.predict(
                batch, imgsz=640, conf=0.25, iou=0.7, rect=False,
                device=args.device, stream=True, save=False, verbose=False,
            )
            for result in results:
                name = Path(result.path).name
                if name in seen or name not in by_name:
                    raise RuntimeError(f"Unexpected inference output: {name}")
                seen.add(name)
                ids = set(result.boxes.cls.tolist())
                by_name[name]["prediction"] = (
                    "violation" if 1 in ids else "ok" if 0 in ids or args.force_binary else "uncertain"
                )
        if seen != by_name.keys():
            raise RuntimeError("The detector did not return a prediction for every image")
    else:
        from helmet_pipeline import AnJianDetection, load_image

        detector = AnJianDetection(model_path=str(args.model), gpu_device=args.device,
                                   debug_logging=False, save_aux_images=False)
        for row, image in zip(rows, images):
            if row["truth"] == "excluded":
                row["prediction"] = "excluded"
                continue
            result = detector.detect("00001001", load_image(image), reviewer=reviewer)
            final = "ok" if result["decision"] == "no_violation" else result["decision"]
            row["prediction"] = result.get("small_model_decision", final)
            row["final_prediction"] = final
            row["review_status"] = result.get("review_status", "")
            row["review_view"] = result.get("review_view", "")
            row["route_reason"] = result["uncertainty_info"].get("route_reason", "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Locally trained best.pt.")
    parser.add_argument("--method", required=True, choices=("single", "three-view"))
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16, help="Single-view inference batch size.")
    parser.add_argument("--force-binary", action="store_true",
                        help="Single-view only: predict ok whenever no NoHelmet box is detected.")
    parser.add_argument("--out", type=Path, required=True, help="New output directory.")
    review_options = parser.add_mutually_exclusive_group()
    review_options.add_argument("--reviews", type=Path, help="Optional local CSV: image,decision.")
    review_options.add_argument("--vlm-review", action="store_true",
                                help="Automatically review only uncertain three-view results.")
    parser.add_argument("--limit", type=int, help="Smoke test only: evaluate the first N images.")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"Missing checkpoint: {args.model}")
    if args.out.exists():
        parser.error(f"Refusing to overwrite {args.out}")
    if args.batch < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--batch and --limit must be positive")
    if args.force_binary and args.method != "single":
        parser.error("--force-binary is only supported with --method single")
    reviewer = None
    if args.vlm_review:
        if args.method != "three-view":
            parser.error("--vlm-review requires --method three-view")
        from vlm_review import VisionReviewer

        try:
            reviewer = VisionReviewer.from_environment()
        except ValueError as exc:
            parser.error(str(exc))
    images, labels = inventory(args.data, "test")
    paths = sorted(images.values())[:args.limit]
    rows = [dict(image=p.name, truth=ground_truth(labels[p.stem]), route_reason="") for p in paths]
    infer(args, rows, paths, reviewer=reviewer)
    for row in rows:
        row.setdefault("final_prediction", row["prediction"])
    report = dict(method=args.method, force_binary=args.force_binary,
                  checkpoint=str(args.model), smoke_test=args.limit is not None,
                  before_review=metrics(rows, "prediction"))
    if reviewer is not None:
        counts = Counter(r["review_status"] for r in rows if r.get("review_status"))
        report["review_mode"] = "vlm"
        report["reviewed_count"] = sum(counts.values())
        report["review_status_counts"] = dict(counts)
        report["after_review"] = metrics(rows, "final_prediction")
    elif args.reviews:
        with args.reviews.open(encoding="utf-8-sig", newline="") as f:
            report["reviewed_count"] = apply_reviews(rows, csv.DictReader(f))
        report["review_mode"] = "offline_csv"
        report["after_review"] = metrics(rows, "final_prediction")
    args.out.mkdir(parents=True)
    with (args.out / "per_image.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("image", "truth", "prediction", "final_prediction",
                                               "route_reason", "review_status", "review_view"))
        writer.writeheader()
        writer.writerows(rows)
    with (args.out / "review_queue.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("image", "decision"))
        writer.writerows((r["image"], "uncertain") for r in rows
                         if r["truth"] in CLASSES and r["prediction"] == "uncertain")
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
