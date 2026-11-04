"""Package, restore and verify the frozen PPE2286 experiment split."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "ppe2286"
MANIFEST = ROOT / "split_manifest.json"
COUNTS = {"train": 1828, "valid": 231, "test": 227}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def inventory(root, split):
    images = {p.stem: p for p in (root / split / "images").glob("*.jpg")}
    labels = {p.stem: p for p in (root / split / "labels").glob("*.txt")}
    if not images or images.keys() != labels.keys():
        raise ValueError(f"Missing image/label pairs in {root / split}")
    return images, labels


def freeze(original, processed, archive):
    original_index = {}
    for split in ("train", "valid"):
        images, labels = inventory(original, split)
        for stem, path in images.items():
            if stem in original_index:
                raise ValueError(f"Duplicate source image: {stem}")
            original_index[stem] = (split, path, labels[stem])
    rows = []
    seen = set()
    for split, count in COUNTS.items():
        images, labels = inventory(processed, split)
        if len(images) != count:
            raise ValueError(f"Unexpected {split} count: {len(images)}")
        for stem, image in sorted(images.items()):
            source_split, source_image, source_label = original_index[stem]
            if stem in seen or source_image.name != image.name:
                raise ValueError(f"Unexpected split member: {image.name}")
            seen.add(stem)
            image_bytes, label_bytes = image.read_bytes(), labels[stem].read_bytes()
            if image_bytes != source_image.read_bytes():
                raise ValueError(f"Image differs from original: {image.name}")
            row = dict(stem=stem, split=split, source_split=source_split,
                       image_sha256=sha256(image_bytes), label_sha256=sha256(label_bytes))
            if label_bytes != source_label.read_bytes():
                row["label_override"] = label_bytes.decode("utf-8")
            rows.append(row)
    if seen != original_index.keys():
        raise ValueError("Split does not cover the original dataset")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive, "x", compression=ZIP_DEFLATED, compresslevel=3) as z:
        for rel, _ in entries(rows):
            z.write(processed / rel, rel)
        z.write(ROOT / "DATASET.md", "DATASET.md")
    with MANIFEST.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(rows, f, indent=2, ensure_ascii=True)
        f.write("\n")
    print(f"Packaged {len(rows)} pairs; {sum('label_override' in r for r in rows)} changed label files.")


def entries(rows):
    seen = set()
    for row in rows:
        stem, split = row["stem"], row["split"]
        if split not in COUNTS or not re.fullmatch(r"[A-Za-z0-9_.-]+", stem) or stem in seen:
            raise ValueError(f"Invalid or duplicate manifest entry: {stem}")
        seen.add(stem)
        yield f"{split}/images/{stem}.jpg", row["image_sha256"]
        yield f"{split}/labels/{stem}.txt", row["label_sha256"]


def require_hash(content, digest, rel):
    if sha256(content) != digest:
        raise ValueError(f"SHA-256 mismatch: {rel}")


def verify(root, rows):
    for split, count in COUNTS.items():
        images, _ = inventory(root, split)
        if len(images) != count:
            raise ValueError(f"Unexpected {split} count: {len(images)}")
    for rel, digest in entries(rows):
        require_hash((root / rel).read_bytes(), digest, rel)
    print(f"Verified {len(rows)} pairs: {COUNTS}")


def restore(root, rows, archive=None, original=None):
    root.mkdir(parents=True, exist_ok=False)
    if archive:
        with ZipFile(archive) as z:
            expected = {rel for rel, _ in entries(rows)} | {"DATASET.md"}
            names = z.namelist()
            if len(names) != len(expected) or set(names) != expected:
                raise ValueError("Archive members do not match the frozen split")
            for rel, digest in entries(rows):
                content = z.read(rel)
                require_hash(content, digest, rel)
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
    else:
        for row in rows:
            source = original / row["source_split"]
            stem = row["stem"]
            for folder, suffix, digest in (("images", ".jpg", "image_sha256"),
                                            ("labels", ".txt", "label_sha256")):
                rel = f"{folder}/{stem}{suffix}"
                if folder == "labels" and "label_override" in row:
                    content = row["label_override"].encode("utf-8")
                else:
                    content = (source / rel).read_bytes()
                require_hash(content, row[digest], rel)
                path = root / row["split"] / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
    verify(root, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--archive", type=Path, help="Unpack the release ZIP into a new directory.")
    mode.add_argument("--original", type=Path, help="Rebuild from the original Roboflow export.")
    mode.add_argument("--freeze", nargs=3, type=Path, metavar=("ORIGINAL", "PROCESSED", "ZIP"),
                      help="Maintainer: create the frozen manifest and release ZIP once.")
    args = parser.parse_args()
    if args.freeze:
        if MANIFEST.exists():
            parser.error("Refusing to overwrite the frozen manifest")
        freeze(*args.freeze)
        return
    rows = json.loads(MANIFEST.read_text(encoding="utf-8"))
    list(entries(rows))
    if args.verify:
        verify(args.data, rows)
    else:
        restore(args.data, rows, archive=args.archive, original=args.original)


if __name__ == "__main__":
    main()
