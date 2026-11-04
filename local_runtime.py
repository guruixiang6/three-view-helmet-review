"""Keep Ultralytics settings inside the reproduction checkout."""

import os
from pathlib import Path


def configure():
    path = Path(os.environ.setdefault(
        "YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / ".ultralytics")
    )).expanduser()
    path.mkdir(parents=True, exist_ok=True)
