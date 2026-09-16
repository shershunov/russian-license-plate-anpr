from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import cv2

from generator.catalog import ALPHABET

PLATE_TYPES = {"type1", "type1a", "type1b", "other"}
NUMBER = re.compile(rf"^[{ALPHABET}D0-9#]{{4,12}}$")
CONDITIONS = {"day", "night", "rain", "snow", "fog", "dirt", "glare", "motion_blur", "angle",
              "infrared", "wet", "snow_cover", "small"}
HEADER = ["image", "plate_num", "plate_type", "bbox", "quad", "is_vehicle", "is_synthetic",
          "source", "license", "conditions"]


def check(root: Path, strict: bool = False) -> dict:
    problems: list[str] = []
    meta = root / "meta.csv"
    if not meta.is_file():
        return {"ok": False, "problems": [f"missing {meta}"], "rows": 0}
    with meta.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if reader.fieldnames != HEADER:
            problems.append(f"meta.csv header {reader.fieldnames} != {HEADER}")
        rows = list(reader)
    images: Counter[str] = Counter()
    for index, row in enumerate(rows, 2):
        images[row["image"]] += 1
        path = root / row["image"]
        if not path.is_file():
            problems.append(f"line {index}: missing image {row['image']}")
            continue
        picture = cv2.imread(str(path))
        if picture is None:
            problems.append(f"line {index}: unreadable image {row['image']}")
            continue
        height, width = picture.shape[:2]
        if row["plate_type"] not in PLATE_TYPES:
            problems.append(f"line {index}: bad plate_type {row['plate_type']}")
        if not NUMBER.match(row["plate_num"]):
            problems.append(f"line {index}: bad plate_num {row['plate_num']}")
        try:
            bbox = [int(value) for value in row["bbox"].split(",")]
            quad = [float(value) for value in row["quad"].split(",")]
        except ValueError:
            problems.append(f"line {index}: malformed bbox/quad")
            continue
        if len(bbox) != 4 or len(quad) != 8:
            problems.append(f"line {index}: bbox/quad arity")
            continue
        x, y, w, h = bbox
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width + 1 or y + h > height + 1:
            problems.append(f"line {index}: bbox {bbox} outside {width}x{height}")
        if strict:
            slack_x, slack_y = width * 0.3 + 3, height * 0.3 + 3
            if not all(-slack_x <= quad[i] <= width + slack_x for i in range(0, 8, 2)):
                problems.append(f"line {index}: quad x far outside frame")
            if not all(-slack_y <= quad[i] <= height + slack_y for i in range(1, 8, 2)):
                problems.append(f"line {index}: quad y far outside frame")
        if row["is_vehicle"] not in {"0", "1"} or row["is_synthetic"] not in {"0", "1"}:
            problems.append(f"line {index}: bad flags")
        unknown = {tag for tag in row["conditions"].split(",") if tag} - CONDITIONS
        if unknown:
            problems.append(f"line {index}: unknown conditions {sorted(unknown)}")
        label = root / "labels" / f"{Path(row['image']).stem}.txt"
        if not label.is_file():
            problems.append(f"line {index}: missing label {label.name}")
    annotations = root / "annotations.jsonl"
    jsonl_rows = 0
    if annotations.is_file():
        for line in annotations.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            jsonl_rows += 1
            if len(payload["glyphs"]) != len(payload["plate_num"]):
                problems.append(f"{payload['image']}: glyph count mismatch")
    return {
        "ok": not problems,
        "rows": len(rows),
        "images": len(images),
        "annotations": jsonl_rows,
        "plate_types": dict(Counter(row["plate_type"] for row in rows)),
        "problem_count": len(problems),
        "problems": problems[:60],
    }
