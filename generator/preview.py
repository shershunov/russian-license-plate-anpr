from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import cv2
import numpy as np

from generator.camera import srgb_encode
from generator.catalog import CATALOG, make_identity
from generator.config import MaterialConfig
from generator.layout import build_layout


def render_plate(subtype: str, region: str = "77", seed: int = 0,
                 pixels_per_mm: float = 4.0, profile: str = "gost") -> np.ndarray:
    rng = np.random.default_rng(seed)
    identity = make_identity(subtype, region, rng, profile)
    layout = build_layout(identity, MaterialConfig(pixels_per_mm=pixels_per_mm))
    picture = np.clip(srgb_encode(layout.albedo) * 255, 0, 255).astype(np.uint8)
    alpha = layout.alpha[..., None]
    return np.uint8(picture * alpha + 255 * (1 - alpha))


def atlas(destination: Path, profile: str = "gost", region: str = "77") -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns, tile_width, tile_height = 3, 720, 300
    subtypes = list(CATALOG)
    rows = (len(subtypes) + columns - 1) // columns
    canvas = np.full((rows * tile_height, columns * tile_width, 3), 26, np.uint8)
    for index, subtype in enumerate(subtypes):
        plate = render_plate(subtype, region, index, 4.0, profile)
        scale = min((tile_width - 40) / plate.shape[1], (tile_height - 70) / plate.shape[0])
        plate = cv2.resize(plate, (int(plate.shape[1] * scale), int(plate.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)
        x, y = (index % columns) * tile_width, (index // columns) * tile_height
        canvas[y + 44:y + 44 + plate.shape[0], x + 20:x + 20 + plate.shape[1]] = plate
        spec = CATALOG[subtype]
        cv2.putText(canvas, f"{subtype}  {spec.width_mm:g}x{spec.height_mm:g} mm  {spec.material}",
                    (x + 18, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1,
                    cv2.LINE_AA)
    cv2.imwrite(str(destination), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return destination


def gallery(root: Path, destination: Path | None = None, limit: int = 96) -> Path:
    destination = destination or root / "preview.jpg"
    records: dict[str, dict] = {}
    annotations = root / "annotations.jsonl"
    if annotations.is_file():
        for line in annotations.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            records[payload["image"]] = payload
    else:
        with (root / "meta.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter=";"):
                records[row["image"]] = row
    files = sorted((root / "images/synthetic").glob("*"))[:limit]
    columns, tile_width, tile_height = 6, 320, 240
    rows = max(1, (len(files) + columns - 1) // columns)
    canvas = np.full((rows * tile_height, columns * tile_width, 3), 24, np.uint8)
    entries = []
    for index, path in enumerate(files):
        relative = path.relative_to(root).as_posix()
        payload = records.get(relative, {})
        image = cv2.imread(str(path))
        if image is None:
            continue
        quad = payload.get("quad")
        if isinstance(quad, str):
            values = [float(v) for v in quad.split(",")]
            quad = [[values[i], values[i + 1]] for i in range(0, 8, 2)]
        if quad:
            cv2.polylines(image, [np.round(np.asarray(quad)).astype(np.int32)], True,
                          (70, 235, 180), 1, cv2.LINE_AA)
        scale = min((tile_width - 10) / image.shape[1], (tile_height - 32) / image.shape[0])
        image = cv2.resize(image, (max(1, int(image.shape[1] * scale)),
                                   max(1, int(image.shape[0] * scale))))
        x, y = (index % columns) * tile_width, (index // columns) * tile_height
        canvas[y + 26:y + 26 + image.shape[0], x + 5:x + 5 + image.shape[1]] = image
        label = f"{payload.get('subtype', '')} {payload.get('plate_num', '')}"
        cv2.putText(canvas, label[:34], (x + 6, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (225, 225, 225), 1, cv2.LINE_AA)
        entries.append(
            f'<figure><a href="{html.escape(relative)}">'
            f'<img src="{html.escape(relative)}"></a>'
            f'<figcaption>{html.escape(label)}</figcaption></figure>')
    cv2.imwrite(str(destination), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (root / "gallery.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Plate generator preview</title>'
        '<style>body{background:#151a20;color:#e8e8e8;font:15px system-ui;margin:22px}'
        'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:18px}'
        'figure{margin:0;background:#1e242c;padding:10px;border-radius:8px}'
        'img{width:100%;image-rendering:pixelated}figcaption{padding-top:8px;font-size:13px}'
        '</style><h1>Synthetic plates</h1><main>' + "".join(entries) + "</main>",
        encoding="utf-8")
    return destination
