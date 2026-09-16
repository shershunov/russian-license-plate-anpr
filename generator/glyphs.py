from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ASSET = Path(__file__).parent / "assets/gost2018/glyphs.json"
SUPERSAMPLE = 4


@lru_cache(maxsize=1)
def _manifest() -> dict:
    if not ASSET.is_file():
        raise FileNotFoundError(
            f"Missing {ASSET}; run python -m generator.build_gost_glyphs gost_50577-2018.pdf")
    return json.loads(ASSET.read_text(encoding="utf-8"))


@lru_cache(maxsize=64)
def outline(name: str) -> tuple[tuple[np.ndarray, ...], float]:
    record = _manifest()["glyphs"].get(name)
    if record is None:
        raise KeyError(f"Glyph {name!r} is not part of GOST R 50577-2018")
    contours = tuple(np.asarray(contour, dtype=np.float64) for contour in record["contours"])
    return contours, float(record["aspect"])


def aspect(name: str) -> float:
    return outline(name)[1]


def contours_at(name: str, x: float, y: float, height: float) -> list[np.ndarray]:
    shapes, ratio = outline(name)
    origin = np.array([x, y], dtype=np.float64)
    scale = np.array([height, height], dtype=np.float64)
    return [shape * scale + origin for shape in shapes]


@lru_cache(maxsize=4096)
def render(name: str, height_px: int) -> np.ndarray:
    shapes, ratio = outline(name)
    height_px = max(1, int(height_px))
    width_px = max(1, int(round(height_px * ratio)))
    scale = np.array([height_px * SUPERSAMPLE, height_px * SUPERSAMPLE], dtype=np.float64)
    canvas = np.zeros((height_px * SUPERSAMPLE, width_px * SUPERSAMPLE), np.uint8)
    cv2.fillPoly(canvas, [np.round(shape * scale).astype(np.int32) for shape in shapes], 255)
    return cv2.resize(canvas, (width_px, height_px),
                      interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def alphabet() -> tuple[str, ...]:
    return tuple(key for key in _manifest()["glyphs"] if len(key) == 1)
