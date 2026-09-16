from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pymupdf

LETTER_PAGE = 23
DIGIT_PAGE = 24
LETTERS = "ABEKMHOPCTYXD"
DIGITS = "0123456789"
WORDS = {
    "RUS": (0.545, 0.478, 0.645, 0.512),
    "TRANSIT": (0.542, 0.443, 0.750, 0.477),
}
UPSCALE = 6
CORNER_EPS = 0.022
STRAIGHT_TOLERANCE = 0.006
ARC_TOLERANCE = 0.018


def page_bitmap(document: pymupdf.Document, index: int) -> np.ndarray:
    reference = document[index].get_images(full=True)[0]
    payload = document.extract_image(reference[0])
    image = cv2.imdecode(np.frombuffer(payload["image"], np.uint8), cv2.IMREAD_UNCHANGED)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return (image < 128).astype(np.uint8)


def glyph_boxes(mask: np.ndarray, expected: int) -> list[tuple[int, int, int, int]]:
    height, width = mask.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if not 0.030 < h / height < 0.10 or area < 2000:
            continue
        if not 0.14 < x / width < 0.88 or not 0.13 < y / height < 0.58:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))
    boxes.sort(key=lambda box: (round(box[1] / height * 22), box[0]))
    if len(boxes) != expected:
        raise ValueError(f"Expected {expected} glyph components, found {len(boxes)}")
    return boxes


def fit_circle(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    x, y = points[:, 0], points[:, 1]
    design = np.stack([x, y, np.ones_like(x)], axis=1)
    target = x ** 2 + y ** 2
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    center = solution[:2] / 2
    radius = float(np.sqrt(max(solution[2] + center @ center, 1e-9)))
    residual = float(np.abs(np.linalg.norm(points - center, axis=1) - radius).max())
    return center, radius, residual


def arc_points(center: np.ndarray, radius: float, start: np.ndarray, end: np.ndarray,
               middle: np.ndarray, samples: int) -> np.ndarray:
    a0 = np.arctan2(*(start - center)[::-1])
    a1 = np.arctan2(*(end - center)[::-1])
    am = np.arctan2(*(middle - center)[::-1])
    span = (a1 - a0) % (2 * np.pi)
    if not ((am - a0) % (2 * np.pi)) < span:
        span -= 2 * np.pi
    angles = a0 + span * np.linspace(0, 1, samples)
    return center + radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)


def smooth_open(points: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(sigma * 3))
    weights = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    weights /= weights.sum()
    padded = np.concatenate([points[1:radius + 1][::-1], points, points[-radius - 1:-1][::-1]])
    smoothed = np.stack([np.convolve(padded[:, axis], weights, mode="valid") for axis in (0, 1)], 1)
    blend = np.clip(np.minimum(np.arange(len(points)), np.arange(len(points))[::-1]) / max(1.0, sigma * 2), 0, 1)
    return points + (smoothed - points) * blend[:, None]


def rebuild(contour: np.ndarray, size: float) -> np.ndarray:
    corners = cv2.approxPolyDP(contour.astype(np.float32).reshape(-1, 1, 2),
                               CORNER_EPS * size, True).reshape(-1, 2)
    if len(corners) < 3:
        return contour.astype(np.float64)
    total = len(contour)
    indices = [int(np.argmin(np.linalg.norm(contour - corner, axis=1))) for corner in corners]
    order = np.argsort(indices)
    indices = [indices[i] for i in order]
    pieces = []
    for position, start in enumerate(indices):
        stop = indices[(position + 1) % len(indices)]
        span = (stop - start) % total
        if span < 2:
            pieces.append(contour[start:start + 1].astype(np.float64))
            continue
        piece = contour[(start + np.arange(span + 1)) % total].astype(np.float64)
        head, tail = piece[0], piece[-1]
        chord = tail - head
        length = float(np.linalg.norm(chord))
        if length < 1e-6:
            pieces.append(piece[:-1])
            continue
        normal = np.array([-chord[1], chord[0]]) / length
        deviation = np.abs((piece - head) @ normal).max()
        if deviation <= STRAIGHT_TOLERANCE * size:
            pieces.append(np.stack([head, head + chord * 0.5]))
            continue
        center, radius, residual = fit_circle(piece)
        if residual <= ARC_TOLERANCE * size and radius < size * 3:
            samples = max(6, int(span / 3))
            pieces.append(arc_points(center, radius, head, tail, piece[span // 2], samples)[:-1])
            continue
        pieces.append(smooth_open(piece, max(2.0, span * 0.09))[:-1])
    return np.concatenate(pieces)


def vectorise(patch: np.ndarray) -> list[list[list[float]]]:
    large = cv2.resize(patch * 255, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    large = cv2.GaussianBlur(large, (0, 0), UPSCALE * 0.75)
    binary = (large > 127).astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        raise ValueError("Empty glyph")
    size = float(binary.shape[0])
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < binary.size * 0.0006:
            continue
        polygons.append(rebuild(contour.reshape(-1, 2), size))
    return polygons


def normalise(polygons: list[np.ndarray]) -> tuple[list[list[list[float]]], float]:
    points = np.concatenate(polygons)
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    span = y1 - y0
    ratio = float((x1 - x0) / span)
    shifted = [[[round(float((x - x0) / span), 5), round(float((y - y0) / span), 5)]
                for x, y in polygon] for polygon in polygons]
    return shifted, round(ratio, 5)


def build(source: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    glyphs: dict[str, dict] = {}
    with pymupdf.open(source) as document:
        for page_index, alphabet in ((LETTER_PAGE, LETTERS), (DIGIT_PAGE, DIGITS)):
            mask = page_bitmap(document, page_index)
            for char, (x, y, w, h) in zip(alphabet, glyph_boxes(mask, len(alphabet)), strict=True):
                polygons, ratio = normalise(vectorise(mask[y:y + h, x:x + w]))
                glyphs[char] = {"page": page_index + 1, "source_box": [x, y, w, h],
                                "aspect": ratio, "contours": polygons}
        mask = page_bitmap(document, LETTER_PAGE)
        height, width = mask.shape
        for name, (x0, y0, x1, y1) in WORDS.items():
            patch = mask[round(y0 * height):round(y1 * height),
                    round(x0 * width):round(x1 * width)]
            ys, xs = np.nonzero(patch)
            if len(xs) == 0:
                raise ValueError(f"Missing lettering {name}")
            patch = patch[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            polygons, ratio = normalise(vectorise(patch))
            glyphs[name] = {"page": LETTER_PAGE + 1,
                            "source_box": [round(x0 * width), round(y0 * height),
                                           patch.shape[1], patch.shape[0]],
                            "aspect": ratio, "contours": polygons}
    manifest = {
        "source": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "method": "Vectorised outlines from appendices B and V of GOST R 50577-2018; "
                  "straight runs refitted as lines, curved runs as circular arcs",
        "upscale": UPSCALE,
        "glyphs": glyphs,
    }
    path = destination / "glyphs.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")),
                    encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "assets/gost2018")
    args = parser.parse_args()
    print(build(args.source, args.output))


if __name__ == "__main__":
    main()
