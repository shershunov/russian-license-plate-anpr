from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from generator import glyphs
from generator.catalog import CATALOG, PALETTES, Identity
from generator.config import MaterialConfig
from generator.plates import BORDER_WIDTH, Geometry, Segment, TextLine, geometry_for


@dataclass(frozen=True)
class Glyph:
    char: str
    index: int
    box: tuple[float, float, float, float]
    contours: tuple[np.ndarray, ...]


@dataclass
class Layout:
    albedo: np.ndarray
    ink: np.ndarray
    border: np.ndarray
    relief: np.ndarray
    alpha: np.ndarray
    char_ids: np.ndarray
    glyphs: list[Glyph]
    identity: Identity
    pixels_per_mm: float
    width_mm: float
    height_mm: float


def _grid(width_px: int, height_px: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    xs = (np.arange(width_px, dtype=np.float32) + 0.5) / scale
    ys = (np.arange(height_px, dtype=np.float32) + 0.5) / scale
    return xs[None, :], ys[:, None]


def _rounded_sdf(xs: np.ndarray, ys: np.ndarray, x0: float, y0: float, x1: float, y1: float,
                 radius: float) -> np.ndarray:
    radius = min(radius, (x1 - x0) / 2, (y1 - y0) / 2)
    dx = np.abs(xs - (x0 + x1) / 2) - ((x1 - x0) / 2 - radius)
    dy = np.abs(ys - (y0 + y1) / 2) - ((y1 - y0) / 2 - radius)
    outside = np.hypot(np.maximum(dx, 0), np.maximum(dy, 0))
    inside = np.minimum(np.maximum(dx, dy), 0)
    return outside + inside - radius


def _coverage(distance: np.ndarray, softness: float) -> np.ndarray:
    return np.clip(0.5 - distance / softness, 0.0, 1.0)


def _segment_widths(segment: Segment, text: str) -> list[float]:
    return [glyphs.aspect(char) * segment.height for char in text[segment.start:segment.stop]]


def place_line(line: TextLine, text: str) -> list[tuple[str, int, float, float, float]]:
    spans = []
    for segment in line.segments:
        widths = _segment_widths(segment, text)
        if not widths:
            continue
        spans.append((segment, widths, sum(widths) + segment.gap * (len(widths) - 1)))
    if not spans:
        return []
    available = line.x1 - line.x0
    content = sum(item[2] for item in spans)
    slack = available - content
    count = len(spans)
    if count > 1 and line.justify:
        inter = slack / (count - 1)
        cursor = line.x0
    else:
        inter = min(14.0, max(0.0, slack / max(1, count - 1))) if count > 1 else 0.0
        cursor = line.x0 + (available - content - inter * (count - 1)) / 2
    squeeze = 1.0
    if slack < 0:
        gaps = sum(segment.gap * (len(widths) - 1) for segment, widths, _ in spans)
        squeeze = max(0.0, 1.0 + slack / gaps) if gaps > 0 else 1.0
        content = sum(sum(widths) for _, widths, _ in spans) + gaps * squeeze
        cursor = line.x0 + (available - content) / 2
        inter = 0.0
    placed = []
    for position, (segment, widths, _) in enumerate(spans):
        gap = segment.gap * squeeze
        for offset, char_index in enumerate(range(segment.start, segment.start + len(widths))):
            width = widths[offset]
            placed.append((text[char_index], char_index, cursor, width, segment.height))
            cursor += width + gap
        cursor -= gap
        if position < count - 1:
            cursor += inter
    return placed


def _draw_glyph(target: np.ndarray, stencil: np.ndarray, x_px: int, y_px: int) -> None:
    h, w = stencil.shape
    height, width = target.shape
    x0, y0 = max(0, x_px), max(0, y_px)
    x1, y1 = min(width, x_px + w), min(height, y_px + h)
    if x0 >= x1 or y0 >= y1:
        return
    patch = stencil[y0 - y_px:y1 - y_px, x0 - x_px:x1 - x_px]
    np.maximum(target[y0:y1, x0:x1], patch, out=target[y0:y1, x0:x1])


def _stamp_ids(target: np.ndarray, stencil: np.ndarray, x_px: int, y_px: int, value: int) -> None:
    h, w = stencil.shape
    height, width = target.shape
    x0, y0 = max(0, x_px), max(0, y_px)
    x1, y1 = min(width, x_px + w), min(height, y_px + h)
    if x0 >= x1 or y0 >= y1:
        return
    patch = stencil[y0 - y_px:y1 - y_px, x0 - x_px:x1 - x_px]
    region = target[y0:y1, x0:x1]
    region[patch > 0.45] = value


def build_layout(identity: Identity, config: MaterialConfig) -> Layout:
    spec = CATALOG[identity.subtype]
    geometry: Geometry = geometry_for(identity.subtype, len(identity.region), identity.profile)
    scale = config.pixels_per_mm
    width_px = int(round(geometry.width * scale))
    height_px = int(round(geometry.height * scale))
    xs, ys = _grid(width_px, height_px, scale)
    softness = 1.0 / scale

    plate = _rounded_sdf(xs, ys, 0.0, 0.0, geometry.width, geometry.height, geometry.plate_radius)
    alpha = _coverage(plate, softness)

    field = geometry.field
    border = _coverage(np.abs(_rounded_sdf(xs, ys, field.x0, field.y0, field.x1, field.y1,
                                           field.radius)) - BORDER_WIDTH / 2, softness)
    inside = _coverage(_rounded_sdf(xs, ys, field.x0, field.y0, field.x1, field.y1,
                                    field.radius) + BORDER_WIDTH / 2, softness)
    for cell in geometry.cells:
        stroke = _coverage(np.abs(_rounded_sdf(xs, ys, cell.x0, cell.y0, cell.x1, cell.y1,
                                               cell.radius)) - BORDER_WIDTH / 2, softness)
        border = np.maximum(border, stroke * inside)
    for divider in geometry.dividers:
        span = np.clip(ys, divider.y0, divider.y1)
        stroke = _coverage(np.hypot(xs - divider.x, ys - span) - BORDER_WIDTH / 2, softness)
        border = np.maximum(border, stroke * inside)
    border *= alpha

    marks = np.zeros((height_px, width_px), np.float32)
    for mark in geometry.marks:
        aspect = glyphs.aspect(mark.name)
        mark_width = min(mark.x1 - mark.x0, mark.height * aspect)
        mark_height = mark_width / aspect
        stencil = glyphs.render(mark.name, int(round(mark_height * scale)))
        _draw_glyph(marks, stencil,
                    int(round((mark.x0 + (mark.x1 - mark.x0 - mark_width) / 2) * scale)),
                    int(round((mark.bottom - mark_height) * scale)))
    if geometry.flag is not None and spec.flag:
        fx, fy, fw, fh = geometry.flag
        marks[int(round(fy * scale)):int(round((fy + fh) * scale)),
        int(round(fx * scale)):int(round((fx + fw) * scale))] = 1.0
    if geometry.marks or (geometry.flag is not None and spec.flag):
        gap = max(3, int(round(1.4 * scale)) | 1)
        shielded = cv2.dilate((marks > 0.15).astype(np.uint8), np.ones((gap, gap), np.uint8))
        border = border * (1.0 - shielded.astype(np.float32))

    if geometry.flag is not None and spec.flag:
        fx, fy, fw, fh = geometry.flag
        marks[int(round(fy * scale)):int(round((fy + fh) * scale)),
        int(round(fx * scale)):int(round((fx + fw) * scale))] = 0.0
    ink = np.maximum(border, marks).astype(np.float32)
    relief = border.astype(np.float32).copy()
    char_ids = np.zeros((height_px, width_px), np.uint16)
    placed_glyphs: list[Glyph] = []
    text = identity.text

    for line in geometry.lines:
        for char, index, x_mm, w_mm, h_mm in place_line(line, text):
            stencil = glyphs.render(char, int(round(h_mm * scale)))
            x_px = int(round(x_mm * scale))
            y_px = int(round((line.bottom - h_mm) * scale))
            _draw_glyph(ink, stencil, x_px, y_px)
            _draw_glyph(relief, stencil, x_px, y_px)
            _stamp_ids(char_ids, stencil, x_px, y_px, index + 1)
            contours = glyphs.contours_at(char, x_mm, line.bottom - h_mm, h_mm)
            placed_glyphs.append(Glyph(char, index, (x_mm, line.bottom - h_mm, w_mm, h_mm),
                                       tuple(contours)))

    background, foreground = (np.array(color, np.float32) for color in PALETTES[spec.palette])
    albedo = np.broadcast_to(background, (height_px, width_px, 3)).copy()
    if geometry.yellow_from is not None:
        albedo[:, int(round(geometry.yellow_from * scale)):] = PALETTES["yellow"][0]
    albedo = albedo * (1 - ink[..., None]) + foreground * ink[..., None]

    if geometry.flag is not None and spec.flag:
        fx, fy, fw, fh = geometry.flag
        x0, y0 = int(round(fx * scale)), int(round(fy * scale))
        x1, y1 = int(round((fx + fw) * scale)), int(round((fy + fh) * scale))
        bands = ((0.86, 0.87, 0.85), (0.015, 0.075, 0.36), (0.62, 0.015, 0.035))
        for position, color in enumerate(bands):
            top = y0 + round((y1 - y0) * position / 3)
            bottom = y0 + round((y1 - y0) * (position + 1) / 3)
            albedo[top:bottom, x0:x1] = color
        edge = max(1, int(round(scale * 0.7)))
        albedo[y0 - edge:y0, x0 - edge:x1 + edge] = foreground
        albedo[y1:y1 + edge, x0 - edge:x1 + edge] = foreground
        albedo[y0 - edge:y1 + edge, x0 - edge:x0] = foreground
        albedo[y0 - edge:y1 + edge, x1:x1 + edge] = foreground

    if spec.material in {"paper", "laminate"}:
        relief[:] = 0.0

    placed_glyphs.sort(key=lambda glyph: glyph.index)
    if [glyph.index for glyph in placed_glyphs] != list(range(len(text))):
        raise ValueError(f"Incomplete character layout for {identity.subtype}: {identity.text}")
    return Layout(albedo, ink, border.astype(np.float32), relief, alpha.astype(np.float32),
                  char_ids, placed_glyphs, identity, scale, geometry.width, geometry.height)
