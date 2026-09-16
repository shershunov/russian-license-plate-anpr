from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from generator.catalog import CATALOG

CSV_HEADER = ("image;plate_num;plate_type;bbox;quad;is_vehicle;is_synthetic;"
              "source;license;conditions")
LICENSE = "CC BY 4.0"
SOURCE = "synthetic_generator"
CLASS_IDS = {"type1": 0, "type1a": 1, "type1b": 2, "other": 3}


@dataclass(frozen=True)
class Record:
    image: str
    plate_num: str
    plate_type: str
    bbox: tuple[int, int, int, int]
    quad: list[list[float]]
    is_vehicle: int
    conditions: tuple[str, ...]
    payload: dict

    def csv_row(self) -> str:
        bbox = ",".join(str(int(round(value))) for value in self.bbox)
        quad = ",".join(f"{value:.1f}" for point in self.quad for value in point)
        conditions = ",".join(self.conditions)
        return (f"{self.image};{self.plate_num};{self.plate_type};{bbox};{quad};"
                f"{self.is_vehicle};1;{SOURCE};{LICENSE};{conditions}")

    def yolo_row(self, width: int, height: int) -> str:
        x, y, w, h = self.bbox
        return (f"{CLASS_IDS[self.plate_type]} {(x + w / 2) / width:.6f} "
                f"{(y + h / 2) / height:.6f} {w / width:.6f} {h / height:.6f}")


def conditions_for(plan, surface_parameters: dict, camera_parameters: dict) -> tuple[str, ...]:
    tags: list[str] = []
    tags.append("night" if plan.lighting in {"dusk", "night", "ir850", "ir940"} else "day")
    if plan.lighting.startswith("ir"):
        tags.append("infrared")
    if plan.weather in {"rain", "snow", "fog"}:
        tags.append(plan.weather)
    if surface_parameters.get("dirt", 0.0) > 0.22:
        tags.append("dirt")
    if surface_parameters.get("wetness", 0.0) > 0.3:
        tags.append("wet")
    if surface_parameters.get("snow", 0.0) > 0.15:
        tags.append("snow_cover")
    if plan.lighting == "sun" or camera_parameters.get("bloom_strong"):
        tags.append("glare")
    if camera_parameters.get("motion_pixels", 0) > 1:
        tags.append("motion_blur")
    if max(abs(plan.angles[0]), abs(plan.angles[1])) > 14:
        tags.append("angle")
    if plan.plate_px < 90:
        tags.append("small")
    return tuple(dict.fromkeys(tags))


def plate_type_of(subtype: str) -> str:
    return CATALOG[subtype].plate_type


def quad_bbox(quad: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    xs = np.clip(quad[:, 0], 0, width - 1)
    ys = np.clip(quad[:, 1], 0, height - 1)
    x0, y0 = float(xs.min()), float(ys.min())
    x1, y1 = float(xs.max()), float(ys.max())
    return (int(round(x0)), int(round(y0)),
            max(1, int(round(x1 - x0))), max(1, int(round(y1 - y0))))


def visible_fraction(quad: np.ndarray, width: int, height: int) -> float:
    polygon = np.round(quad).astype(np.int32)
    canvas = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(canvas, polygon, 1)
    total = cv2.contourArea(polygon.astype(np.float32))
    return float(canvas.sum()) / max(total, 1.0)


class DatasetWriter:
    def __init__(self, root: Path, image_format: str = "jpg", save_masks: bool = False) -> None:
        self.root = root
        self.image_format = image_format
        self.save_masks = save_masks
        self.images = root / "images" / "synthetic"
        self.labels = root / "labels"
        self.masks = root / "masks"
        for folder in (self.images, self.labels, root / "images" / "real"):
            folder.mkdir(parents=True, exist_ok=True)
        if save_masks:
            self.masks.mkdir(parents=True, exist_ok=True)

    def write_image(self, name: str, image: np.ndarray, quality: int = 95) -> str:
        path = self.images / f"{name}.{self.image_format}"
        params = [cv2.IMWRITE_JPEG_QUALITY, quality] if self.image_format == "jpg" else []
        if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), params):
            raise RuntimeError(f"Failed to write {path}")
        return path.relative_to(self.root).as_posix()

    def write_mask(self, name: str, characters: np.ndarray) -> None:
        if self.save_masks:
            cv2.imwrite(str(self.masks / f"{name}.png"), characters.astype(np.uint16))

    def write_label(self, name: str, rows: list[str]) -> None:
        (self.labels / f"{name}.txt").write_text("\n".join(rows) + ("\n" if rows else ""),
                                                 encoding="utf-8")

    def finalise(self, records: list[Record], config_snapshot: dict, report: dict) -> None:
        with (self.root / "meta.csv").open("w", encoding="utf-8", newline="") as handle:
            handle.write(CSV_HEADER + "\n")
            for record in records:
                handle.write(record.csv_row() + "\n")
        with (self.root / "annotations.jsonl").open("w", encoding="utf-8", newline="") as handle:
            for record in records:
                handle.write(json.dumps(record.payload, ensure_ascii=False) + "\n")
        (self.root / "generator_config.json").write_text(
            json.dumps(config_snapshot, ensure_ascii=False, indent=1), encoding="utf-8")
        (self.root / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
