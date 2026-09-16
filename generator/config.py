from __future__ import annotations

import dataclasses
import json
import math
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LIGHTING = ("sun", "overcast", "shade", "dusk", "night", "ir850", "ir940")
WEATHER = ("clear", "rain", "snow", "fog")
CAMERAS = ("cctv", "dashcam", "phone", "telephoto", "analog")


@dataclass(frozen=True)
class MaterialConfig:
    pixels_per_mm: float = 4.0
    min_pixels_per_mm: float = 1.4
    pixels_per_plate_pixel: float = 3.0
    emboss_mm: tuple[float, float] = (1.0, 2.0)
    thickness_mm: tuple[float, float] = (0.9, 1.1)
    roughness: tuple[float, float] = (0.18, 0.62)
    coat: tuple[float, float] = (0.05, 0.6)
    dirt: tuple[float, float] = (0.0, 0.55)
    scratches: tuple[float, float] = (0.0, 0.45)
    wetness: tuple[float, float] = (0.0, 0.85)
    bend_mm: tuple[float, float] = (-3.0, 3.0)
    snow: tuple[float, float] = (0.0, 0.5)
    screws: float = 0.55


@dataclass(frozen=True)
class CameraConfig:
    profiles: tuple[str, ...] = CAMERAS
    yaw_degrees: tuple[float, float] = (-58.0, 58.0)
    pitch_degrees: tuple[float, float] = (-34.0, 34.0)
    roll_degrees: tuple[float, float] = (-18.0, 18.0)
    exposure_ev: tuple[float, float] = (-1.0, 0.9)
    jpeg_quality: tuple[int, int] = (55, 98)
    jpeg_probability: float = 0.75
    distortion: float = 1.0
    motion_blur: float = 1.0
    sensor_noise: float = 1.0
    bloom: float = 1.0
    rolling_shutter: float = 1.0
    chromatic_aberration: float = 0.6


@dataclass(frozen=True)
class FrameConfig:
    plate_width_px: tuple[int, int] = (55, 517)
    padding: tuple[float, float] = (-0.045, 0.10)
    max_side: int = 1024
    lighting: tuple[str, ...] = LIGHTING
    weather: tuple[str, ...] = WEATHER
    difficulty: tuple[float, float, float] = (0.34, 0.44, 0.22)
    context: float = 0.85
    holder: float = 0.35


@dataclass(frozen=True)
class RenderConfig:
    backend: str = "cycles"
    blender: str = ""
    device: str = "CUDA"
    samples: int = 512
    supersampling: int = 2
    batch_size: int = 24
    denoise: bool = False
    timeout_seconds: int = 3600
    processes: int = 1
    gpus: int = 1


@dataclass(frozen=True)
class Config:
    seed: int = 20260910
    count: int = 5000
    workers: int = min(8, os.cpu_count() or 1)
    profile: str = "gost"
    types: tuple[str, ...] = ("all",)
    regions: tuple[str, ...] = ("catalog",)
    target_share: float = 0.6
    coverage: str = "balanced"
    format: str = "jpg"
    save_masks: bool = False
    material: MaterialConfig = field(default_factory=MaterialConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    frame: FrameConfig = field(default_factory=FrameConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    def validate(self) -> None:
        from generator.catalog import CATALOG
        from generator.regions import resolve_regions

        if not 0 <= self.seed < 2 ** 63 or self.count < 1 or self.workers < 1:
            raise ValueError("seed must be nonnegative; count and workers must be positive")
        if self.profile not in {"gost", "competition"}:
            raise ValueError("profile must be gost or competition")
        if self.coverage not in {"balanced", "random", "exhaustive"}:
            raise ValueError("coverage must be balanced, random or exhaustive")
        if self.format not in {"png", "jpg"}:
            raise ValueError("format must be png or jpg")
        if self.render.backend not in {"cpu", "cycles"}:
            raise ValueError("backend must be cpu or cycles")
        if self.render.device not in {"CPU", "CUDA", "OPTIX"}:
            raise ValueError("device must be CPU, CUDA or OPTIX")
        if self.types != ("all",) and (not self.types or set(self.types) - CATALOG.keys()):
            raise ValueError(f"Unknown plate types: {self.types}")
        if len(set(self.types)) != len(self.types):
            raise ValueError("Duplicate plate types")
        resolve_regions(self.regions, self.profile)
        if not 0.0 <= self.target_share <= 1.0:
            raise ValueError("target_share must lie in [0, 1]")
        if not set(self.frame.lighting) <= set(LIGHTING) or not self.frame.lighting:
            raise ValueError("Unknown lighting preset")
        if not set(self.frame.weather) <= set(WEATHER) or not self.frame.weather:
            raise ValueError("Unknown weather preset")
        if not self.camera.profiles or not set(self.camera.profiles) <= set(CAMERAS):
            raise ValueError("Unknown camera profile")
        if len(self.frame.difficulty) != 3 or min(self.frame.difficulty) < 0:
            raise ValueError("difficulty must contain three nonnegative weights")
        if sum(self.frame.difficulty) <= 0:
            raise ValueError("difficulty weights cannot all be zero")
        for section in (self.material, self.camera, self.frame):
            for item in dataclasses.fields(section):
                value = getattr(section, item.name)
                if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], (int, float)):
                    if not all(math.isfinite(x) for x in value) or value[0] > value[1]:
                        raise ValueError(f"Invalid interval {item.name}: {value}")
        if not 1.5 <= self.material.pixels_per_mm <= 16:
            raise ValueError("pixels_per_mm must be in [1.5, 16]")
        if not 0.5 <= self.material.min_pixels_per_mm <= self.material.pixels_per_mm:
            raise ValueError("min_pixels_per_mm must be in [0.5, pixels_per_mm]")
        if not 0.5 <= self.material.pixels_per_plate_pixel <= 12:
            raise ValueError("pixels_per_plate_pixel must be in [0.5, 12]")
        for name in ("dirt", "scratches", "wetness", "snow", "roughness", "coat"):
            low, high = getattr(self.material, name)
            if not 0 <= low <= high <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.material.emboss_mm[0] < 0 or self.material.thickness_mm[0] <= 0:
            raise ValueError("Invalid plate thickness or emboss height")
        if self.frame.plate_width_px[0] < 24 or self.frame.plate_width_px[1] > 4096:
            raise ValueError("plate_width_px must lie in [24, 4096]")
        if not -0.12 <= self.frame.padding[0] <= self.frame.padding[1] <= 1.0:
            raise ValueError("padding must lie in [-0.12, 1]")
        if self.frame.max_side < 64:
            raise ValueError("max_side must be at least 64")
        if max(abs(x) for x in self.camera.yaw_degrees + self.camera.pitch_degrees) >= 80:
            raise ValueError("Yaw and pitch must remain below 80 degrees")
        if not 1 <= self.camera.jpeg_quality[0] <= self.camera.jpeg_quality[1] <= 100:
            raise ValueError("JPEG quality must be in [1, 100]")
        if not 0 <= self.camera.jpeg_probability <= 1:
            raise ValueError("jpeg_probability must lie in [0, 1]")
        if self.render.samples < 1 or self.render.supersampling not in {1, 2, 3, 4}:
            raise ValueError("Invalid render samples or supersampling")
        if self.render.batch_size < 1 or self.render.timeout_seconds < 1:
            raise ValueError("Batch size and timeout must be positive")
        if self.render.processes < 1 or self.render.gpus < 1:
            raise ValueError("render.processes and render.gpus must be positive")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def from_dict(values: dict[str, Any]) -> Config:
    values = dict(values)
    for name, cls in (
            ("material", MaterialConfig), ("camera", CameraConfig),
            ("frame", FrameConfig), ("render", RenderConfig),
    ):
        if name in values:
            values[name] = cls(**{
                key: tuple(value) if isinstance(value, list) else value
                for key, value in values[name].items()
            })
    for key, value in values.items():
        if isinstance(value, list):
            values[key] = tuple(value)
    config = Config(**values)
    config.validate()
    return config


def load_config(path: Path | None) -> Config:
    if path is None:
        config = Config()
        config.validate()
        return config
    with path.open("rb") as handle:
        values = tomllib.load(handle) if path.suffix == ".toml" else json.load(handle)
    return from_dict(values)
