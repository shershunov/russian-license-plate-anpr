from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from generator.catalog import CATALOG, Identity, make_identity
from generator.config import Config
from generator.geometry import project_plate
from generator.regions import resolve_regions

TARGET_TYPES = ("type1", "type1a", "type1b")
SENSORS = {
    "cctv": (1920, 1080, 1.02),
    "dashcam": (1920, 1080, 0.42),
    "phone": (2160, 1440, 0.92),
    "telephoto": (1920, 1080, 2.60),
    "analog": (720, 576, 0.78),
}


def random_stream(seed: int, index: int, stage: int = 0) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, index, stage]))


def selected_types(config: Config) -> tuple[str, ...]:
    return tuple(CATALOG) if config.types == ("all",) else config.types


def type_schedule(config: Config) -> tuple[tuple[str, ...], tuple[str, ...]]:
    chosen = selected_types(config)
    targets = tuple(name for name in chosen if name in TARGET_TYPES)
    others = tuple(name for name in chosen if name not in TARGET_TYPES)
    return targets, others


def pick_subtype(config: Config, index: int, rng: np.random.Generator) -> str:
    targets, others = type_schedule(config)
    if config.target_share is None or not targets or not others:
        pool = selected_types(config)
    else:
        pool = targets if rng.random() < config.target_share else others
    return str(rng.choice(pool)) if config.coverage == "random" else pool[index % len(pool)]


def compatible_regions(subtype: str, regions: tuple[str, ...], profile: str) -> list[str]:
    spec = CATALOG[subtype]
    allows_three = spec.allows_three_digit_region or (profile == "competition"
                                                      and subtype == "type1b")
    valid = [code for code in regions if allows_three or len(code) == 2]
    if not valid:
        raise ValueError(f"No compatible region supplied for {subtype}")
    return valid


@dataclass
class FramePlan:
    index: int
    seed: int
    identity: Identity
    width: int
    height: int
    margin: int
    plate_px: float
    focal_px: float
    angles: tuple[float, float, float]
    quad: list[list[float]]
    centre: tuple[float, float]
    sensor: tuple[int, int]
    window: tuple[float, float]
    camera_profile: str
    lighting: str
    weather: str
    difficulty: float
    depth_mm: float
    is_vehicle: bool
    holder: bool
    background_color: tuple[float, float, float]
    light_direction: tuple[float, float, float]

    @property
    def render_size(self) -> tuple[int, int]:
        return self.width + 2 * self.margin, self.height + 2 * self.margin

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["identity"] = self.identity.to_dict()
        return payload


def _plate_pixels(rng: np.random.Generator, bounds: tuple[int, int], difficulty: float) -> float:
    low, high = bounds
    shape = float(rng.beta(1.6, 2.1)) - 0.13 * (difficulty - 0.6)
    value = low * (high / low) ** float(np.clip(shape, 0.0, 1.0))
    return float(np.clip(value, low, high))


def make_plan(config: Config, index: int) -> FramePlan:
    rng = random_stream(config.seed, index)
    regions = resolve_regions(config.regions, config.profile)
    weights = np.asarray(config.frame.difficulty, dtype=float)
    difficulty = float(rng.choice([0.22, 0.58, 1.0], p=weights / weights.sum()))
    subtype = pick_subtype(config, index, rng)
    valid = compatible_regions(subtype, regions, config.profile)
    picker = random_stream(config.seed, index, 7)
    region = (valid[(index * 37 + int(picker.integers(0, 5))) % len(valid)]
              if config.coverage == "balanced" else str(picker.choice(valid)))
    identity = make_identity(subtype, region, random_stream(config.seed, index, 11), config.profile)
    spec = CATALOG[subtype]

    camera_profile = str(rng.choice(config.camera.profiles))
    sensor_w, sensor_h, factor = SENSORS[camera_profile]
    focal = sensor_w * factor
    plate_px = _plate_pixels(rng, config.frame.plate_width_px, difficulty)
    scaled_angles = tuple(
        float(rng.uniform(*bounds) * (0.30 + 0.70 * difficulty))
        for bounds in (config.camera.yaw_degrees, config.camera.pitch_degrees,
                       config.camera.roll_degrees)
    )
    lighting = str(rng.choice(config.frame.lighting))
    if lighting.startswith("ir"):
        scaled_angles = (scaled_angles[0] * 0.75, scaled_angles[1] * 0.75, scaled_angles[2])

    depth_mm = focal * spec.width_mm / plate_px
    principal = np.array([sensor_w / 2, sensor_h / 2], dtype=np.float32)
    quad = project_plate(spec.width_mm, spec.height_mm, scaled_angles, plate_px,
                         (float(principal[0]), float(principal[1])), focal,
                         (float(principal[0]), float(principal[1])))
    span = quad.max(axis=0) - quad.min(axis=0)

    low, high = config.frame.padding
    shape = float(rng.beta(1.6, 3.2))
    pad_x = low + (high - low) * shape
    pad_y = low + (high - low) * float(rng.beta(1.6, 3.2))
    width = int(round(span[0] * (1 + 2 * pad_x)))
    height = int(round(span[1] * (1 + 2 * pad_y)))
    limit = config.frame.max_side
    if max(width, height) > limit:
        shrink = limit / max(width, height)
        quad = (quad - principal) * shrink + principal
        focal *= shrink
        plate_px *= shrink
        span = quad.max(axis=0) - quad.min(axis=0)
        width = max(16, int(round(width * shrink)))
        height = max(16, int(round(height * shrink)))
    margin = int(max(10, round(0.16 * max(width, height))))
    jitter = np.array([rng.uniform(-0.32, 0.32) * (width - span[0]),
                       rng.uniform(-0.32, 0.32) * (height - span[1])])
    local = quad.min(axis=0) - np.array([(width - span[0]) / 2,
                                         (height - span[1]) / 2]) - jitter
    quad = quad - local + margin
    centre = principal.astype(float) - local + margin
    frame_x = float(rng.uniform(0.0, max(1.0, sensor_w - width))) - margin
    frame_y = float(rng.uniform(0.0, max(1.0, sensor_h - height))) - margin

    direction = rng.normal(size=3)
    direction[2] = abs(direction[2]) + rng.uniform(0.9, 2.6)
    direction /= np.linalg.norm(direction)
    ambient = float(rng.uniform(0.04, 0.3))
    return FramePlan(
        index=index, seed=config.seed, identity=identity,
        width=width, height=height, margin=margin,
        plate_px=float(span[0]), focal_px=float(focal),
        angles=scaled_angles, quad=quad.astype(float).tolist(),
        centre=(float(centre[0]), float(centre[1])),
        sensor=(sensor_w, sensor_h), window=(frame_x, frame_y),
        depth_mm=float(depth_mm),
        camera_profile=camera_profile, lighting=lighting,
        weather=str(rng.choice(config.frame.weather)), difficulty=difficulty,
        is_vehicle=bool(rng.random() < config.frame.context),
        holder=bool(rng.random() < config.frame.holder),
        background_color=(ambient, ambient * float(rng.uniform(0.85, 1.15)),
                          ambient * float(rng.uniform(0.85, 1.2))),
        light_direction=tuple(float(value) for value in direction),
    )
