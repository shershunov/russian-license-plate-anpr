from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from generator.catalog import CATALOG
from generator.config import MaterialConfig
from generator.layout import Layout


@dataclass
class Surface:
    albedo: np.ndarray
    height_mm: np.ndarray
    roughness: np.ndarray
    metallic: np.ndarray
    alpha: np.ndarray
    char_ids: np.ndarray
    obstruction: np.ndarray
    nir: np.ndarray
    parameters: dict


def fractal_noise(shape: tuple[int, int], rng: np.random.Generator,
                  octaves: int = 5) -> np.ndarray:
    h, w = shape
    result = np.zeros(shape, dtype=np.float32)
    weight = 0.0
    for octave in range(octaves):
        frequency = 2 ** (octave + 1)
        rows = max(2, frequency)
        columns = max(2, round(frequency * w / h))
        lattice = rng.random((rows, columns), dtype=np.float32)
        amplitude = 0.55 ** octave
        result += amplitude * cv2.resize(lattice, (w, h), interpolation=cv2.INTER_CUBIC)
        weight += amplitude
    return np.clip(result / weight, 0, 1)


def make_surface(layout: Layout, config: MaterialConfig, rng: np.random.Generator,
                 difficulty: float, weather: str) -> Surface:
    spec = CATALOG[layout.identity.subtype]
    h, w = layout.ink.shape
    shape = (h, w)
    ppm = layout.pixels_per_mm
    noise = fractal_noise(shape, rng)
    detail = fractal_noise(shape, rng, 7)
    albedo = layout.albedo.copy()
    emboss = float(rng.uniform(*config.emboss_mm)) if spec.material == "metal" else 0.0
    thickness = float(rng.uniform(*config.thickness_mm)) if spec.material == "metal" else (
        0.45 if spec.material == "laminate" else 0.18
    )
    height = cv2.GaussianBlur(layout.relief, (0, 0), max(0.3, ppm * 0.32)) * emboss
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    x = xx / max(1, w - 1) * 2 - 1
    y = yy / max(1, h - 1)
    bend = float(rng.uniform(*config.bend_mm))
    height += bend * x * x + (noise - 0.5) * 0.12
    roughness = np.full(shape, rng.uniform(*config.roughness), dtype=np.float32)
    roughness += (noise - 0.5) * 0.13
    roughness = roughness * (1 - layout.ink * 0.25)
    metallic = np.zeros(shape, dtype=np.float32)
    obstruction = np.zeros(shape, dtype=np.float32)
    dirt_strength = float(rng.uniform(*config.dirt) * (0.3 + 0.7 * difficulty))
    dirt = np.clip((noise + y * 0.12 - (0.93 - dirt_strength * 0.9)) * 5, 0, 1)
    splatter = np.zeros(shape, dtype=np.float32)
    for _ in range(round(dirt_strength * 260)):
        radius = max(1, round(rng.lognormal(0.5, 0.95) * ppm))
        center = (int(rng.integers(w)), int(rng.beta(2.5, 0.9) * (h - 1)))
        cv2.circle(splatter, center, radius, float(rng.uniform(0.12, 0.9)), -1, cv2.LINE_AA)
    dirt = np.maximum(dirt, cv2.GaussianBlur(splatter, (0, 0), max(0.35, ppm * 0.2)))
    dirt *= 1.0 - layout.relief * 0.55
    dirt_color = np.array(rng.choice([
        (0.095, 0.072, 0.040), (0.19, 0.13, 0.066), (0.055, 0.052, 0.048),
        (0.31, 0.27, 0.18),
    ]), dtype=np.float32)
    albedo = albedo * (1 - dirt[..., None]) + dirt_color * dirt[..., None] * (0.65 + detail[..., None])
    roughness = roughness * (1 - dirt) + 0.93 * dirt
    height += dirt * rng.uniform(0.03, 0.6)
    obstruction = np.maximum(obstruction, dirt)
    scratch_strength = float(rng.uniform(*config.scratches) * (0.4 + difficulty * 0.6))
    scratch = np.zeros(shape, dtype=np.float32)
    for _ in range(round(scratch_strength * 100)):
        px, py = int(rng.integers(w)), int(rng.integers(h))
        end = (int(px + rng.normal(0, ppm * 14)), int(py + rng.normal(0, ppm * 2)))
        cv2.line(scratch, (px, py), end, float(rng.uniform(0.1, 0.8)),
                 max(1, round(rng.uniform(0.1, 0.5) * ppm)), cv2.LINE_AA)
    if spec.material == "metal":
        worn = scratch * (0.25 + layout.ink * 0.75)
        albedo = albedo * (1 - worn[..., None]) + np.array([0.48, 0.49, 0.5]) * worn[..., None]
        metallic = np.maximum(metallic, worn)
        height -= scratch * 0.05
    snow_strength = float(rng.uniform(*config.snow)) if weather == "snow" else 0.0
    snow = np.clip((detail + y * 0.25 - (1.0 - snow_strength * 0.5)) * 7, 0, 1)
    albedo = albedo * (1 - snow[..., None]) + np.array([0.82, 0.86, 0.88]) * snow[..., None]
    height += snow * rng.uniform(1.0, 3.5)
    roughness = roughness * (1 - snow) + 0.9 * snow
    obstruction = np.maximum(obstruction, snow)
    wetness = float(rng.uniform(*config.wetness)) if weather in {"rain", "fog"} else (
            float(rng.uniform(*config.wetness)) * (0.25 if rng.random() < 0.2 else 0)
    )
    coat = float(rng.uniform(*config.coat))
    roughness *= 1 - 0.7 * wetness
    albedo *= 1 - dirt[..., None] * wetness * 0.3
    if wetness > 0.1:
        drops = np.zeros(shape, dtype=np.float32)
        for _ in range(round(wetness * 75)):
            cx, cy = float(rng.uniform(0, w)), float(rng.uniform(0, h))
            radius = float(rng.uniform(0.5, 2.2) * ppm)
            x0, x1 = max(0, int(cx - radius * 2)), min(w, int(cx + radius * 2 + 1))
            y0, y1 = max(0, int(cy - radius * 2)), min(h, int(cy + radius * 2 + 1))
            distance = ((xx[y0:y1, x0:x1] - cx) ** 2 + (yy[y0:y1, x0:x1] - cy) ** 2) / radius ** 2
            drops[y0:y1, x0:x1] += np.exp(-distance * 2) * 0.35
        height += drops
        roughness = np.where(drops > 0.04, 0.06, roughness)
    if spec.material == "metal" and rng.random() < config.screws:
        width_mm, height_mm = layout.width_mm, layout.height_mm
        if width_mm > 400:
            screw_positions = [(20.0, height_mm / 2), (width_mm - 20.0, height_mm / 2)]
        else:
            screw_positions = [(width_mm / 2, 10.5), (width_mm / 2, height_mm - 10.5)]
        for sx, sy in screw_positions:
            radius = np.sqrt((xx / ppm - sx) ** 2 + (yy / ppm - sy) ** 2)
            head = np.clip(4.2 - radius, 0, 1)
            albedo = albedo * (1 - head[..., None]) + 0.42 * head[..., None]
            metallic = np.maximum(metallic, head)
            height += head * np.clip(1 - radius / 4, 0, 1) * 1.8
            slot = ((abs(xx / ppm - sx) < 2.6) & (abs(yy / ppm - sy) < 0.4)).astype(np.float32)
            albedo *= 1 - slot[..., None] * 0.9
            height -= slot * 0.5
            obstruction = np.maximum(obstruction, head)
    fine = rng.normal(0, 0.003, albedo.shape).astype(np.float32)
    albedo = np.clip(albedo * (0.96 + 0.07 * detail[..., None]) + fine, 0.002, 0.96).astype(np.float32)
    nir_base = 0.68 if spec.palette in {"white", "yellow"} else (
        0.21 if spec.palette == "black" else 0.45
    )
    nir_ink = 0.028 if spec.palette in {"white", "yellow"} else 0.76
    nir = nir_base * (1 - layout.ink) + nir_ink * layout.ink
    nir = nir * (1 - dirt) + (0.07 + 0.08 * noise) * dirt
    nir = nir * (1 - snow) + 0.75 * snow
    parameters = {
        "emboss_mm": emboss, "thickness_mm": thickness, "bend_mm": bend,
        "dirt": dirt_strength, "scratch_strength": scratch_strength, "wetness": wetness,
        "snow": snow_strength, "coat": coat, "material": spec.material,
        "retroreflective": spec.palette != "black" and spec.material == "metal",
    }
    return Surface(albedo, height.astype(np.float32), np.clip(roughness, 0.045, 0.96),
                   metallic, layout.alpha, layout.char_ids, obstruction,
                   nir.astype(np.float32), parameters)
