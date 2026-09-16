from __future__ import annotations

import cv2
import numpy as np

from generator.config import Config
from generator.geometry import homography, transform_points
from generator.layout import Layout
from generator.materials import Surface, fractal_noise
from generator.scene import FramePlan, random_stream

AMBIENT = {
    "sun": (0.20, 0.95), "overcast": (0.55, 0.35), "shade": (0.42, 0.18),
    "dusk": (0.09, 0.22), "night": (0.02, 0.40), "ir850": (0.015, 0.90),
    "ir940": (0.012, 0.72),
}


def shade(surface: Surface, plan: FramePlan, ppm: float) -> np.ndarray:
    dy, dx = np.gradient(surface.height_mm, 1.0 / ppm)
    normals = np.stack([-dx, -dy, np.ones_like(dx)], axis=-1)
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-6)
    light = np.asarray(plan.light_direction, dtype=np.float32)
    half = light + np.array([0.0, 0.0, 1.0], np.float32)
    half /= np.linalg.norm(half)
    nl = np.maximum(normals @ light, 0.001)
    nv = np.maximum(normals[..., 2], 0.001)
    nh = np.maximum(normals @ half, 0.001)
    alpha2 = np.maximum(surface.roughness, 0.04) ** 4
    distribution = alpha2 / np.maximum(np.pi * (nh ** 2 * (alpha2 - 1) + 1) ** 2, 1e-5)
    k = (surface.roughness + 1) ** 2 / 8
    geometry = nl / (nl * (1 - k) + k) * nv / (nv * (1 - k) + k)
    f0 = 0.04 * (1 - surface.metallic[..., None]) + surface.albedo * surface.metallic[..., None]
    fresnel = f0 + (1 - f0) * (1 - max(float(light @ half), 0.0)) ** 5
    specular = distribution[..., None] * geometry[..., None] * fresnel
    specular /= np.maximum(4 * nv[..., None], 1e-5)
    ambient, direct = AMBIENT[plan.lighting]
    if plan.lighting.startswith("ir"):
        h, w = surface.nir.shape
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        falloff = np.exp(-(((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2) * 1.1))
        retro = 1.0 if surface.parameters["retroreflective"] else 0.35
        response = surface.nir * (ambient + direct * falloff * (1.1 + 1.4 * retro))
        return np.repeat(response[..., None], 3, axis=2).astype(np.float32)
    image = surface.albedo * ambient + surface.albedo * (1 - surface.metallic[..., None]) * nl[..., None] * direct
    image += specular * direct
    if surface.parameters["retroreflective"]:
        image += surface.albedo * np.clip(nv - 0.55, 0, 1)[..., None] * 0.35 * direct
    if plan.lighting == "night":
        h, w = surface.roughness.shape
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        lamps = np.exp(-((xx / w - 0.25) ** 2 / 0.12 + (yy / h - 0.1) ** 2 / 0.9))
        lamps += 0.8 * np.exp(-((xx / w - 0.8) ** 2 / 0.12 + (yy / h - 0.1) ** 2 / 0.9))
        image += surface.albedo * lamps[..., None] * np.array([0.5, 0.35, 0.2], np.float32)
    return image.astype(np.float32)


def _context(canvas: np.ndarray, plan: FramePlan, matrix: np.ndarray, shape: tuple[int, int],
             rng: np.random.Generator, scale: int) -> None:
    h, w = shape
    base = np.array(plan.background_color, np.float32)
    if plan.is_vehicle:
        panel = base * float(rng.uniform(0.35, 1.5))
        body = transform_points(np.array([[-w * 1.4, -h * 2.6], [w * 2.4, -h * 2.6],
                                          [w * 2.4, h * 3.4], [-w * 1.4, h * 3.4]], np.float32),
                                matrix)
        cv2.fillConvexPoly(canvas, np.round(body).astype(np.int32),
                           tuple(float(v) for v in panel))
        grille_top, grille_bottom = h * 1.18, h * 2.4
        grille = transform_points(np.array([[-w * 0.5, grille_top], [w * 1.5, grille_top],
                                            [w * 1.5, grille_bottom], [-w * 0.5, grille_bottom]],
                                           np.float32), matrix)
        cv2.fillConvexPoly(canvas, np.round(grille).astype(np.int32), (0.012, 0.013, 0.014))
        for position in np.linspace(-w * 0.5, w * 1.5, 26):
            bar = transform_points(np.array([[position, grille_top], [position, grille_bottom]],
                                            np.float32), matrix)
            cv2.line(canvas, tuple(np.round(bar[0]).astype(int)),
                     tuple(np.round(bar[1]).astype(int)), (0.05, 0.052, 0.055), max(1, 3 * scale))
        seam = transform_points(np.array([[-w * 1.4, -h * 0.75], [w * 2.4, -h * 0.75]],
                                         np.float32), matrix)
        cv2.line(canvas, tuple(np.round(seam[0]).astype(int)), tuple(np.round(seam[1]).astype(int)),
                 (0.006, 0.006, 0.007), max(1, 2 * scale))
    else:
        board = transform_points(np.array([[-w * 1.4, -h * 2.2], [w * 2.4, -h * 2.2],
                                           [w * 2.4, h * 3.0], [-w * 1.4, h * 3.0]], np.float32),
                                 matrix)
        cv2.fillConvexPoly(canvas, np.round(board).astype(np.int32),
                           tuple(float(v) for v in base * float(rng.uniform(0.5, 2.0))))
    if plan.holder:
        margin_x, margin_y = w * 0.035, h * 0.09
        outer = transform_points(np.array([[-margin_x, -margin_y], [w + margin_x, -margin_y],
                                           [w + margin_x, h + margin_y], [-margin_x, h + margin_y]],
                                          np.float32), matrix)
        cv2.fillConvexPoly(canvas, np.round(outer).astype(np.int32), (0.02, 0.02, 0.022))


def render_frame(plan: FramePlan, layout: Layout, surface: Surface, config: Config
                 ) -> tuple[np.ndarray, np.ndarray, dict]:
    scale = config.render.supersampling
    render_width, render_height = plan.render_size
    width, height = render_width * scale, render_height * scale
    rng = random_stream(plan.seed, plan.index, 501)
    noise = fractal_noise((height, width), rng)
    base = np.array(plan.background_color, np.float32)
    canvas = (base[None, None] * (0.4 + 0.9 * noise[..., None])).astype(np.float32)
    gradient = np.linspace(1.15, 0.7, height, dtype=np.float32)[:, None, None]
    canvas *= gradient
    if plan.lighting in {"night", "ir850", "ir940"}:
        canvas *= 0.12
    elif plan.lighting == "dusk":
        canvas *= 0.4

    quad = np.array(plan.quad, np.float32) * scale
    h, w = layout.ink.shape
    matrix = homography((w, h), quad)
    _context(canvas, plan, matrix, (h, w), rng, scale)

    alpha = cv2.warpPerspective(surface.alpha, matrix, (width, height), flags=cv2.INTER_LINEAR)
    shadow = cv2.GaussianBlur(alpha, (0, 0), max(1.0, 2.5 * scale))
    shift = np.float32([[1, 0, 2.5 * scale], [0, 1, 3.5 * scale]])
    shadow = cv2.warpAffine(shadow, shift, (width, height))
    canvas *= 1 - shadow[..., None] * 0.6

    colour = shade(surface, plan, layout.pixels_per_mm)
    warped = cv2.warpPerspective(colour * surface.alpha[..., None], matrix, (width, height),
                                 flags=cv2.INTER_LINEAR)
    canvas = canvas * (1 - alpha[..., None]) + warped

    ids = cv2.warpPerspective(layout.char_ids, matrix, (width, height), flags=cv2.INTER_NEAREST)
    characters = np.where(alpha > 0.5, ids, 0).astype(np.uint16)
    if scale > 1:
        canvas = cv2.resize(canvas, (render_width, render_height),
                            interpolation=cv2.INTER_AREA)
        characters = cv2.resize(characters, (render_width, render_height),
                                interpolation=cv2.INTER_NEAREST)
    return canvas, characters, {"backend": "cpu"}
