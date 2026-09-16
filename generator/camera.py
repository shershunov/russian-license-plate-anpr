from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from generator.config import CameraConfig
from generator.scene import FramePlan

DISTORTION = {"cctv": 0.030, "dashcam": 0.135, "phone": 0.045, "telephoto": 0.008,
              "analog": 0.038}


@dataclass(frozen=True)
class Optics:
    sensor_width: int
    sensor_height: int
    origin_x: float
    origin_y: float
    k1: float
    k2: float
    rolling: float

    @property
    def scale(self) -> float:
        return max(self.sensor_width, self.sensor_height) / 2

    @property
    def centre(self) -> np.ndarray:
        return np.array([self.sensor_width / 2 - self.origin_x,
                         self.sensor_height / 2 - self.origin_y], dtype=np.float32)

    def forward(self, points: np.ndarray) -> np.ndarray:
        normalized = (np.asarray(points, np.float32) - self.centre) / self.scale
        radius = np.sum(normalized ** 2, axis=-1, keepdims=True)
        warped = normalized * (1 + self.k1 * radius + self.k2 * radius ** 2)
        warped[..., 0] += self.rolling * warped[..., 1]
        return warped * self.scale + self.centre

    def inverse_maps(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        yy, xx = np.mgrid[:height, :width].astype(np.float32)
        target = np.stack([(xx - self.centre[0]) / self.scale,
                           (yy - self.centre[1]) / self.scale], -1)
        target[..., 0] -= self.rolling * target[..., 1]
        source = target.copy()
        for _ in range(10):
            radius = np.sum(source ** 2, axis=-1, keepdims=True)
            source = target / (1 + self.k1 * radius + self.k2 * radius ** 2)
        return ((source[..., 0] * self.scale + self.centre[0]).astype(np.float32),
                (source[..., 1] * self.scale + self.centre[1]).astype(np.float32))


def make_optics(plan: FramePlan, config: CameraConfig, rng: np.random.Generator) -> Optics:
    amplitude = DISTORTION[plan.camera_profile] * config.distortion
    k1 = float(rng.uniform(-amplitude, amplitude * 0.3))
    k2 = float(abs(k1) * rng.uniform(0.08, 0.32))
    rolling = float(rng.normal(0, 0.010) * config.rolling_shutter * plan.difficulty)
    return Optics(plan.sensor[0], plan.sensor[1], plan.window[0], plan.window[1], k1, k2, rolling)


def srgb_encode(linear: np.ndarray) -> np.ndarray:
    linear = np.maximum(linear, 0.0)
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)


def motion_kernel(length: int, angle: float) -> np.ndarray:
    length = max(1, length | 1)
    kernel = np.zeros((length, length), np.float32)
    centre = (length - 1) / 2
    delta = np.array([np.cos(angle), np.sin(angle)]) * centre
    cv2.line(kernel, tuple(np.round(centre - delta).astype(int)),
             tuple(np.round(centre + delta).astype(int)), 1.0, 1, cv2.LINE_AA)
    return kernel / max(float(kernel.sum()), 1.0)


def _plate_region(image: np.ndarray, characters: np.ndarray) -> float:
    ys, xs = np.nonzero(characters)
    luma = image @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    if len(xs) < 8:
        height, width = luma.shape
        patch = luma[height // 4:height * 3 // 4, width // 4:width * 3 // 4]
        return float(np.percentile(patch, 75)) if patch.size else float(luma.mean())
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    patch = luma[y0:y1, x0:x1]
    return float(np.percentile(patch, 80)) if patch.size else float(luma.mean())


def capture(image: np.ndarray, characters: np.ndarray, plan: FramePlan, config: CameraConfig,
            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, Optics, dict]:
    optics = make_optics(plan, config, rng)
    budget = float(rng.uniform(0.45, 1.0)) if plan.difficulty > 0.5 else 1.0
    height, width = image.shape[:2]
    mx, my = optics.inverse_maps(width, height)
    image = cv2.remap(image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    characters = cv2.remap(characters, mx, my, cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    full_x = (xx + plan.window[0] - plan.sensor[0] / 2) / (plan.sensor[0] / 2)
    full_y = (yy + plan.window[1] - plan.sensor[1] / 2) / (plan.sensor[1] / 2)
    radius = np.clip(full_x ** 2 + full_y ** 2, 0, 4)
    image = image * np.clip(1 - radius[..., None] * rng.uniform(0.05, 0.26), 0.2, 1.0)

    metered = _plate_region(image, characters)
    target = float(rng.uniform(0.30, 0.55))
    gain = float(np.clip(target / max(metered, 1e-4), 0.08, 24.0))
    exposure = float(rng.uniform(*config.exposure_ev))
    image = image * gain * 2 ** exposure
    infrared = plan.lighting.startswith("ir")
    lowlight = plan.lighting in {"dusk", "night", "ir850", "ir940"}
    if infrared:
        image = np.repeat(image.mean(axis=2, keepdims=True), 3, axis=2)
    else:
        image = image * np.array([rng.uniform(0.85, 1.15), 1.0, rng.uniform(0.83, 1.2)], np.float32)

    if plan.weather == "fog":
        fog = float(rng.uniform(0.05, 0.30)) * (0.45 + 0.55 * plan.difficulty) * budget
        image = image * (1 - fog) + fog * (0.28 if lowlight else 0.7)
    if plan.weather in {"rain", "snow"}:
        particles = np.zeros_like(image)
        count = int(rng.integers(6, 40) * max(1.0, width * height / 60000))
        for _ in range(count):
            x, y = int(rng.integers(width)), int(rng.integers(height))
            if plan.weather == "snow":
                cv2.circle(particles, (x, y), int(rng.integers(1, max(2, width // 90))),
                           (0.3,) * 3, -1)
            else:
                cv2.line(particles, (x, y), (x + int(rng.integers(-2, 3)),
                                             y + int(rng.integers(4, 18))), (0.07,) * 3, 1)
        image = image + cv2.GaussianBlur(particles, (0, 0), 0.8)

    highlights = np.maximum(image - (0.7 if infrared else 0.88), 0)
    image = image + cv2.GaussianBlur(highlights, (0, 0), max(1.0, width / 90)) * \
            config.bloom * (0.55 if lowlight else 0.22)

    defocus = float(rng.uniform(0.0, 1.15 if lowlight else 0.8) * plan.difficulty * budget)
    defocus *= max(0.35, min(2.0, plan.plate_px / 160))
    if defocus > 0.15:
        image = cv2.GaussianBlur(image, (0, 0), defocus)
    smear = int(rng.uniform(0, 7 if lowlight else 4) * plan.difficulty * config.motion_blur
                * budget * max(0.3, min(2.2, plan.plate_px / 160)))
    if smear > 1:
        image = cv2.filter2D(image, -1, motion_kernel(smear, float(rng.uniform(-np.pi, np.pi))))
    if plan.camera_profile == "analog":
        small = cv2.resize(image, (max(16, width // 2), max(16, height // 2)),
                           interpolation=cv2.INTER_AREA)
        image = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
        image[::2] *= 0.93

    if config.sensor_noise > 0:
        well = (1500 if lowlight else 12000) / config.sensor_noise
        image = rng.poisson(np.clip(image, 0, 8) * well).astype(np.float32) / well
        read = (0.005 if lowlight else 0.0009) * config.sensor_noise
        image = image + rng.normal(0, read, image.shape).astype(np.float32)
        if lowlight:
            image = image + rng.normal(0, read * 0.4, (height, 1, 1)).astype(np.float32)
    if not infrared and config.chromatic_aberration > 0:
        shift = float(rng.uniform(0.0, 0.7) * config.chromatic_aberration)
        image[..., 0] = cv2.remap(image[..., 0], xx + shift, yy, cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)
        image[..., 2] = cv2.remap(image[..., 2], xx - shift, yy, cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)
    if infrared:
        image = np.repeat(image.mean(axis=2, keepdims=True), 3, axis=2)

    margin = plan.margin
    image = image[margin:margin + plan.height, margin:margin + plan.width]
    characters = characters[margin:margin + plan.height, margin:margin + plan.width]
    result = np.uint8(np.clip(srgb_encode(np.clip(image, 0, 1)) * 255 + 0.5, 0, 255))
    quality = 0
    if rng.random() < config.jpeg_probability:
        quality = int(rng.integers(config.jpeg_quality[0], config.jpeg_quality[1] + 1))
        success, encoded = cv2.imencode(".jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR),
                                        [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not success:
            raise RuntimeError("JPEG camera simulation failed")
        result = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    parameters = {"optics": asdict(optics), "margin": margin,
                  "auto_gain": round(gain, 3), "metered": round(metered, 4),
                  "exposure_ev": round(exposure, 3),
                  "defocus_sigma": round(defocus, 3), "motion_pixels": smear,
                  "jpeg_quality": quality, "infrared": infrared}
    return result, characters, optics, parameters
