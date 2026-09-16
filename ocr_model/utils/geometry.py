from typing import Dict, Sequence, Tuple

import cv2
import numpy as np

PATCH_SIZE: int = 4
SIZE_STEP: int = 4
TOKEN_BUDGET: int = 4096

BUCKETS: Tuple[Tuple[int, int], ...] = (
    (68, 64), (60, 72), (56, 80), (52, 88), (48, 96),
    (44, 104), (36, 112), (36, 128), (32, 132), (32, 148), (28, 144),
)

MIN_ASPECT_RATIO: float = 0.70
MAX_ASPECT_RATIO: float = 6.50

_ORDER: np.ndarray = np.argsort([w / h for h, w in BUCKETS])
BUCKETS = tuple(BUCKETS[i] for i in _ORDER)

BUCKET_AR: np.ndarray = np.array([w / h for h, w in BUCKETS], dtype=np.float64)
BUCKET_LOG_AR: np.ndarray = np.log(BUCKET_AR)
_BUCKET_EDGES: np.ndarray = 0.5 * (BUCKET_LOG_AR[1:] + BUCKET_LOG_AR[:-1])

BUCKET_GRIDS: Tuple[Tuple[int, int], ...] = tuple((h // PATCH_SIZE, w // PATCH_SIZE) for h, w in BUCKETS)
BUCKET_TOKENS: Tuple[int, ...] = tuple(gh * gw for gh, gw in BUCKET_GRIDS)
MAX_TOKENS: int = max(BUCKET_TOKENS)
MAX_GRID_H: int = max(gh for gh, _ in BUCKET_GRIDS)
MAX_GRID_W: int = max(gw for _, gw in BUCKET_GRIDS)
NUM_BUCKETS: int = len(BUCKETS)

INTERPOLATION_MODES: Dict[str, int] = {
    'nearest': cv2.INTER_NEAREST,
    'linear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'lanczos': cv2.INTER_LANCZOS4,
    'area': cv2.INTER_AREA,
}

EVAL_INTERPOLATION: int = cv2.INTER_AREA

INTERP_NAMES: Tuple[str, ...] = ('nearest', 'linear', 'bicubic', 'lanczos', 'area')
INTERP_WEIGHTS: Tuple[float, ...] = (0.14, 0.28, 0.14, 0.14, 0.30)


def select_bucket(aspect_ratio: float) -> int:
    return int(np.searchsorted(_BUCKET_EDGES, np.log(max(aspect_ratio, 1e-3))))


def select_buckets(aspect_ratios: np.ndarray) -> np.ndarray:
    return np.searchsorted(_BUCKET_EDGES, np.log(np.maximum(aspect_ratios, 1e-3)))


def sample_window(bbox: Sequence[float], base_pad: float, jitter: float,
                  rng: np.random.Generator) -> Tuple[float, float, float, float]:
    x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    if jitter > 0.0:
        pads: np.ndarray = base_pad + rng.uniform(-jitter * 0.5, jitter, size=4)
    else:
        pads = np.full(4, base_pad, dtype=np.float64)
    x0: float = x - pads[0] * w
    y0: float = y - pads[1] * h
    x1: float = x + w + pads[2] * w
    y1: float = y + h + pads[3] * h
    return x0, y0, max(x1, x0 + 1.0), max(y1, y0 + 1.0)


def render_window(image: np.ndarray, window: Tuple[float, float, float, float],
                  bucket_idx: int, interpolation: int) -> np.ndarray:
    height, width = BUCKETS[bucket_idx]
    x0, y0, x1, y1 = window
    scale_x: float = width / (x1 - x0)
    scale_y: float = height / (y1 - y0)
    matrix: np.ndarray = np.array(
        [[scale_x, 0.0, -x0 * scale_x], [0.0, scale_y, -y0 * scale_y]], dtype=np.float64
    )
    if interpolation == cv2.INTER_AREA and (scale_x >= 1.0 or scale_y >= 1.0):
        interpolation = cv2.INTER_LINEAR
    return cv2.warpAffine(
        image, matrix, (width, height), flags=interpolation,
        borderMode=cv2.BORDER_REPLICATE,
    )


def downscale_source(image: np.ndarray, target_height: int, interpolation: int) -> np.ndarray:
    height, width = image.shape[:2]
    if target_height >= height:
        return image
    scale: float = target_height / height
    return cv2.resize(image, (max(1, int(round(width * scale))), target_height), interpolation=interpolation)


def normalize_points(points: np.ndarray, window: Tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = window
    out: np.ndarray = np.empty_like(points, dtype=np.float32)
    out[..., 0] = (points[..., 0] - x0) / (x1 - x0)
    out[..., 1] = (points[..., 1] - y0) / (y1 - y0)
    return out


def quad_to_box(quad: np.ndarray) -> np.ndarray:
    lo: np.ndarray = quad.min(axis=0)
    hi: np.ndarray = quad.max(axis=0)
    center: np.ndarray = 0.5 * (lo + hi)
    size: np.ndarray = hi - lo
    return np.array([center[0], center[1], size[0], size[1]], dtype=np.float32)


def order_quad(quad: np.ndarray) -> np.ndarray:
    center: np.ndarray = quad.mean(axis=0)
    angles: np.ndarray = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
    start: int = int(np.argmin((angles + 0.75 * np.pi) % (2.0 * np.pi)))
    order: np.ndarray = (np.arange(4) + start) % 4
    return quad[order]


def sample_interpolation(rng: np.random.Generator) -> int:
    name: str = INTERP_NAMES[int(rng.choice(len(INTERP_NAMES), p=INTERP_WEIGHTS))]
    return INTERPOLATION_MODES[name]
