from __future__ import annotations

import cv2
import numpy as np


def rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    y, p, r = np.radians([yaw, pitch, roll])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return rz @ ry @ rx


def homography(size: tuple[int, int], quad: np.ndarray) -> np.ndarray:
    w, h = size
    source = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    return cv2.getPerspectiveTransform(source, quad.astype(np.float32))


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(points, dtype=np.float32).reshape(1, -1, 2), matrix)[0]


def project_plate(width_mm: float, height_mm: float, angles: tuple[float, float, float],
                  pixel_width: float, center: tuple[float, float], focal: float,
                  principal: tuple[float, float] | None = None) -> np.ndarray:
    points = np.array([[-width_mm / 2, -height_mm / 2, 0], [width_mm / 2, -height_mm / 2, 0],
                       [width_mm / 2, height_mm / 2, 0], [-width_mm / 2, height_mm / 2, 0]])
    rotated = points @ rotation_matrix(*angles).T
    depth = focal * width_mm / pixel_width
    principal = center if principal is None else principal
    translation = (np.asarray(center) - principal) * depth / focal
    projected = focal * (rotated[:, :2] + translation) / (rotated[:, 2:3] + depth)
    return (projected + np.asarray(principal)).astype(np.float32)


def perimeter(width: int, height: int, samples: int = 32) -> np.ndarray:
    x = np.linspace(0, width - 1, samples)
    y = np.linspace(0, height - 1, samples)
    return np.concatenate((np.stack([x, np.zeros_like(x)], 1),
                           np.stack([np.full_like(y, width - 1), y], 1),
                           np.stack([x[::-1], np.full_like(x, height - 1)], 1),
                           np.stack([np.zeros_like(y), y[::-1]], 1))).astype(np.float32)


def bbox_from_mask(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1)]
