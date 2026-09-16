from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import nms as nms_utils
from ultralytics.utils import ops


class PlateDetector:
    def __init__(self, weights: Path, device: torch.device, imgsz: int, conf: float, iou: float,
                 max_det: int) -> None:
        source: YOLO = YOLO(str(weights))
        self.names: Dict[int, str] = dict(source.names)
        self.model = source.model.fuse().eval().to(device)
        self.parameters: int = sum(p.numel() for p in self.model.parameters())
        self.device: torch.device = device
        self.imgsz: int = imgsz
        self.conf: float = conf
        self.iou: float = iou
        self.max_det: int = max_det
        self.letterbox: LetterBox = LetterBox((imgsz, imgsz), auto=True, stride=32)

    def _prepare(self, image: np.ndarray) -> Tensor:
        padded: np.ndarray = self.letterbox(image=image)
        chw: np.ndarray = np.ascontiguousarray(padded[..., ::-1].transpose(2, 0, 1))
        return torch.from_numpy(chw).to(self.device).float().div_(255.0).unsqueeze(0)

    @torch.no_grad()
    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        tensor: Tensor = self._prepare(image)
        produced = self.model(tensor)
        predictions: Tensor = produced[0] if isinstance(produced, (list, tuple)) else produced
        detections: Tensor = nms_utils.non_max_suppression(
            predictions, self.conf, self.iou, None, False,
            max_det=self.max_det, nc=0, end2end=False, rotated=False,
        )[0]
        if not detections.shape[0]:
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)
        boxes: Tensor = ops.scale_boxes(tuple(tensor.shape[2:]), detections[:, :4], image.shape[:2])
        return boxes.cpu().numpy(), detections[:, 4].float().cpu().numpy()

    def warmup(self, image: Optional[np.ndarray] = None) -> None:
        canvas: np.ndarray = image if image is not None else np.full(
            (self.imgsz, self.imgsz, 3), 114, dtype=np.uint8,
        )
        for _ in range(3):
            self(canvas)
