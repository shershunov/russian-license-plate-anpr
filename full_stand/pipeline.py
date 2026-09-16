import base64
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor

OCR_ROOT: Path = Path(__file__).resolve().parents[1] / 'ocr_model'
if str(OCR_ROOT) not in sys.path:
    sys.path.insert(0, str(OCR_ROOT))

from nn.inference import GRAPH_BATCH_SIZES, GraphedPredictor, PlateRecognizer
from nn.model import PlateOCR
from utils.alphabet import (EOS_IDX, MAX_SEQ_LEN, PAD_IDX, UNKNOWN_SUBTYPE, idx_to_char,
                            idx_to_subtype, is_valid_plate, num_subtypes)
from utils.geometry import BUCKETS, EVAL_INTERPOLATION, render_window, sample_window, select_bucket

from full_stand.config import StandConfig
from full_stand.detection import PlateDetector

Window = Tuple[float, float, float, float]
TOP_SUBTYPES: int = 6
GRAPH_BATCH: int = GRAPH_BATCH_SIZES[-1]


def encode_png(patch: np.ndarray) -> str:
    ok, buffer = cv2.imencode('.png', cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError('png encode failed')
    return 'data:image/png;base64,' + base64.b64encode(buffer.tobytes()).decode('ascii')


def decode_image(payload: bytes, max_pixels: int) -> np.ndarray:
    buffer: np.ndarray = np.frombuffer(payload, dtype=np.uint8)
    image: Optional[np.ndarray] = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('unsupported or corrupted image')
    height, width = image.shape[:2]
    if min(height, width) < 8:
        raise ValueError('image is too small')
    if height * width > max_pixels:
        raise ValueError(f'image exceeds {max_pixels // 1_000_000} megapixels')
    return image


class PlatePipeline:
    def __init__(self, config: StandConfig) -> None:
        self.config: StandConfig = config
        requested: Optional[torch.device] = torch.device(config.device) if config.device else None
        recognizer: PlateRecognizer = PlateRecognizer.from_checkpoint(
            str(config.recognizer), device=requested, use_ema=config.use_ema,
            base_pad=config.base_pad, use_graphs=config.use_graphs, use_script=config.use_script,
        )
        self.model: PlateOCR = recognizer.model
        self.predictor: GraphedPredictor = recognizer.predictor
        self.device: torch.device = recognizer.device
        self.detector: PlateDetector = PlateDetector(
            config.detector, self.device, config.imgsz, config.confidence, config.iou,
            config.max_detections,
        )
        self.ocr_parameters: int = sum(p.numel() for p in self.model.parameters())
        self.detector_parameters: int = self.detector.parameters
        self.detector_classes: Dict[int, str] = self.detector.names

    def warmup(self) -> float:
        started: float = time.perf_counter()
        canvas: np.ndarray = np.full((640, 640, 3), 96, dtype=np.uint8)
        cv2.rectangle(canvas, (180, 280), (460, 360), (240, 240, 240), -1)
        self.detector.warmup(canvas)
        self.predictor.warmup()
        rgb: np.ndarray = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        self._read(rgb, np.array([[180.0, 280.0, 460.0, 360.0]], dtype=np.float32),
                   np.array([1.0], dtype=np.float32))
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        return (time.perf_counter() - started) * 1000.0

    def _detect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        boxes, scores = self.detector(image)
        if boxes.size == 0:
            return boxes.reshape(0, 4), scores.reshape(0)

        height, width = image.shape[:2]
        boxes[:, 0::2] = boxes[:, 0::2].clip(0.0, float(width))
        boxes[:, 1::2] = boxes[:, 1::2].clip(0.0, float(height))
        sides: np.ndarray = np.stack([boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], axis=1)
        keep: np.ndarray = sides.min(axis=1) >= self.config.min_box_side
        boxes, scores = boxes[keep], scores[keep]
        order: np.ndarray = np.argsort(-scores)[:self.config.max_detections]
        return boxes[order], scores[order]

    def _to_tensor(self, patches: List[np.ndarray]) -> Tensor:
        stacked: np.ndarray = np.ascontiguousarray(np.stack(patches))
        tensor: Tensor = torch.from_numpy(stacked).to(self.device, non_blocking=True)
        return (255.0 - tensor.permute(0, 3, 1, 2).float()) / 255.0

    def _characters(self, tokens: np.ndarray, probs: np.ndarray, top_values: np.ndarray,
                    top_indices: np.ndarray) -> List[Dict]:
        limit: int = self.config.top_alternatives
        characters: List[Dict] = []
        for position in range(MAX_SEQ_LEN):
            token: int = int(tokens[position])
            if token in (EOS_IDX, PAD_IDX):
                break
            alternatives: List[Dict] = [
                                           {'char': idx_to_char[int(index)], 'probability': float(value)}
                                           for value, index in zip(top_values[position], top_indices[position])
                                           if int(index) != token and value > 0.0
                                       ][:limit]
            characters.append({
                'position': position,
                'char': idx_to_char[token],
                'probability': float(probs[position]),
                'alternatives': alternatives,
            })
        return characters

    def _read(self, image: np.ndarray, boxes: np.ndarray, scores: np.ndarray) -> List[Dict]:
        if boxes.shape[0] == 0:
            return []

        patches: List[Optional[np.ndarray]] = [None] * boxes.shape[0]
        windows: List[Optional[Window]] = [None] * boxes.shape[0]
        groups: Dict[int, List[int]] = defaultdict(list)
        for index, box in enumerate(boxes):
            window: Window = sample_window(
                (box[0], box[1], box[2] - box[0], box[3] - box[1]), self.config.base_pad, 0.0, None,
            )
            bucket: int = select_bucket((window[2] - window[0]) / (window[3] - window[1]))
            patches[index] = render_window(image, window, bucket, EVAL_INTERPOLATION)
            windows[index] = window
            groups[bucket].append(index)

        plates: List[Optional[Dict]] = [None] * boxes.shape[0]
        alternatives: int = self.config.top_alternatives + 1
        for bucket, indices in groups.items():
            bucket_h, bucket_w = BUCKETS[bucket]
            for start in range(0, len(indices), GRAPH_BATCH):
                chunk: List[int] = indices[start:start + GRAPH_BATCH]
                output: Dict[str, Tensor] = self.predictor(self._to_tensor([patches[i] for i in chunk]))
                tokens: Tensor = output['tokens']
                probs: Tensor = output['probs']
                top = probs.topk(min(alternatives, probs.shape[-1]), dim=-1)
                chosen: np.ndarray = probs.gather(2, tokens.unsqueeze(-1)).squeeze(-1).cpu().numpy()
                top_values: np.ndarray = top.values.cpu().numpy()
                top_indices: np.ndarray = top.indices.cpu().numpy()
                token_ids: np.ndarray = tokens.cpu().numpy()
                subtypes: np.ndarray = output['subtype'].cpu().numpy()
                confidence: np.ndarray = output['confidence'].float().cpu().numpy()
                type_probs: Tensor = output['type_probs'].float().cpu()
                ranked = type_probs.topk(min(TOP_SUBTYPES, num_subtypes), dim=-1)

                for slot, index in enumerate(chunk):
                    subtype: str = idx_to_subtype[int(subtypes[slot])]
                    characters: List[Dict] = self._characters(
                        token_ids[slot], chosen[slot], top_values[slot], top_indices[slot],
                    )
                    text: str = ''.join(item['char'] for item in characters)
                    readable: bool = subtype != UNKNOWN_SUBTYPE
                    if not readable:
                        text, characters = '', []
                    box: np.ndarray = boxes[index]
                    window = windows[index]
                    plates[index] = {
                        'index': index,
                        'text': text,
                        'subtype': subtype,
                        'readable': readable,
                        'valid': is_valid_plate(text, subtype),
                        'confidence': float(confidence[slot]),
                        'detection': float(scores[index]),
                        'character_floor': min((c['probability'] for c in characters), default=0.0),
                        'box': [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                        'window': [float(v) for v in window],
                        'bucket': {'index': bucket, 'height': bucket_h, 'width': bucket_w},
                        'characters': characters,
                        'subtype_probs': [
                            {'name': idx_to_subtype[int(i)], 'probability': float(v)}
                            for v, i in zip(ranked.values[slot], ranked.indices[slot])
                        ],
                        'patch_png': encode_png(patches[index]),
                    }
        return [plate for plate in plates if plate is not None]

    def __call__(self, payload: bytes) -> Dict:
        started: float = time.perf_counter()
        image: np.ndarray = decode_image(payload, self.config.max_pixels)
        decoded: float = time.perf_counter()

        boxes, scores = self._detect(image)
        detected: float = time.perf_counter()

        rgb: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plates: List[Dict] = self._read(rgb, boxes, scores)
        finished: float = time.perf_counter()

        height, width = image.shape[:2]
        return {
            'source': {'width': int(width), 'height': int(height)},
            'plates': sorted(plates, key=lambda item: (item['box'][1], item['box'][0])),
            'timing_ms': {
                'decode': (decoded - started) * 1000.0,
                'detect': (detected - decoded) * 1000.0,
                'recognize': (finished - detected) * 1000.0,
                'total': (finished - started) * 1000.0,
            },
        }
