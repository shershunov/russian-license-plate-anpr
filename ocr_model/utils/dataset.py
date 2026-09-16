from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler
from utils.alphabet import BOS_IDX, EOS_IDX, MAX_PLATE_LEN, MAX_SEQ_LEN, PAD_IDX, char_to_idx
from utils.geometry import (EVAL_INTERPOLATION, NUM_BUCKETS, normalize_points, quad_to_box,
                            render_window, sample_interpolation, sample_window)
from utils.samples import (CACHE_VERSION, COLUMNS, SampleBuilder, SampleTable,
                           load_annotations_jsonl, load_meta_csv, load_plates_tsv,
                           load_sharded_annotations, subtype_candidates)

cv2.setNumThreads(0)

__all__ = ['BucketBatchSampler', 'CACHE_VERSION', 'COLUMNS', 'PhotometricAugmenter',
           'PlateDataset', 'SampleBuilder', 'SampleTable', 'collate', 'load_annotations_jsonl',
           'load_meta_csv', 'load_plates_tsv', 'load_sharded_annotations', 'subtype_candidates']


class PhotometricAugmenter:
    def __init__(self, probability: float = 0.9) -> None:
        self.probability: float = probability

    def __call__(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() >= self.probability:
            return image
        if rng.random() < 0.55:
            image = self._brightness_contrast(image, rng)
        if rng.random() < 0.25:
            image = self._gamma(image, rng)
        if rng.random() < 0.20:
            image = self._blur(image, rng)
        if rng.random() < 0.20:
            image = self._sharpen(image, rng)
        if rng.random() < 0.25:
            image = self._color_shift(image, rng)
        if rng.random() < 0.12:
            image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        if rng.random() < 0.30:
            image = self._noise(image, rng)
        if rng.random() < 0.20:
            image = self._jpeg(image, rng)
        return image

    @staticmethod
    def _brightness_contrast(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return cv2.convertScaleAbs(image, alpha=float(rng.uniform(0.65, 1.4)), beta=float(rng.uniform(-45.0, 40.0)))

    @staticmethod
    def _gamma(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        gamma: float = float(rng.uniform(0.6, 1.7))
        table: np.ndarray = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
        return cv2.LUT(image, table)

    @staticmethod
    def _blur(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() < 0.5:
            return cv2.GaussianBlur(image, (3, 3), sigmaX=float(rng.uniform(0.4, 1.6)))
        length: int = int(rng.integers(3, 8))
        kernel: np.ndarray = np.zeros((length, length), dtype=np.float32)
        angle: float = float(rng.uniform(0.0, np.pi))
        center: float = (length - 1) * 0.5
        for i in range(length):
            offset: float = i - center
            row: int = int(round(center + offset * np.sin(angle)))
            col: int = int(round(center + offset * np.cos(angle)))
            kernel[np.clip(row, 0, length - 1), np.clip(col, 0, length - 1)] = 1.0
        kernel /= kernel.sum()
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def _sharpen(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        amount: float = float(rng.uniform(0.2, 0.8))
        blurred: np.ndarray = cv2.GaussianBlur(image, (0, 0), 1.0)
        return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0.0)

    @staticmethod
    def _color_shift(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        hsv: np.ndarray = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + int(rng.integers(-8, 9))) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] + int(rng.integers(-25, 26)), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] + int(rng.integers(-20, 21)), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    @staticmethod
    def _noise(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        noise: np.ndarray = rng.normal(0.0, float(rng.uniform(2.0, 14.0)), image.shape).astype(np.float32)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    @staticmethod
    def _jpeg(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        ok, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), int(rng.integers(28, 75))])
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR) if ok else image


class PlateDataset(Dataset):
    def __init__(self, table: SampleTable, indices: Optional[np.ndarray] = None, train: bool = True,
                 base_pad: float = 0.04, jitter: float = 0.20,
                 augmenter: Optional[PhotometricAugmenter] = None, seed: int = 1337) -> None:
        self.table: SampleTable = table
        self.indices: np.ndarray = (np.arange(len(table), dtype=np.int64) if indices is None
                                    else np.asarray(indices, dtype=np.int64))
        self.train: bool = train
        self.base_pad: float = base_pad
        self.jitter: float = jitter if train else 0.0
        self.augmenter: Optional[PhotometricAugmenter] = augmenter if train else None
        self.seed: int = seed
        self.epoch: int = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    @property
    def buckets(self) -> np.ndarray:
        return np.asarray(self.table.bucket)[self.indices].astype(np.int64)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        row: int = int(self.indices[index])
        table: SampleTable = self.table
        rng: Optional[np.random.Generator] = np.random.default_rng(
            (self.seed * 1000003 + self.epoch * 9176 + row) & 0x7FFFFFFF
        ) if self.train else None

        path: str = table.path(row)
        image: Optional[np.ndarray] = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bbox: Tuple[float, float, float, float] = tuple(float(v) for v in table.bbox[row])
        bucket: int = int(table.bucket[row])
        window: Tuple[float, float, float, float] = sample_window(bbox, self.base_pad, self.jitter, rng)
        interpolation: int = sample_interpolation(rng) if self.train else EVAL_INTERPOLATION
        patch: np.ndarray = render_window(image, window, bucket, interpolation)

        if self.augmenter is not None:
            patch = self.augmenter(patch, rng)

        tensor: Tensor = torch.from_numpy(np.ascontiguousarray(patch)).permute(2, 0, 1).float()
        tensor = (255.0 - tensor) / 255.0

        plate_num: str = table.text(row)
        tokens: Tensor = torch.full((MAX_SEQ_LEN,), PAD_IDX, dtype=torch.long)
        target: Tensor = torch.full((MAX_SEQ_LEN,), PAD_IDX, dtype=torch.long)
        ids: List[int] = [char_to_idx[c] for c in plate_num] + [EOS_IDX]
        tokens[0] = BOS_IDX
        tokens[1:len(ids)] = torch.tensor(ids[:-1], dtype=torch.long)
        target[:len(ids)] = torch.tensor(ids, dtype=torch.long)

        glyph_boxes: Tensor = torch.zeros(MAX_PLATE_LEN, 4, dtype=torch.float32)
        glyph_valid: Tensor = torch.zeros(MAX_PLATE_LEN, dtype=torch.bool)
        stored: int = int(table.glyph_count[row])
        if stored and plate_num:
            count: int = min(len(plate_num), stored, MAX_PLATE_LEN)
            normalized: np.ndarray = normalize_points(
                np.asarray(table.glyph_quads[row, :count], dtype=np.float32), window)
            boxes: np.ndarray = np.stack([quad_to_box(q) for q in normalized])
            inside: np.ndarray = (boxes[:, 0] > -0.05) & (boxes[:, 0] < 1.05)
            inside &= (boxes[:, 1] > -0.05) & (boxes[:, 1] < 1.05)
            glyph_boxes[:count] = torch.from_numpy(boxes)
            glyph_valid[:count] = torch.from_numpy(inside)

        corners: Tensor = torch.zeros(4, 2, dtype=torch.float32)
        corners_valid: Tensor = torch.zeros((), dtype=torch.bool)
        if bool(table.has_quad[row]):
            corners = torch.from_numpy(
                normalize_points(np.asarray(table.quad[row], dtype=np.float32), window))
            corners_valid = torch.ones((), dtype=torch.bool)

        return {
            'image': tensor,
            'tokens': tokens,
            'target': target,
            'subtype': torch.tensor(int(table.subtype[row]), dtype=torch.long),
            'bucket': torch.tensor(bucket, dtype=torch.long),
            'glyph_boxes': glyph_boxes,
            'glyph_valid': glyph_valid,
            'corners': corners,
            'corners_valid': corners_valid,
        }


class BucketBatchSampler(Sampler[List[int]]):
    def __init__(self, buckets: np.ndarray, batch_size: int, shuffle: bool = True,
                 drop_last: bool = False, seed: int = 1337, rank: int = 0, world_size: int = 1) -> None:
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self.drop_last: bool = drop_last
        self.seed: int = seed
        self.epoch: int = 0
        self.rank: int = rank
        self.world_size: int = world_size
        groups: List[np.ndarray] = [np.flatnonzero(buckets == b) for b in range(NUM_BUCKETS)]
        self.groups: List[np.ndarray] = [g for g in groups if g.size > 0]
        self._length: int = self._count_batches()

    def _count_batches(self) -> int:
        total: int = 0
        for group in self.groups:
            total += group.size // self.batch_size if self.drop_last else -(-group.size // self.batch_size)
        return max(1, total // self.world_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[List[int]]:
        rng: np.random.Generator = np.random.default_rng(self.seed + self.epoch)
        batches: List[np.ndarray] = []
        for group in self.groups:
            order: np.ndarray = rng.permutation(group) if self.shuffle else group
            limit: int = (order.size // self.batch_size) * self.batch_size if self.drop_last else order.size
            if limit == 0:
                continue
            batches.extend(np.array_split(order[:limit], -(-limit // self.batch_size)))
        if self.shuffle:
            rng.shuffle(batches)
        shard: List[np.ndarray] = batches[self.rank::self.world_size]
        while len(shard) < self._length:
            shard.append(batches[len(shard) % len(batches)])
        for batch in shard[:self._length]:
            yield batch.tolist()


def collate(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}
