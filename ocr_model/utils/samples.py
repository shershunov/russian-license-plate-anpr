import csv
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from utils.alphabet import (MAX_PLATE_LEN, SUBTYPES, UNKNOWN_SUBTYPE_IDX, idx_to_subtype,
                            infer_subtype, is_readable_subtype, is_valid_plate, subtype_to_idx)
from utils.geometry import MAX_ASPECT_RATIO, MIN_ASPECT_RATIO, order_quad, select_bucket

CIVIL_SUBTYPES: Tuple[str, ...] = ('type1', 'type1a', 'type1b')
MILITARY_SUBTYPES: Tuple[str, ...] = ('type5', 'type6', 'type7', 'type8')


def subtype_candidates(source: str) -> Tuple[str, ...]:
    head: Tuple[str, ...] = MILITARY_SUBTYPES if source == 'military' else CIVIL_SUBTYPES
    return head + tuple(name for name in SUBTYPES if name not in head)


COLUMNS: Tuple[str, ...] = (
    'path_blob', 'path_index', 'text_blob', 'text_index', 'subtype', 'bucket',
    'bbox', 'quad', 'has_quad', 'glyph_quads', 'glyph_count', 'readable_share', 'is_synthetic',
)
CACHE_VERSION: str = 'v1'


@dataclass(slots=True)
class SampleTable:
    path_blob: np.ndarray
    path_index: np.ndarray
    text_blob: np.ndarray
    text_index: np.ndarray
    subtype: np.ndarray
    bucket: np.ndarray
    bbox: np.ndarray
    quad: np.ndarray
    has_quad: np.ndarray
    glyph_quads: np.ndarray
    glyph_count: np.ndarray
    readable_share: np.ndarray
    is_synthetic: np.ndarray

    def __len__(self) -> int:
        return int(self.subtype.shape[0])

    def path(self, index: int) -> str:
        start, stop = self.path_index[index], self.path_index[index + 1]
        return self.path_blob[start:stop].tobytes().decode('utf-8')

    def text(self, index: int) -> str:
        start, stop = self.text_index[index], self.text_index[index + 1]
        return self.text_blob[start:stop].tobytes().decode('utf-8')

    def paths(self) -> List[str]:
        return [self.path(i) for i in range(len(self))]

    def nbytes(self) -> int:
        return sum(int(getattr(self, name).nbytes) for name in COLUMNS)

    def save(self, folder: Path) -> None:
        staging: Path = folder.with_name(folder.name + '.tmp')
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        for name in COLUMNS:
            np.save(staging / f'{name}.npy', getattr(self, name), allow_pickle=False)
        (staging / 'version.txt').write_text(CACHE_VERSION, encoding='utf-8')
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)
        staging.replace(folder)

    @classmethod
    def load(cls, folder: Path) -> 'SampleTable':
        return cls(**{name: np.load(folder / f'{name}.npy', mmap_mode='r') for name in COLUMNS})

    @classmethod
    def concat(cls, tables: Sequence['SampleTable']) -> 'SampleTable':
        alive: List['SampleTable'] = [t for t in tables if len(t) > 0]
        if not alive:
            return SampleBuilder().build()
        if len(alive) == 1:
            return alive[0]
        payload: Dict[str, np.ndarray] = {}
        for name in ('path', 'text'):
            blobs: List[np.ndarray] = [np.asarray(getattr(t, f'{name}_blob')) for t in alive]
            payload[f'{name}_blob'] = np.concatenate(blobs)
            shift: int = 0
            parts: List[np.ndarray] = []
            for table in alive:
                index: np.ndarray = np.asarray(getattr(table, f'{name}_index'))
                parts.append(index[1:] + shift)
                shift += int(index[-1])
            payload[f'{name}_index'] = np.concatenate(
                [np.zeros(1, dtype=np.int64)] + parts).astype(np.int64)
        for name in COLUMNS:
            if name.endswith('_blob') or name.endswith('_index'):
                continue
            payload[name] = np.concatenate([np.asarray(getattr(t, name)) for t in alive])
        return cls(**payload)


class SampleBuilder:
    def __init__(self) -> None:
        self.path_blob: bytearray = bytearray()
        self.path_index: List[int] = [0]
        self.text_blob: bytearray = bytearray()
        self.text_index: List[int] = [0]
        self.subtype: List[int] = []
        self.bucket: List[int] = []
        self.bbox: List[Tuple[float, float, float, float]] = []
        self.quad: List[Optional[np.ndarray]] = []
        self.glyph_quads: List[Optional[np.ndarray]] = []
        self.readable_share: List[float] = []
        self.is_synthetic: List[bool] = []

    def add(self, image_path: str, plate_num: str, subtype: int,
            bbox: Tuple[float, float, float, float], quad: Optional[np.ndarray],
            glyph_quads: Optional[np.ndarray], readable_share: float,
            is_synthetic: bool, bucket: int) -> None:
        self.path_blob.extend(image_path.encode('utf-8'))
        self.path_index.append(len(self.path_blob))
        self.text_blob.extend(plate_num.encode('utf-8'))
        self.text_index.append(len(self.text_blob))
        self.subtype.append(subtype)
        self.bucket.append(bucket)
        self.bbox.append(bbox)
        self.quad.append(quad)
        self.glyph_quads.append(glyph_quads)
        self.readable_share.append(readable_share)
        self.is_synthetic.append(is_synthetic)

    def __len__(self) -> int:
        return len(self.subtype)

    def build(self) -> SampleTable:
        count: int = len(self.subtype)
        quads: np.ndarray = np.zeros((count, 4, 2), dtype=np.float32)
        has_quad: np.ndarray = np.zeros(count, dtype=bool)
        glyphs: np.ndarray = np.zeros((count, MAX_PLATE_LEN, 4, 2), dtype=np.float32)
        glyph_count: np.ndarray = np.zeros(count, dtype=np.int8)
        for i, (plate_quad, glyph) in enumerate(zip(self.quad, self.glyph_quads)):
            if plate_quad is not None:
                quads[i] = plate_quad
                has_quad[i] = True
            if glyph is not None and len(glyph):
                taken: int = min(len(glyph), MAX_PLATE_LEN)
                glyphs[i, :taken] = glyph[:taken]
                glyph_count[i] = taken
        return SampleTable(
            path_blob=np.frombuffer(bytes(self.path_blob), dtype=np.uint8),
            path_index=np.asarray(self.path_index, dtype=np.int64),
            text_blob=np.frombuffer(bytes(self.text_blob), dtype=np.uint8),
            text_index=np.asarray(self.text_index, dtype=np.int64),
            subtype=np.asarray(self.subtype, dtype=np.int16),
            bucket=np.asarray(self.bucket, dtype=np.int8),
            bbox=np.asarray(self.bbox, dtype=np.float32).reshape(count, 4),
            quad=quads,
            has_quad=has_quad,
            glyph_quads=glyphs,
            glyph_count=glyph_count,
            readable_share=np.asarray(self.readable_share, dtype=np.float32),
            is_synthetic=np.asarray(self.is_synthetic, dtype=bool),
        )


def _resolve_image_layout(root: Path, relatives: Sequence[str],
                          images_root: Optional[Path]) -> Callable[[str], str]:
    bases: List[Tuple[Path, bool]] = []
    if images_root is not None:
        bases.extend([(images_root, False), (images_root, True)])
    bases.extend([
        (root, True), (root, False),
        (root / 'images', False), (root / 'images' / 'synthetic', False),
    ])
    for base, keep_relative in bases:
        if all((base / rel if keep_relative else base / Path(rel).name).exists() for rel in relatives):
            if keep_relative:
                return lambda rel, base=base: str(base / rel)
            return lambda rel, base=base: str(base / Path(rel).name)
    searched: str = ', '.join(str(base) for base, _ in bases)
    raise FileNotFoundError(f'cannot locate images for {relatives[0]!r}; looked under: {searched}')


def _bbox_from_quad(quad: np.ndarray) -> Tuple[float, float, float, float]:
    lo: np.ndarray = quad.min(axis=0)
    hi: np.ndarray = quad.max(axis=0)
    return float(lo[0]), float(lo[1]), float(hi[0] - lo[0]), float(hi[1] - lo[1])


def _fill_annotations_jsonl(path: Path, builder: SampleBuilder,
                            images_root: Optional[Path] = None, validate: bool = True) -> None:
    root: Path = path.parent
    with open(path, 'r', encoding='utf-8') as fin:
        probe: List[str] = []
        for line in fin:
            if line.strip():
                probe.append(json.loads(line)['image'])
            if len(probe) >= 3:
                break
        resolve: Callable[[str], str] = _resolve_image_layout(root, probe, images_root)
        fin.seek(0)
        for line in fin:
            if not line.strip():
                continue
            record: Dict = json.loads(line)
            text: str = record.get('plate_num_true', record['plate_num'])
            subtype_idx: int = subtype_to_idx.get(record.get('subtype', ''), UNKNOWN_SUBTYPE_IDX)
            readable: bool = is_readable_subtype(subtype_idx)
            if validate and readable and not is_valid_plate(text, idx_to_subtype[subtype_idx]):
                continue
            bbox: Tuple[float, float, float, float] = tuple(float(v) for v in record['bbox'])
            if bbox[2] <= 1.0 or bbox[3] <= 1.0:
                continue
            if not MIN_ASPECT_RATIO <= bbox[2] / bbox[3] <= MAX_ASPECT_RATIO:
                continue
            glyphs: List[Dict] = record.get('glyphs') or []
            quads: Optional[np.ndarray] = None
            if glyphs and readable:
                quads = np.array([g['quad'] for g in glyphs], dtype=np.float32)
            plate_quad: Optional[np.ndarray] = None
            if record.get('quad') is not None:
                plate_quad = order_quad(np.array(record['quad'], dtype=np.float32))
            builder.add(
                image_path=resolve(record['image']),
                plate_num=text if readable else '',
                subtype=subtype_idx,
                bbox=bbox,
                quad=plate_quad,
                glyph_quads=quads,
                readable_share=float(record.get('readable_share', 1.0)),
                is_synthetic=True,
                bucket=select_bucket(bbox[2] / bbox[3]),
            )


def load_annotations_jsonl(path: Path, images_root: Optional[Path] = None,
                           validate: bool = True) -> SampleTable:
    builder: SampleBuilder = SampleBuilder()
    _fill_annotations_jsonl(path, builder, images_root, validate)
    return builder.build()


def _fill_meta_csv(path: Path, builder: SampleBuilder, images_root: Optional[Path] = None) -> None:
    root: Path = images_root if images_root is not None else path.parent
    with open(path, 'r', encoding='utf-8', newline='') as fin:
        reader = csv.reader(fin, delimiter=';', quoting=csv.QUOTE_NONE)
        header: List[str] = next(reader)
        cols: Dict[str, int] = {name: i for i, name in enumerate(header)}
        for row in reader:
            if len(row) < len(header):
                continue
            text: str = row[cols['plate_num']]
            declared: str = row[cols['subtype']] if 'subtype' in cols else row[cols['plate_type']]
            subtype_idx: int = subtype_to_idx.get(declared, UNKNOWN_SUBTYPE_IDX)
            if subtype_idx == UNKNOWN_SUBTYPE_IDX:
                subtype_idx = infer_subtype(text, subtype_candidates(
                    row[cols['source']] if 'source' in cols else ''))
            raw_quad: str = row[cols['quad']] if 'quad' in cols else ''
            quad_values: List[float] = [float(v) for v in raw_quad.split(',')] if raw_quad else []
            quad: Optional[np.ndarray] = (
                order_quad(np.array(quad_values, dtype=np.float32).reshape(4, 2)) if len(quad_values) == 8 else None
            )
            raw_bbox: str = row[cols['bbox']] if 'bbox' in cols else ''
            if raw_bbox:
                bbox = tuple(float(v) for v in raw_bbox.split(','))
            elif quad is not None:
                bbox = _bbox_from_quad(quad)
            else:
                continue
            if bbox[2] <= 1.0 or bbox[3] <= 1.0:
                continue
            builder.add(
                image_path=str(root / row[cols['image']]),
                plate_num=text if is_readable_subtype(subtype_idx) else '',
                subtype=subtype_idx,
                bbox=bbox,
                quad=quad,
                glyph_quads=None,
                readable_share=1.0,
                is_synthetic=bool(int(row[cols['is_synthetic']])) if 'is_synthetic' in cols else False,
                bucket=select_bucket(bbox[2] / bbox[3]),
            )


def load_meta_csv(path: Path, images_root: Optional[Path] = None) -> SampleTable:
    builder: SampleBuilder = SampleBuilder()
    _fill_meta_csv(path, builder, images_root)
    return builder.build()


def _fill_plates_tsv(path: Path, builder: SampleBuilder, images_root: Optional[Path] = None) -> None:
    root: Path = images_root if images_root is not None else path.parent / path.stem / 'images'
    with open(path, 'r', encoding='utf-8', newline='') as fin:
        reader = csv.reader(fin, delimiter='\t', quoting=csv.QUOTE_NONE)
        header: List[str] = next(reader)
        cols: Dict[str, int] = {name: i for i, name in enumerate(header)}
        for row in reader:
            if len(row) < len(header):
                continue
            text: str = row[cols['text']].strip().upper()
            width: float = float(row[cols['width']] or 0.0)
            height: float = float(row[cols['height']] or 0.0)
            if width <= 1.0 or height <= 1.0:
                continue
            subtype_idx: int = infer_subtype(text, subtype_candidates(
                row[cols['source']] if 'source' in cols else ''))
            builder.add(
                image_path=str(root / row[cols['filename']]),
                plate_num=text if is_readable_subtype(subtype_idx) else '',
                subtype=subtype_idx,
                bbox=(0.0, 0.0, width, height),
                quad=None,
                glyph_quads=None,
                readable_share=1.0,
                is_synthetic=False,
                bucket=select_bucket(width / height),
            )


def load_plates_tsv(path: Path, images_root: Optional[Path] = None) -> SampleTable:
    builder: SampleBuilder = SampleBuilder()
    _fill_plates_tsv(path, builder, images_root)
    return builder.build()


def _cache_signature(sources: Sequence[Path]) -> str:
    digest = hashlib.sha1(CACHE_VERSION.encode('utf-8'))
    for source in sources:
        stat = source.stat()
        digest.update(f'{source}|{stat.st_size}|{int(stat.st_mtime)}'.encode('utf-8'))
    return digest.hexdigest()[:16]


def _cache_folder(root: Path, signature: str) -> Path:
    candidates: List[Path] = [root / '.ocr_cache', Path.home() / '.cache' / 'plate_ocr']
    for base in candidates:
        try:
            base.mkdir(parents=True, exist_ok=True)
            probe: Path = base / '.writable'
            probe.touch()
            probe.unlink()
            return base / signature
        except OSError:
            continue
    return Path(tempfile.gettempdir()) / 'plate_ocr' / signature


def load_sharded_annotations(root: Path, pattern: str = 'shard_*') -> SampleTable:
    shards: List[Path] = [shard / 'annotations.jsonl' for shard in sorted(root.glob(pattern))]
    sources: List[Path] = [path for path in shards if path.exists()]
    if not sources:
        return SampleBuilder().build()
    folder: Path = _cache_folder(root, _cache_signature(sources))
    if (folder / 'version.txt').exists():
        return SampleTable.load(folder)
    builder: SampleBuilder = SampleBuilder()
    for source in sources:
        _fill_annotations_jsonl(source, builder)
    table: SampleTable = builder.build()
    try:
        table.save(folder)
        return SampleTable.load(folder)
    except OSError:
        return table


def main() -> None:
    import argparse
    import time

    parser = argparse.ArgumentParser(description='build the sample cache for a sharded dataset')
    parser.add_argument('root')
    parser.add_argument('--pattern', default='shard_*')
    args = parser.parse_args()
    started: float = time.perf_counter()
    table: SampleTable = load_sharded_annotations(Path(args.root), args.pattern)
    print(f'{len(table)} samples, {table.nbytes() / 1e9:.2f} GB columns, '
          f'{time.perf_counter() - started:.1f}s')


if __name__ == '__main__':
    main()
