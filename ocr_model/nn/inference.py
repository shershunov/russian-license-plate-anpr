from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import torch
from nn.model import PlateOCR
from torch import Tensor
from utils.alphabet import UNKNOWN_SUBTYPE, decode_tokens, idx_to_subtype, is_valid_plate
from utils.geometry import BUCKETS, EVAL_INTERPOLATION, render_window, sample_window, select_bucket

GRAPH_BATCH_SIZES: Tuple[int, ...] = (1, 2, 4, 8)


@dataclass(slots=True)
class PlateResult:
    text: str
    subtype: str
    confidence: float
    valid: bool


class _TupleAdapter(torch.nn.Module):
    def __init__(self, model: PlateOCR, keys: Tuple[str, ...]) -> None:
        super().__init__()
        self.model: PlateOCR = model
        self.keys: Tuple[str, ...] = keys

    def forward(self, images: Tensor) -> Tuple[Tensor, ...]:
        output: Dict[str, Tensor] = self.model(images)
        return tuple(output[key] for key in self.keys)


class GraphedPredictor:
    def __init__(self, model: PlateOCR, device: torch.device, dtype: torch.dtype = torch.float32,
                 warmup_iters: int = 3, enabled: bool = True, script: bool = False) -> None:
        self.model: PlateOCR = model
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype
        self.warmup_iters: int = warmup_iters
        self.enabled: bool = enabled and device.type == 'cuda'
        self.script: bool = script
        self._graphs: Dict[Tuple[int, int, int], Tuple[torch.cuda.CUDAGraph, Tensor, Dict[str, Tensor]]] = {}
        self._scripted: Dict[Tuple[int, int, int], torch.jit.ScriptModule] = {}

    @property
    def captured(self) -> int:
        return len(self._graphs)

    @property
    def traced(self) -> int:
        return len(self._scripted)

    def _trace(self, key: Tuple[int, int, int], static_input: Tensor,
               keys: Tuple[str, ...]) -> Optional[torch.jit.ScriptModule]:
        if not self.script:
            return None
        try:
            with torch.no_grad():
                adapter: _TupleAdapter = _TupleAdapter(self.model, keys).eval()
                traced: torch.jit.ScriptModule = torch.jit.freeze(
                    torch.jit.trace(adapter, (static_input,), check_trace=False),
                )
            self._scripted[key] = traced
            return traced
        except Exception:
            self.script = False
            return None

    def _capture(self, key: Tuple[int, int, int]) -> Tuple[torch.cuda.CUDAGraph, Tensor, Dict[str, Tensor]]:
        batch, height, width = key
        static_input: Tensor = torch.zeros(batch, 3, height, width, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            names: Tuple[str, ...] = tuple(self.model(static_input)) if self.script else ()
        runner = self._trace(key, static_input, names) or self.model

        stream: torch.cuda.Stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(self.warmup_iters):
                runner(static_input)
        torch.cuda.current_stream().wait_stream(stream)

        graph: torch.cuda.CUDAGraph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph):
            produced = runner(static_input)
        static_output: Dict[str, Tensor] = (
            produced if isinstance(produced, dict) else dict(zip(names, produced))
        )
        self._graphs[key] = (graph, static_input, static_output)
        return self._graphs[key]

    def warmup(self, batch_sizes: Sequence[int] = GRAPH_BATCH_SIZES) -> None:
        if not self.enabled:
            return
        for height, width in BUCKETS:
            for batch in batch_sizes:
                self._capture((batch, height, width))

    @torch.no_grad()
    def __call__(self, images: Tensor) -> Dict[str, Tensor]:
        if not self.enabled:
            return self.model(images)
        count, _, height, width = images.shape
        batch: int = next((size for size in GRAPH_BATCH_SIZES if size >= count), count)
        key: Tuple[int, int, int] = (batch, height, width)
        entry = self._graphs.get(key) or self._capture(key)
        graph, static_input, static_output = entry
        static_input[:count].copy_(images)
        if batch > count:
            static_input[count:].zero_()
        graph.replay()
        return {name: value[:count].clone() for name, value in static_output.items()}


class PlateRecognizer:
    def __init__(self, model: PlateOCR, device: Optional[torch.device] = None, base_pad: float = 0.04,
                 use_graphs: bool = True, use_script: bool = False,
                 dtype: torch.dtype = torch.float32) -> None:
        self.device: torch.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype: torch.dtype = dtype
        self.model: PlateOCR = model.to(self.device).eval()
        self.model.prepare_inference()
        self.base_pad: float = base_pad
        self.predictor: GraphedPredictor = GraphedPredictor(
            self.model, self.device, dtype, enabled=use_graphs, script=use_script,
        )

    @classmethod
    def from_checkpoint(cls, path: str, device: Optional[torch.device] = None,
                        use_ema: bool = True, **kwargs) -> 'PlateRecognizer':
        target: torch.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint: Dict = torch.load(path, map_location=target, weights_only=False)
        config: Dict = checkpoint.get('config', {})
        model: PlateOCR = PlateOCR(
            dim=config.get('dim', 128), n_heads=config.get('n_heads', 4),
            n_encoder_layers=config.get('n_encoder_layers', 6),
            n_decoder_layers=config.get('n_decoder_layers', 2),
            mlp_ratio=config.get('mlp_ratio', 2.66), dropout=0.0, drop_path_rate=0.0,
            n_registers=config.get('n_registers', 4), rope_base=config.get('rope_base', 64.0),
            value_residual=config.get('value_residual', True), stem_dropblock=0.0,
        )
        ema: Optional[Dict] = checkpoint.get('ema') if use_ema else None
        state: Dict[str, Tensor] = ema.get('shadow_params', ema) if ema else checkpoint['model']
        missing, _ = model.load_state_dict(state, strict=False)
        required: Set[str] = {name for name, _ in model.named_parameters()}
        absent: List[str] = [name for name in missing if name in required]
        if absent:
            raise RuntimeError(f'{path}: checkpoint misses {len(absent)} parameters, first: {absent[:5]}')
        return cls(model, target, **kwargs)

    def _prepare(self, image: np.ndarray, bbox: Sequence[float]) -> Tuple[np.ndarray, int]:
        window: Tuple[float, float, float, float] = sample_window(bbox, self.base_pad, 0.0, None)
        bucket: int = select_bucket((window[2] - window[0]) / (window[3] - window[1]))
        return render_window(image, window, bucket, EVAL_INTERPOLATION), bucket

    def _to_tensor(self, patches: List[np.ndarray]) -> Tensor:
        stacked: np.ndarray = np.stack(patches)
        tensor: Tensor = torch.from_numpy(stacked).to(self.device, non_blocking=True)
        tensor = tensor.permute(0, 3, 1, 2).to(self.dtype)
        return (255.0 - tensor) / 255.0

    def __call__(self, image: np.ndarray, bboxes: Sequence[Sequence[float]]) -> List[PlateResult]:
        if not bboxes:
            return []

        by_bucket: Dict[int, List[int]] = {}
        patches: List[Optional[np.ndarray]] = [None] * len(bboxes)
        for index, bbox in enumerate(bboxes):
            patch, bucket = self._prepare(image, bbox)
            patches[index] = patch
            by_bucket.setdefault(bucket, []).append(index)

        results: List[Optional[PlateResult]] = [None] * len(bboxes)
        for indices in by_bucket.values():
            batch: Tensor = self._to_tensor([patches[i] for i in indices])
            output: Dict[str, Tensor] = self.predictor(batch)
            tokens: Tensor = output['tokens'].cpu()
            subtypes: Tensor = output['subtype'].cpu()
            confidence: Tensor = output['confidence'].cpu()
            for position, index in enumerate(indices):
                type_name: str = idx_to_subtype[int(subtypes[position])]
                text: str = decode_tokens(tokens[position])
                if type_name == UNKNOWN_SUBTYPE:
                    text = ''
                results[index] = PlateResult(
                    text=text,
                    subtype=type_name,
                    confidence=float(confidence[position]),
                    valid=is_valid_plate(text, type_name),
                )
        return [result for result in results if result is not None]


def recognize_directory(recognizer: PlateRecognizer, detections: Dict[str, List[Sequence[float]]],
                        images_root: str) -> List[Tuple[str, PlateResult]]:
    rows: List[Tuple[str, PlateResult]] = []
    for name, bboxes in detections.items():
        image: Optional[np.ndarray] = cv2.imread(f'{images_root}/{name}', cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for result in recognizer(image, bboxes):
            rows.append((name, result))
    return rows


def write_results_csv(path: str, rows: Sequence[Tuple[str, PlateResult]], skip_unknown: bool = False) -> None:
    with open(path, 'w', encoding='utf-8', newline='') as fout:
        fout.write('image;plate_num;subtype;confidence\n')
        for name, result in rows:
            if skip_unknown and result.subtype == UNKNOWN_SUBTYPE:
                continue
            fout.write(f'{name};{result.text};{result.subtype};{result.confidence:.4f}\n')
