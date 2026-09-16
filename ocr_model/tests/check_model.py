import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nn.model import PlateOCR
from utils.alphabet import (BOS_IDX, EOS_IDX, MAX_PLATE_LEN, MAX_SEQ_LEN, PAD_IDX,
                            SUBTYPE_PATTERNS, UNKNOWN_SUBTYPE_IDX, build_position_masks,
                            idx_to_char, idx_to_subtype, is_valid_plate, num_classes,
                            num_subtypes, subtype_to_idx)
from utils.dataset import PlateDataset, collate, load_annotations_jsonl
from utils.geometry import BUCKETS, BUCKET_GRIDS
from utils.optimizer import build_param_groups_hybrid
from utils.train import LossWeights, compute_losses, glyph_attention_target

DEVICE: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA: Path = Path(os.environ.get(
    'PLATE_TEST_DATA', str(Path(__file__).resolve().parents[2] / 'outputs' / 'samples_v3'),
))
IMAGES: Path = DATA / 'images' / 'synthetic' if (DATA / 'images' / 'synthetic').is_dir() else DATA / 'images'

PASS: str = 'PASS'
FAIL: str = 'FAIL'
_results: List[Tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = '') -> None:
    _results.append((name, condition, detail))
    status: str = PASS if condition else FAIL
    print(f'[{status}] {name}' + (f' — {detail}' if detail else ''), flush=True)


def build_batch(bucket: int = 4, batch_size: int = 6) -> Dict[str, Tensor]:
    samples = load_annotations_jsonl(DATA / 'annotations.jsonl', IMAGES)
    chosen = [s for s in samples if s.bucket == bucket][:batch_size]
    if len(chosen) < batch_size:
        chosen = (chosen * batch_size)[:batch_size]
    dataset = PlateDataset(chosen, train=False, jitter=0.0)
    return {k: v.to(DEVICE) for k, v in collate([dataset[i] for i in range(len(chosen))]).items()}


def test_grid_consistency(model: PlateOCR) -> None:
    mismatches: List[str] = []
    for index, (height, width) in enumerate(BUCKETS):
        image: Tensor = torch.zeros(1, 3, height, width, device=DEVICE)
        with torch.no_grad():
            _, grid_h, grid_w = model.patch_embed(image)
        if (grid_h, grid_w) != BUCKET_GRIDS[index]:
            mismatches.append(f'{height}x{width}: got {grid_h}x{grid_w}, expected {BUCKET_GRIDS[index]}')
    check('patch grid matches BUCKET_GRIDS for every bucket', not mismatches, '; '.join(mismatches))


def test_causality(model: PlateOCR, batch: Dict[str, Tensor]) -> None:
    model.eval()
    tokens: Tensor = batch['tokens'].clone()
    with torch.no_grad():
        base = model(batch['image'], tokens)['logits']
        altered: Tensor = tokens.clone()
        altered[:, 5] = (altered[:, 5] + 7) % num_classes
        changed = model(batch['image'], altered)['logits']

    prefix_delta: float = (base[:, :5] - changed[:, :5]).abs().max().item()
    suffix_delta: float = (base[:, 5:] - changed[:, 5:]).abs().max().item()
    check('decoder is causal (past logits unaffected by future token)',
          prefix_delta < 1e-5 < suffix_delta, f'prefix delta {prefix_delta:.2e}, suffix delta {suffix_delta:.2e}')


def test_kv_cache_equivalence(model: PlateOCR, batch: Dict[str, Tensor]) -> None:
    model.eval()
    model.prepare_inference()
    images: Tensor = batch['image']

    with torch.no_grad():
        memory, summary, grid_h, grid_w = model.encode(images)
        cache = model.create_cache(images.shape[0], DEVICE, memory.dtype)
        for layer, layer_cache in zip(model.decoder_layers, cache.layers):
            layer.fill_cross_cache(memory, layer_cache, grid_h, grid_w)

        tokens: List[int] = []
        current: Tensor = torch.full((images.shape[0], 1), BOS_IDX, device=DEVICE, dtype=torch.long)
        step_logits: List[Tensor] = []
        for _ in range(MAX_SEQ_LEN):
            x: Tensor = model.text_embed(current) * model.text_scale
            for layer, layer_cache in zip(model.decoder_layers, cache.layers):
                x = layer.forward_cached(x, layer_cache)
            logits: Tensor = model.output_proj(model.decoder_norm(x)[:, -1])
            step_logits.append(logits)
            current = logits.argmax(-1, keepdim=True)
            tokens.append(current)

        sequence: Tensor = torch.cat([torch.full((images.shape[0], 1), BOS_IDX, device=DEVICE, dtype=torch.long)]
                                     + tokens[:-1], dim=1)
        full = model(images, sequence)['logits']

    cached: Tensor = torch.stack(step_logits, dim=1)
    delta: float = (full - cached).abs().max().item()
    check('KV-cache decoding equals full teacher-forcing pass', delta < 2e-3, f'max logit delta {delta:.2e}')


def test_masks_allow_targets() -> None:
    masks: Tensor = build_position_masks()
    samples = load_annotations_jsonl(DATA / 'annotations.jsonl', IMAGES)[:4000]
    dataset = PlateDataset(samples, train=False, jitter=0.0)
    violations: List[str] = []

    for index in range(len(dataset)):
        item = dataset[index]
        target: Tensor = item['target']
        subtype: int = int(item['subtype'])
        for position in range(MAX_SEQ_LEN):
            token: int = int(target[position])
            if token == PAD_IDX:
                break
            if not bool(masks[subtype, position, token]):
                violations.append(
                    f'{idx_to_subtype[subtype]} pos {position} token {idx_to_char[token]!r}'
                )
    check('position masks allow every ground-truth token', not violations,
          f'{len(violations)} violations: {violations[:5]}' if violations else f'{len(dataset)} samples clean')


def test_attention_target(batch: Dict[str, Tensor]) -> None:
    grid_h, grid_w = 11, 26
    target: Tensor = glyph_attention_target(batch['glyph_boxes'], grid_h, grid_w, 4)
    sums: Tensor = target.sum(-1)
    valid: Tensor = batch['glyph_valid']
    check('attention target rows sum to one', bool(((sums - 1.0).abs() < 1e-4).all()),
          f'min {sums.min():.4f} max {sums.max():.4f}')

    peaks: Tensor = target[..., 4:].argmax(-1)
    peak_y: Tensor = (peaks // grid_w).float() / grid_h
    peak_x: Tensor = (peaks % grid_w).float() / grid_w
    error_x: Tensor = (peak_x - batch['glyph_boxes'][..., 0]).abs()[valid]
    error_y: Tensor = (peak_y - batch['glyph_boxes'][..., 1]).abs()[valid]
    check('attention target peaks at the glyph centre',
          float(error_x.max()) < 0.12 and float(error_y.max()) < 0.14,
          f'max offset x {error_x.max():.3f} y {error_y.max():.3f} (cell {1 / grid_w:.3f}x{1 / grid_h:.3f})')


def test_gradient_flow(model: PlateOCR, batch: Dict[str, Tensor]) -> None:
    model.train()
    warmup = torch.optim.SGD(model.parameters(), lr=1e-4)
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        outputs = model(batch['image'], batch['tokens'], need_weights=True)
        compute_losses(outputs, batch, LossWeights(), model.n_registers)['total'].backward()
        warmup.step()

    model.zero_grad(set_to_none=True)
    outputs = model(batch['image'], batch['tokens'], need_weights=True)
    parts = compute_losses(outputs, batch, LossWeights(), model.n_registers, maxsup_alpha=0.002)
    parts['total'].backward()

    missing: List[str] = []
    zeros: List[str] = []
    bad: List[str] = []
    norms: Dict[str, List[float]] = defaultdict(list)

    for name, param in model.named_parameters():
        if param.grad is None:
            missing.append(name)
            continue
        value: float = param.grad.norm().item()
        if not np.isfinite(value):
            bad.append(name)
        elif value == 0.0:
            zeros.append(name)
        group: str = (
            'stem' if 'patch_embed' in name else
            'encoder' if 'encoder_layers' in name else
            'decoder' if 'decoder_layers' in name else
            'heads' if any(h in name for h in ('glyph_head', 'type_head', 'corner_head')) else
            'embed' if 'text_embed' in name or 'output_proj' in name else
            'other'
        )
        norms[group].append(value)

    check('every parameter receives a gradient', not missing, f'missing: {missing[:5]}')
    check('no NaN or Inf gradients', not bad, f'bad: {bad[:5]}')
    check('no all-zero gradients', not zeros, f'zero: {zeros[:6]}')

    summary: str = ' | '.join(
        f'{group}: n={len(values)} med={np.median(values):.2e} max={max(values):.2e}'
        for group, values in sorted(norms.items())
    )
    print(f'       grad norms  {summary}', flush=True)
    medians: List[float] = [float(np.median(v)) for v in norms.values()]
    spread: float = max(medians) / max(1e-12, min(medians))
    check('gradient magnitudes are within three orders across blocks', spread < 1e3, f'spread {spread:.1f}x')


def test_loss_isolation(model: PlateOCR, batch: Dict[str, Tensor]) -> None:
    head_names: Dict[str, str] = {
        'glyph': 'glyph_head', 'corner': 'corner_head', 'subtype': 'type_head',
    }
    problems: List[str] = []
    for loss_name, head in head_names.items():
        weights = LossWeights(char=0.0, glyph=0.0, attention=0.0, corner=0.0, subtype=0.0)
        setattr(weights, loss_name, 1.0)

        model.zero_grad(set_to_none=True)
        outputs = model(batch['image'], batch['tokens'], need_weights=True)
        parts = compute_losses(outputs, batch, weights, model.n_registers)
        parts['total'].backward()

        head_grad: float = sum(
            p.grad.norm().item() for n, p in model.named_parameters() if head in n and p.grad is not None
        )
        encoder_grad: float = sum(
            p.grad.norm().item() for n, p in model.named_parameters()
            if 'encoder_layers' in n and p.grad is not None
        )
        if head_grad <= 0.0:
            problems.append(f'{loss_name} -> {head} no grad')
        if encoder_grad <= 0.0:
            problems.append(f'{loss_name} does not reach encoder')
    check('each auxiliary loss trains its head and reaches the encoder', not problems, '; '.join(problems))

    model.zero_grad(set_to_none=True)
    weights = LossWeights(char=0.0, glyph=0.0, attention=1.0, corner=0.0, subtype=0.0)
    outputs = model(batch['image'], batch['tokens'], need_weights=True)
    compute_losses(outputs, batch, weights, model.n_registers)['total'].backward()
    cross_grad: float = sum(
        p.grad.norm().item() for n, p in model.named_parameters()
        if 'cross_attn' in n and p.grad is not None
    )
    check('attention supervision reaches cross-attention weights', cross_grad > 0.0, f'norm {cross_grad:.2e}')


def test_activation_scale(model: PlateOCR, batch: Dict[str, Tensor]) -> None:
    model.eval()
    stats: List[Tuple[str, float]] = []
    handles = []

    def hook(name: str):
        def fn(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            stats.append((name, tensor.detach().float().std().item()))

        return fn

    for index, layer in enumerate(model.encoder_layers):
        handles.append(layer.register_forward_hook(hook(f'enc{index}')))
    for index, layer in enumerate(model.decoder_layers):
        handles.append(layer.register_forward_hook(hook(f'dec{index}')))

    with torch.no_grad():
        model(batch['image'], batch['tokens'])
    for handle in handles:
        handle.remove()

    values: List[float] = [v for _, v in stats]
    growth: float = max(values) / max(1e-9, min(values))
    print('       activation std  ' + ' '.join(f'{n}={v:.2f}' for n, v in stats), flush=True)
    check('activation scale stays bounded across depth', growth < 25.0 and max(values) < 60.0,
          f'min {min(values):.2f} max {max(values):.2f} growth {growth:.1f}x')


def test_optimizer_partition(model: PlateOCR) -> None:
    muon, decay, no_decay = build_param_groups_hybrid(model)
    total: int = sum(1 for p in model.parameters() if p.requires_grad)
    covered: int = len(muon) + len(decay) + len(no_decay)
    check('optimizer groups cover every trainable tensor', covered == total, f'{covered} of {total}')

    muon_ids = {id(p) for p in muon}
    wrong: List[str] = [
        name for name, param in model.named_parameters()
        if id(param) in muon_ids and (param.ndim != 2 or 'embed' in name or 'head' in name)
    ]
    check('Muon only takes 2D non-embedding matrices', not wrong, f'{wrong[:4]}')
    print(f'       params  muon={sum(p.numel() for p in muon):,} '
          f'adamw_decay={sum(p.numel() for p in decay):,} '
          f'adamw_no_decay={sum(p.numel() for p in no_decay):,}', flush=True)


def test_type_drives_mask(model: PlateOCR, batch: Dict[str, Tensor]) -> None:
    model.eval()
    model.prepare_inference()
    with torch.no_grad():
        prediction = model(batch['image'])

    check('every predicted subtype is a real class',
          bool(((prediction['subtype'] >= 0) & (prediction['subtype'] < num_subtypes)).all()), '')

    masks: Tensor = build_position_masks()
    violations: List[str] = []
    for index in range(prediction['tokens'].shape[0]):
        subtype: int = int(prediction['subtype'][index])
        for position, token in enumerate(prediction['tokens'][index].tolist()):
            if token == PAD_IDX:
                break
            if not bool(masks[subtype, position, token]):
                violations.append(f'{idx_to_subtype[subtype]} pos {position} -> {idx_to_char[token]!r}')
    check('decoded tokens always satisfy the mask of the predicted subtype', not violations,
          f'{violations[:4]}')

    confidence: Tensor = prediction['confidence']
    check('confidence stays inside [0, 1]', bool(((confidence >= 0) & (confidence <= 1)).all()),
          f'min {confidence.min():.3f} max {confidence.max():.3f}')


def main() -> None:
    torch.manual_seed(0)
    print(f'device: {DEVICE}\n')
    model: PlateOCR = PlateOCR(dropout=0.0, drop_path_rate=0.0, stem_dropblock=0.0).to(DEVICE)
    batch: Dict[str, Tensor] = build_batch()

    test_grid_consistency(model)
    test_masks_allow_targets()
    test_attention_target(batch)
    test_causality(model, batch)
    test_kv_cache_equivalence(model, batch)
    test_gradient_flow(model, batch)
    test_loss_isolation(model, batch)
    test_activation_scale(model, batch)
    test_optimizer_partition(model)
    test_type_drives_mask(model, batch)

    failed: int = sum(1 for _, ok, _ in _results if not ok)
    print(f'\n{len(_results) - failed}/{len(_results)} checks passed')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
