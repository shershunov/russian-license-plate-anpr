import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.alphabet import (MAX_PLATE_LEN, PAD_IDX, UNKNOWN_SUBTYPE_IDX, decode_tokens, idx_to_subtype)
from utils.ema import EMAModel, use_ema_for_eval


@dataclass
class LossWeights:
    char: float = 1.0
    glyph: float = 1.6
    attention: float = 0.12
    corner: float = 0.32
    subtype: float = 0.35


@dataclass
class ValidationResult:
    exact: float = 0.0
    exact_target: float = 0.0
    char_accuracy: float = 0.0
    subtype_accuracy: float = 0.0
    unknown_rejection: float = 0.0
    samples: int = 0
    errors: List[Tuple[str, str, str, str]] = field(default_factory=list)


class PrefetchLoader:
    def __init__(self, dataloader: DataLoader, device: torch.device) -> None:
        self.dataloader: DataLoader = dataloader
        self.device: torch.device = device
        self.stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=device) if device.type == 'cuda' else None
        )

    def __iter__(self) -> Iterator[Dict[str, Tensor]]:
        if self.stream is None:
            for batch in self.dataloader:
                yield {key: value.to(self.device) for key, value in batch.items()}
            return

        first: bool = True
        next_batch: Optional[Dict[str, Tensor]] = None
        for batch in self.dataloader:
            with torch.cuda.stream(self.stream):
                staged: Dict[str, Tensor] = {
                    key: value.to(self.device, non_blocking=True) for key, value in batch.items()
                }
            if not first:
                yield next_batch
            else:
                first = False
            torch.cuda.current_stream().wait_stream(self.stream)
            next_batch = staged
        if next_batch is not None:
            yield next_batch

    def __len__(self) -> int:
        return len(self.dataloader)


class WarmupStableDecayLR(_LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_epochs: int, total_epochs: int,
                 decay_epochs: int, eta_min: float = 0.0,
                 decay_shape: str = 'sqrt', last_epoch: int = -1) -> None:
        self.warmup_epochs: int = warmup_epochs
        self.total_epochs: int = total_epochs
        self.decay_epochs: int = decay_epochs
        self.eta_min: float = eta_min
        self.decay_shape: str = decay_shape
        self.stable_epochs: int = total_epochs - warmup_epochs - decay_epochs

        if self.stable_epochs < 0:
            raise ValueError(
                f'warmup_epochs + decay_epochs > total_epochs: {warmup_epochs}+{decay_epochs} > {total_epochs}'
            )
        if decay_shape not in ('sqrt', 'linear', 'cosine'):
            raise ValueError(f'Unknown decay_shape: {decay_shape}')

        super(WarmupStableDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        epoch: int = self.last_epoch

        if epoch < self.warmup_epochs:
            factor: float = (epoch + 1) / self.warmup_epochs
            return [base_lr * factor for base_lr in self.base_lrs]

        if epoch < self.warmup_epochs + self.stable_epochs:
            return list(self.base_lrs)

        progress: float = (epoch - self.warmup_epochs - self.stable_epochs) / max(self.decay_epochs - 1, 1)
        progress = min(progress, 1.0)

        if self.decay_shape == 'sqrt':
            decay_factor: float = 1.0 - progress ** 0.5
        elif self.decay_shape == 'linear':
            decay_factor = 1.0 - progress
        else:
            decay_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        decay_factor = max(decay_factor, 0.0)
        return [self.eta_min + (base_lr - self.eta_min) * decay_factor for base_lr in self.base_lrs]


def compute_maxsup_alpha(global_step: int, total_steps: int, alpha_max: float) -> float:
    if total_steps <= 0:
        return alpha_max
    return alpha_max * min(1.0, global_step / float(total_steps))


def maxsup_ce_loss(logits: Tensor, target: Tensor, alpha: float, weight: Tensor,
                   label_smoothing: float = 0.0) -> Tuple[Tensor, Tensor, Tensor]:
    per_token: Tensor = F.cross_entropy(
        logits, target, reduction='none', label_smoothing=label_smoothing,
    )
    denominator: Tensor = weight.sum().clamp(min=1.0)
    ce: Tensor = (per_token * weight).sum() / denominator
    if alpha <= 0.0:
        return ce, ce.detach(), torch.zeros((), device=logits.device, dtype=torch.float32)
    z: Tensor = logits.float()
    reg_token: Tensor = z.amax(dim=-1) - z.mean(dim=-1)
    reg: Tensor = (reg_token * weight).sum() / denominator
    return ce + alpha * reg, ce.detach(), reg.detach()


def glyph_attention_target(boxes: Tensor, grid_h: int, grid_w: int, n_registers: int = 0) -> Tensor:
    device: torch.device = boxes.device
    centers_y: Tensor = (torch.arange(grid_h, device=device, dtype=boxes.dtype) + 0.5) / grid_h
    centers_x: Tensor = (torch.arange(grid_w, device=device, dtype=boxes.dtype) + 0.5) / grid_w

    cx, cy, bw, bh = boxes.unbind(-1)
    sigma_x: Tensor = (bw * 0.5).clamp(min=1.0 / grid_w)
    sigma_y: Tensor = (bh * 0.5).clamp(min=1.0 / grid_h)

    dx: Tensor = (centers_x.view(1, 1, 1, -1) - cx[..., None, None]) / sigma_x[..., None, None]
    dy: Tensor = (centers_y.view(1, 1, -1, 1) - cy[..., None, None]) / sigma_y[..., None, None]
    weights: Tensor = torch.exp(-0.5 * (dx * dx + dy * dy)).flatten(2)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    if n_registers > 0:
        weights = F.pad(weights, (n_registers, 0))
    return weights


def attention_kl_loss(attention: Tensor, boxes: Tensor, valid: Tensor, grid_h: int, grid_w: int,
                      n_registers: int) -> Tensor:
    target: Tensor = glyph_attention_target(boxes, grid_h, grid_w)
    patches: Tensor = attention[..., n_registers:]
    patches = patches / patches.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    safe_target: Tensor = target.clamp_min(1e-8)
    kl: Tensor = (safe_target * (safe_target.log() - patches.clamp_min(1e-8).log())).sum(-1)
    return (kl * valid).sum() / valid.sum().clamp_min(1.0)


def compute_losses(outputs: Dict[str, Optional[Tensor]], batch: Dict[str, Tensor], weights: LossWeights,
                   n_registers: int, maxsup_alpha: float = 0.0,
                   label_smoothing: float = 0.0) -> Dict[str, Tensor]:
    logits: Tensor = outputs['logits']
    target: Tensor = batch['target']

    char_weight: Tensor = (target != PAD_IDX).to(logits.dtype)

    char_loss, ce_pure, maxsup_reg = maxsup_ce_loss(
        logits.reshape(-1, logits.shape[-1]), target.reshape(-1), maxsup_alpha,
        char_weight.reshape(-1), label_smoothing,
    )

    glyph_valid: Tensor = batch['glyph_valid'].to(logits.dtype)
    glyph_count: Tensor = glyph_valid.sum().clamp_min(1.0)
    glyph_pred: Tensor = outputs['glyph_boxes'][:, :MAX_PLATE_LEN]
    glyph_loss: Tensor = (
                                 F.smooth_l1_loss(glyph_pred, batch['glyph_boxes'], reduction='none', beta=0.05).mean(
                                     -1) * glyph_valid
                         ).sum() / glyph_count

    grid: Tensor = outputs['grid']
    attention_loss: Tensor = attention_kl_loss(
        outputs['attention'][:, -1, :MAX_PLATE_LEN], batch['glyph_boxes'], glyph_valid,
        int(grid[0]), int(grid[1]), n_registers,
    )

    corners_valid: Tensor = batch['corners_valid'].to(logits.dtype)
    corner_loss: Tensor = (
                                  F.smooth_l1_loss(outputs['corners'], batch['corners'], reduction='none',
                                                   beta=0.05).mean((1, 2))
                                  * corners_valid
                          ).sum() / corners_valid.sum().clamp_min(1.0)

    type_loss: Tensor = F.cross_entropy(outputs['type_logits'], batch['subtype'], label_smoothing=0.02)

    total: Tensor = (
            weights.char * char_loss
            + weights.glyph * glyph_loss
            + weights.attention * attention_loss
            + weights.corner * corner_loss
            + weights.subtype * type_loss
    )
    return {
        'total': total, 'char': char_loss, 'ce_pure': ce_pure, 'maxsup_reg': maxsup_reg,
        'glyph': glyph_loss, 'attention': attention_loss, 'corner': corner_loss,
        'subtype': type_loss,
    }


_METRIC_KEYS: Tuple[str, ...] = (
    'total', 'char', 'ce_pure', 'maxsup_reg', 'glyph', 'attention', 'corner', 'subtype',
    'grad_norm', 'clip_rate',
)


def train_one_epoch(model: nn.Module, loader: PrefetchLoader, optimizer: Any, device: torch.device,
                    weights: LossWeights, n_registers: int, epoch: int, rank: int,
                    global_step: int, maxsup_alpha: float, maxsup_ramp_steps: int,
                    grad_clip: float = 15.0, amp_dtype: Optional[torch.dtype] = torch.bfloat16,
                    distributed: bool = True, ema: Optional[EMAModel] = None) -> Tuple[Dict[str, float], int]:
    model.train()
    totals: Tensor = torch.zeros(len(_METRIC_KEYS), device=device, dtype=torch.float32)
    started: float = time.perf_counter()

    iterator: Union[tqdm, PrefetchLoader] = (
        tqdm(loader, desc='Training', leave=True) if rank == 0 else loader
    )

    for index, batch in enumerate(iterator):
        alpha: float = compute_maxsup_alpha(global_step, maxsup_ramp_steps, maxsup_alpha)
        optimizer.zero_grad(set_to_none=True)

        if amp_dtype is not None and device.type == 'cuda':
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                outputs = model(batch['image'], batch['tokens'], need_weights=True)
                parts = compute_losses(outputs, batch, weights, n_registers, alpha)
        else:
            outputs = model(batch['image'], batch['tokens'], need_weights=True)
            parts = compute_losses(outputs, batch, weights, n_registers, alpha)

        parts['total'].backward()
        grad_norm: Tensor = nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        global_step += 1

        for position, key in enumerate(_METRIC_KEYS[:-2]):
            totals[position] += parts[key].detach()
        totals[-2] += grad_norm.detach()
        totals[-1] += (grad_norm.detach() > grad_clip).to(totals.dtype)

        if rank == 0 and index % 50 == 0:
            iterator.set_postfix({
                'shape': f"{batch['image'].shape[2]}x{batch['image'].shape[3]}",
                'loss': f"{parts['total'].detach().item():.3f}",
                'a': f'{alpha:.4f}',
            })

    if distributed and dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        batches: int = len(loader) * dist.get_world_size()
    else:
        batches = len(loader)

    stats: Dict[str, float] = {
        key: (totals[index] / max(1, batches)).item() for index, key in enumerate(_METRIC_KEYS)
    }
    stats['seconds'] = time.perf_counter() - started
    return stats, global_step


@torch.inference_mode()
def evaluate(model: nn.Module, loader: PrefetchLoader, device: torch.device, rank: int = 0,
             max_errors: int = 32, amp_dtype: Optional[torch.dtype] = torch.bfloat16,
             distributed: bool = True, desc: str = 'Validating') -> ValidationResult:
    model.eval()
    result: ValidationResult = ValidationResult()
    counters: Tensor = torch.zeros(9, device=device, dtype=torch.float32)

    iterator: Union[tqdm, PrefetchLoader] = tqdm(loader, desc=desc, leave=True) if rank == 0 else loader

    for batch in iterator:
        if amp_dtype is not None and device.type == 'cuda':
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                prediction = model(batch['image'])
        else:
            prediction = model(batch['image'])

        tokens: Tensor = prediction['tokens']
        predicted_type: Tensor = prediction['subtype']
        counters[8] += float((predicted_type == batch['subtype']).sum())

        for index in range(tokens.shape[0]):
            reference: str = decode_tokens(batch['target'][index])
            hypothesis: str = decode_tokens(tokens[index])
            true_type: int = int(batch['subtype'][index])
            guess_type: int = int(predicted_type[index])

            counters[0] += float(true_type == guess_type)
            if true_type == UNKNOWN_SUBTYPE_IDX:
                counters[1] += 1.0
                counters[2] += float(guess_type == UNKNOWN_SUBTYPE_IDX and hypothesis == '')
            else:
                counters[3] += 1.0
                counters[4] += float(hypothesis == reference and guess_type == true_type)

            counters[5] += float(hypothesis == reference)
            counters[6] += float(len(reference))
            counters[7] += float(sum(1 for a, b in zip(reference, hypothesis) if a == b))

            if hypothesis != reference and len(result.errors) < max_errors:
                result.errors.append((
                    reference, hypothesis, idx_to_subtype[true_type], idx_to_subtype[guess_type],
                ))

    if distributed and dist.is_initialized():
        dist.all_reduce(counters, op=dist.ReduceOp.SUM)

    samples: float = max(1.0, counters[1].item() + counters[3].item())
    result.samples = int(samples)
    result.unknown_rejection = counters[2].item() / max(1.0, counters[1].item())
    result.exact_target = counters[4].item() / max(1.0, counters[3].item())
    result.exact = counters[5].item() / samples
    result.char_accuracy = counters[7].item() / max(1.0, counters[6].item())
    result.subtype_accuracy = counters[8].item() / samples
    return result
