from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.optim import Muon

_NO_DECAY_NAME_SUBSTR: Tuple[str, ...] = (
    'norm',
    'text_scale',
    'rope',
    'freqs_',
    'register_tokens',
    'v_gate',
)

_MUON_EXCLUDED_SUBSTR: Tuple[str, ...] = (
    'text_embed',
    'patch_embed',
    'output_proj',
    'glyph_head',
    'type_head',
    'corner_head',
    'quality_head',
    'v_gate',
)


def _is_no_decay(name: str, ndim: int) -> bool:
    if ndim <= 1:
        return True
    return any(s in name for s in _NO_DECAY_NAME_SUBSTR)


def _is_muon_excluded(name: str, ndim: int) -> bool:
    if ndim != 2:
        return True
    return any(s in name for s in _MUON_EXCLUDED_SUBSTR)


def build_param_groups_hybrid(
        model: nn.Module,
) -> Tuple[List[nn.Parameter], List[nn.Parameter], List[nn.Parameter]]:
    muon_params: List[nn.Parameter] = []
    adamw_decay: List[nn.Parameter] = []
    adamw_no_decay: List[nn.Parameter] = []

    seen: set = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        pid: int = id(param)
        if pid in seen:
            continue
        seen.add(pid)

        if _is_no_decay(name, param.ndim):
            adamw_no_decay.append(param)
            continue
        if _is_muon_excluded(name, param.ndim):
            adamw_decay.append(param)
            continue
        muon_params.append(param)

    return muon_params, adamw_decay, adamw_no_decay


class HybridOptimizer:
    def __init__(self, optimizers: Sequence[torch.optim.Optimizer]) -> None:
        if not optimizers:
            raise ValueError('HybridOptimizer requires at least one optimizer')
        self.optimizers: List[torch.optim.Optimizer] = list(optimizers)

    @property
    def param_groups(self) -> List[dict]:
        groups: List[dict] = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups

    @property
    def defaults(self) -> dict:
        return self.optimizers[0].defaults

    @property
    def state(self) -> Dict[Any, Any]:
        merged: Dict[Any, Any] = {}
        for opt in self.optimizers:
            merged.update(opt.state)
        return merged

    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            loss = closure()
        for opt in self.optimizers:
            opt.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {f'opt_{i}': opt.state_dict() for i, opt in enumerate(self.optimizers)}

    def load_state_dict(self, state_dict: dict) -> None:
        for i, opt in enumerate(self.optimizers):
            opt.load_state_dict(state_dict[f'opt_{i}'])


class MultiScheduler:
    def __init__(self, schedulers: Sequence[Any]) -> None:
        self.schedulers: List[Any] = list(schedulers)

    def step(self, *args: Any, **kwargs: Any) -> None:
        for s in self.schedulers:
            s.step(*args, **kwargs)

    def get_last_lr(self) -> List[float]:
        lrs: List[float] = []
        for s in self.schedulers:
            lrs.extend(s.get_last_lr())
        return lrs

    def state_dict(self) -> dict:
        return {f's_{i}': s.state_dict() for i, s in enumerate(self.schedulers)}

    def load_state_dict(self, state_dict: dict) -> None:
        for i, s in enumerate(self.schedulers):
            s.load_state_dict(state_dict[f's_{i}'])


def build_hybrid_optimizer(
        model: nn.Module,
        muon_lr: float,
        muon_momentum: float,
        muon_weight_decay: float,
        muon_ns_steps: int,
        adamw_lr: float,
        adamw_betas: Tuple[float, float],
        adamw_weight_decay: float,
) -> Tuple[HybridOptimizer, int, int, int]:
    muon_params, adamw_decay, adamw_no_decay = build_param_groups_hybrid(model)

    muon_opt: Muon = Muon(
        muon_params,
        lr=muon_lr,
        momentum=muon_momentum,
        weight_decay=muon_weight_decay,
        ns_steps=muon_ns_steps,
    )

    adamw_groups: List[dict] = [
        {'params': adamw_decay, 'weight_decay': adamw_weight_decay},
        {'params': adamw_no_decay, 'weight_decay': 0.0},
    ]
    adamw_opt: torch.optim.AdamW = torch.optim.AdamW(
        adamw_groups,
        lr=adamw_lr,
        betas=adamw_betas,
    )

    n_muon: int = sum(p.numel() for p in muon_params)
    n_decay: int = sum(p.numel() for p in adamw_decay)
    n_no_decay: int = sum(p.numel() for p in adamw_no_decay)

    return HybridOptimizer([muon_opt, adamw_opt]), n_muon, n_decay, n_no_decay
