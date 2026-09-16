from collections import OrderedDict
from contextlib import contextmanager
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
from torch import Tensor


class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.9999,
                 warmup_steps: int = 3000, inv_gamma: float = 1.0,
                 power: float = 0.75, update_every: int = 10) -> None:
        self.base_decay: float = decay
        self.warmup_steps: int = warmup_steps
        self.inv_gamma: float = inv_gamma
        self.power: float = power
        self.num_updates: int = 0
        self.update_every: int = update_every
        self.step_counter: int = 0

        self.shadow_params: OrderedDict = OrderedDict()
        self._param_refs: List[Tensor] = []
        self._shadow_refs: List[Tensor] = []
        self._param_names: List[str] = []

        self._stream: Optional[torch.cuda.Stream] = None
        self._device: Optional[torch.device] = None

        self._register_params(model)

    def _register_params(self, model: nn.Module) -> None:
        base_model = model.module if hasattr(model, 'module') else model

        for name, param in base_model.named_parameters():
            if not param.requires_grad:
                continue
            shadow = param.data.float().clone().detach()
            self.shadow_params[name] = shadow
            self._param_refs.append(param)
            self._shadow_refs.append(shadow)
            self._param_names.append(name)

            if self._device is None:
                self._device = param.device
                self._stream = torch.cuda.Stream(device=self._device)

        for name, buffer in base_model.named_buffers():
            if buffer.dtype.is_floating_point:
                self.shadow_params[name] = buffer.float().clone().detach()

    def _get_decay(self) -> float:
        if self.num_updates < self.warmup_steps:
            return 0.0
        step = self.num_updates - self.warmup_steps
        decay = 1.0 - (1.0 + step / self.inv_gamma) ** (-self.power)
        return min(decay, self.base_decay)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step_counter += 1
        if self.step_counter % self.update_every != 0:
            return

        decay = self._get_decay()
        self.num_updates += 1

        if decay == 0.0:
            self._copy_params_to_shadow()
            return

        with torch.cuda.stream(self._stream):
            shadow_list = self._shadow_refs
            param_fp32 = [p.data.float() for p in self._param_refs]
            torch._foreach_lerp_(shadow_list, param_fp32, 1.0 - decay)

    def _copy_params_to_shadow(self) -> None:
        with torch.cuda.stream(self._stream):
            for shadow, param in zip(self._shadow_refs, self._param_refs):
                shadow.copy_(param.data.float(), non_blocking=True)

    def sync(self) -> None:
        if self._stream is not None:
            self._stream.synchronize()

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        self.sync()
        base_model = model.module if hasattr(model, 'module') else model

        for name, param in base_model.named_parameters():
            if name in self.shadow_params:
                shadow = self.shadow_params[name]
                param.data.copy_(shadow.to(param.dtype))

        for name, buffer in base_model.named_buffers():
            if name in self.shadow_params:
                shadow = self.shadow_params[name]
                buffer.data.copy_(shadow.to(buffer.dtype))

    def state_dict(self) -> Dict[str, Any]:
        self.sync()
        return {
            'shadow_params': self.shadow_params,
            'base_decay': self.base_decay,
            'warmup_steps': self.warmup_steps,
            'inv_gamma': self.inv_gamma,
            'power': self.power,
            'update_every': self.update_every,
            'num_updates': self.num_updates,
            'step_counter': self.step_counter
        }

    def load_state_dict(self, state_dict: Dict[str, Any], device: torch.device) -> None:
        self.shadow_params = OrderedDict()
        for name, param in state_dict['shadow_params'].items():
            self.shadow_params[name] = param.to(device)

        self._shadow_refs = []
        for name in self._param_names:
            if name in self.shadow_params:
                self._shadow_refs.append(self.shadow_params[name])

        self.base_decay = state_dict.get('base_decay', self.base_decay)
        self.warmup_steps = state_dict.get('warmup_steps', self.warmup_steps)
        self.inv_gamma = state_dict.get('inv_gamma', self.inv_gamma)
        self.power = state_dict.get('power', self.power)
        self.update_every = state_dict.get('update_every', self.update_every)
        self.num_updates = state_dict.get('num_updates', 0)
        self.step_counter = state_dict.get('step_counter', 0)

        self._device = device
        self._stream = torch.cuda.Stream(device=device)

    def get_weight_diff(self, model: nn.Module) -> float:
        self.sync()
        base_model = model.module if hasattr(model, 'module') else model

        total_diff = 0.0
        count = 0

        for name, param in base_model.named_parameters():
            if name not in self.shadow_params:
                continue
            shadow = self.shadow_params[name]
            diff = torch.norm(shadow - param.data.float()).item()
            total_diff += diff
            count += 1

        return total_diff / count if count > 0 else 0.0


@contextmanager
def use_ema_for_eval(model: nn.Module, ema: Optional['EMAModel']):
    if ema is None:
        yield
        return

    ema.sync()
    base_model = model.module if hasattr(model, 'module') else model

    original_params = {}
    for name, param in base_model.named_parameters():
        original_params[name] = param.data.clone()

    original_buffers = {}
    for name, buffer in base_model.named_buffers():
        if buffer.dtype.is_floating_point:
            original_buffers[name] = buffer.data.clone()

    try:
        ema.copy_to(model)
        yield
    finally:
        for name, param in base_model.named_parameters():
            if name in original_params:
                param.data.copy_(original_params[name])

        for name, buffer in base_model.named_buffers():
            if name in original_buffers:
                buffer.data.copy_(original_buffers[name])
