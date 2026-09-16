from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter

try:
    from flash_attn import flash_attn_func

    FLASH_AVAILABLE: bool = True
except ImportError:
    FLASH_AVAILABLE: bool = False

from utils.alphabet import (BOS_IDX, EOS_IDX, MAX_SEQ_LEN, PAD_IDX, REGION3_PREFIX_IDS,
                            UNKNOWN_SUBTYPE_IDX, build_position_masks, build_region_gate,
                            build_union_position_mask, num_classes, num_subtypes)
from utils.geometry import BUCKET_GRIDS, MAX_GRID_H, MAX_GRID_W, MAX_TOKENS, PATCH_SIZE


def _cast(t: Tensor, dtype: torch.dtype) -> Tensor:
    return t if t.dtype == dtype else t.to(dtype)


@dataclass
class KVCache:
    k: Tensor
    v: Tensor
    seq_len: int = 0

    def update(self, new_k: Tensor, new_v: Tensor) -> Tuple[Tensor, Tensor]:
        batch: int = new_k.shape[0]
        end: int = self.seq_len + new_k.shape[1]
        self.k[:batch, self.seq_len:end] = new_k
        self.v[:batch, self.seq_len:end] = new_v
        self.seq_len = end
        return self.k[:batch, :end], self.v[:batch, :end]

    def get(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        return self.k[:batch_size, :self.seq_len], self.v[:batch_size, :self.seq_len]

    def reset(self) -> None:
        self.seq_len = 0


@dataclass
class DecoderLayerCache:
    self_cache: KVCache
    cross_cache: KVCache


@dataclass
class FullCache:
    layers: List[DecoderLayerCache]
    output_tokens: Tensor
    current_pos: int = 0

    def reset(self) -> None:
        self.current_pos = 0
        for layer in self.layers:
            layer.self_cache.reset()
            layer.cross_cache.reset()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps: float = eps
        self.weight: Parameter = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), _cast(self.weight, x.dtype), self.eps)


class XSwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False) -> None:
        super().__init__()
        self.gate_up_x_proj: nn.Linear = nn.Linear(dim, hidden_dim * 3, bias=bias)
        self.down_proj: nn.Linear = nn.Linear(hidden_dim, dim, bias=bias)
        self.hidden_dim: int = hidden_dim
        self.gate_up_x_proj._is_gate_up_x = True
        self.down_proj._is_down = True

    def forward(self, x: Tensor) -> Tensor:
        gux: Tensor = self.gate_up_x_proj(x)
        d: int = self.hidden_dim
        return self.down_proj(F.silu(gux.narrow(-1, 0, d)).mul_(gux.narrow(-1, d, d)).mul_(gux.narrow(-1, 2 * d, d)))


class DropBlock2D(nn.Module):
    def __init__(self, drop_prob: float = 0.1, block_size: int = 3) -> None:
        super().__init__()
        self.drop_prob: float = drop_prob
        self.block_size: int = block_size

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        _, _, height, width = x.shape
        valid_h: int = max(1, height - self.block_size + 1)
        valid_w: int = max(1, width - self.block_size + 1)
        gamma: float = (self.drop_prob / (self.block_size ** 2)) * (height * width) / (valid_h * valid_w)
        mask: Tensor = (torch.rand_like(x) < gamma).float()
        mask = F.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size // 2)
        mask = 1.0 - mask
        return x * mask * (mask.numel() / (mask.sum() + 1e-7))


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob: float = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        mask: Tensor = torch.empty(x.shape[0], 1, 1, device=x.device, dtype=x.dtype)
        mask.bernoulli_(1 - self.drop_prob).div_(1 - self.drop_prob)
        return x * mask


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    d: int = x.shape[-1] // 2
    x1: Tensor = x[..., :d]
    x2: Tensor = x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class RoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = MAX_SEQ_LEN, base: float = 40.0) -> None:
        super().__init__()
        self.dim: int = dim
        self.max_seq_len: int = max_seq_len
        inv_freq: Tensor = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        positions: Tensor = torch.arange(max_seq_len, dtype=torch.float32)
        freqs: Tensor = torch.outer(positions, inv_freq)
        self.register_buffer('cos_cached', freqs.cos().view(1, max_seq_len, 1, -1), persistent=False)
        self.register_buffer('sin_cached', freqs.sin().view(1, max_seq_len, 1, -1), persistent=False)

    def forward(self, q: Tensor, k: Tensor, offset: int = 0) -> Tuple[Tensor, Tensor]:
        length: int = q.shape[1]
        cos: Tensor = _cast(self.cos_cached[:, offset:offset + length], q.dtype)
        sin: Tensor = _cast(self.sin_cached[:, offset:offset + length], q.dtype)
        return _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)


class RoPE2DMixed(nn.Module):
    def __init__(self, head_dim: int, num_heads: int, base: float = 64.0, n_prefix: int = 0) -> None:
        super().__init__()
        self.head_dim: int = head_dim
        self.num_heads: int = num_heads
        self.n_prefix: int = n_prefix

        half_dim: int = head_dim // 2
        scale: Tensor = torch.arange(0, half_dim, dtype=torch.float32) / half_dim
        init: Tensor = 1.0 / (base ** scale)
        self.freqs_x: Parameter = nn.Parameter(init.unsqueeze(0).repeat(num_heads, 1))
        self.freqs_y: Parameter = nn.Parameter(init.unsqueeze(0).repeat(num_heads, 1))

        self.register_buffer('pos_h', torch.arange(MAX_GRID_H, dtype=torch.float32), persistent=False)
        self.register_buffer('pos_w', torch.arange(MAX_GRID_W, dtype=torch.float32), persistent=False)
        self._cache: Dict[Tuple[int, int], Tuple[Tensor, Tensor]] = {}

    def _compute(self, height: int, width: int) -> Tuple[Tensor, Tensor]:
        grid_h, grid_w = torch.meshgrid(self.pos_h[:height], self.pos_w[:width], indexing='ij')
        angles: Tensor = (
                grid_h.reshape(-1, 1, 1) * self.freqs_y.unsqueeze(0)
                + grid_w.reshape(-1, 1, 1) * self.freqs_x.unsqueeze(0)
        )
        if self.n_prefix > 0:
            angles = torch.cat([angles.new_zeros(self.n_prefix, self.num_heads, angles.shape[-1]), angles], dim=0)
        return angles.cos().unsqueeze(0), angles.sin().unsqueeze(0)

    @torch.no_grad()
    def precompute(self) -> None:
        self._cache = {grid: self._compute(*grid) for grid in BUCKET_GRIDS}

    def cos_sin(self, height: int, width: int, dtype: torch.dtype, device: torch.device) -> Tuple[Tensor, Tensor]:
        entry = self._cache.get((height, width))
        if entry is not None and not self.training and entry[0].device == device:
            return _cast(entry[0], dtype), _cast(entry[1], dtype)
        cos, sin = self._compute(height, width)
        return _cast(cos, dtype), _cast(sin, dtype)

    def forward(self, q: Tensor, k: Tensor, height: int, width: int) -> Tuple[Tensor, Tensor]:
        cos, sin = self.cos_sin(height, width, q.dtype, q.device)
        return _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)

    def rotate_keys(self, k: Tensor, height: int, width: int) -> Tensor:
        cos, sin = self.cos_sin(height, width, k.dtype, k.device)
        return _apply_rope(k, cos, sin)


class RoPE2DSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, bias: bool = False,
                 qk_norm: bool = True, attn_dropout: Optional[float] = None,
                 resid_dropout: Optional[float] = None, n_prefix: int = 0,
                 value_residual: bool = False, rope_base: float = 64.0) -> None:
        super().__init__()
        self.num_heads: int = num_heads
        self.head_dim: int = dim // num_heads
        self.qk_norm: bool = qk_norm
        self.value_residual: bool = value_residual
        self.dropout_p: float = attn_dropout if attn_dropout is not None else dropout
        resid_p: float = resid_dropout if resid_dropout is not None else dropout

        self.qkv_proj: nn.Linear = nn.Linear(dim, 3 * dim, bias=bias)
        self.out_proj: nn.Linear = nn.Linear(dim, dim, bias=bias)
        self.rope_2d: RoPE2DMixed = RoPE2DMixed(self.head_dim, num_heads, rope_base, n_prefix)

        if qk_norm:
            self.q_norm: RMSNorm = RMSNorm(self.head_dim)
            self.k_norm: RMSNorm = RMSNorm(self.head_dim)
        if value_residual:
            self.v_gate: nn.Linear = nn.Linear(dim, num_heads, bias=False)
            self.v_res_norm: RMSNorm = RMSNorm(self.head_dim)

        self.resid_dropout: nn.Module = nn.Dropout(resid_p) if resid_p > 0 else nn.Identity()

    def forward(self, x: Tensor, height: int, width: int,
                v_first: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        batch, length, dim = x.shape
        qkv: Tensor = self.qkv_proj(x).view(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        v_source: Tensor = v

        if self.value_residual and v_first is not None:
            alpha: Tensor = torch.tanh(self.v_gate(x)).unsqueeze(-1)
            v = v + alpha * self.v_res_norm(v_first)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k = self.rope_2d(q, k, height, width)

        if FLASH_AVAILABLE and q.dtype in (torch.float16, torch.bfloat16):
            attn: Tensor = flash_attn_func(
                q, k, v, dropout_p=self.dropout_p if self.training else 0.0, causal=False,
            )
        else:
            attn = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                dropout_p=self.dropout_p if self.training else 0.0, is_causal=False,
            ).transpose(1, 2)
        return self.resid_dropout(self.out_proj(attn.reshape(batch, length, dim))), v_source


class RoPESelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int = MAX_SEQ_LEN, dropout: float = 0.0,
                 bias: bool = False, qk_norm: bool = True, attn_dropout: Optional[float] = None,
                 resid_dropout: Optional[float] = None) -> None:
        super().__init__()
        self.num_heads: int = num_heads
        self.head_dim: int = dim // num_heads
        self.qk_norm: bool = qk_norm
        self.max_seq_len: int = max_seq_len
        self.dropout_p: float = attn_dropout if attn_dropout is not None else dropout
        resid_p: float = resid_dropout if resid_dropout is not None else dropout

        self.qkv_proj: nn.Linear = nn.Linear(dim, 3 * dim, bias=bias)
        self.out_proj: nn.Linear = nn.Linear(dim, dim, bias=bias)
        self.rope: RoPE = RoPE(self.head_dim, max_seq_len, base=max_seq_len * 4.0)

        if qk_norm:
            self.q_norm: RMSNorm = RMSNorm(self.head_dim)
            self.k_norm: RMSNorm = RMSNorm(self.head_dim)
        self.resid_dropout: nn.Module = nn.Dropout(resid_p) if resid_p > 0 else nn.Identity()

    def _project(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        batch, length, _ = x.shape
        qkv: Tensor = self.qkv_proj(x).view(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        return q, k, v

    def forward(self, x: Tensor, is_causal: bool = True) -> Tensor:
        batch, length, dim = x.shape
        q, k, v = self._project(x)
        q, k = self.rope(q, k)

        if FLASH_AVAILABLE and q.dtype in (torch.float16, torch.bfloat16):
            attn: Tensor = flash_attn_func(
                q, k, v, dropout_p=self.dropout_p if self.training else 0.0, causal=is_causal,
            )
        else:
            attn = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                dropout_p=self.dropout_p if self.training else 0.0, is_causal=is_causal,
            ).transpose(1, 2)
        return self.resid_dropout(self.out_proj(attn.reshape(batch, length, dim)))

    def forward_cached(self, x: Tensor, cache: KVCache) -> Tensor:
        batch, length, dim = x.shape
        q, k, v = self._project(x)
        q, k = self.rope(q, k, offset=cache.seq_len)
        k, v = cache.update(_cast(k, cache.k.dtype), _cast(v, cache.v.dtype))
        k = _cast(k, q.dtype)
        v = _cast(v, q.dtype)

        if FLASH_AVAILABLE and q.dtype in (torch.float16, torch.bfloat16):
            attn: Tensor = flash_attn_func(q, k, v, causal=False)
        else:
            attn = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False,
            ).transpose(1, 2)
        return self.out_proj(attn.reshape(batch, length, dim))

    def create_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> KVCache:
        shape: Tuple[int, int, int, int] = (batch_size, self.max_seq_len, self.num_heads, self.head_dim)
        return KVCache(k=torch.zeros(shape, device=device, dtype=dtype),
                       v=torch.zeros(shape, device=device, dtype=dtype))


class CrossAttention2D(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_ctx_len: int, dropout: float = 0.0, bias: bool = False,
                 qk_norm: bool = True, attn_dropout: Optional[float] = None,
                 resid_dropout: Optional[float] = None, n_prefix: int = 0,
                 rope_base: float = 64.0) -> None:
        super().__init__()
        self.num_heads: int = num_heads
        self.head_dim: int = dim // num_heads
        self.qk_norm: bool = qk_norm
        self.max_ctx_len: int = max_ctx_len
        self.dropout_p: float = attn_dropout if attn_dropout is not None else dropout
        resid_p: float = resid_dropout if resid_dropout is not None else dropout

        self.q_proj: nn.Linear = nn.Linear(dim, dim, bias=bias)
        self.kv_proj: nn.Linear = nn.Linear(dim, 2 * dim, bias=bias)
        self.out_proj: nn.Linear = nn.Linear(dim, dim, bias=bias)
        self.rope_2d: RoPE2DMixed = RoPE2DMixed(self.head_dim, num_heads, rope_base, n_prefix)

        if qk_norm:
            self.q_norm: RMSNorm = RMSNorm(self.head_dim)
            self.k_norm: RMSNorm = RMSNorm(self.head_dim)
        self.resid_dropout: nn.Module = nn.Dropout(resid_p) if resid_p > 0 else nn.Identity()

    def _keys_values(self, context: Tensor, height: int, width: int) -> Tuple[Tensor, Tensor]:
        batch, length, _ = context.shape
        kv: Tensor = self.kv_proj(context).view(batch, length, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        if self.qk_norm:
            k = self.k_norm(k)
        return self.rope_2d.rotate_keys(k, height, width), v

    def forward(self, x: Tensor, context: Tensor, height: int, width: int,
                need_weights: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        batch, length, dim = x.shape
        q: Tensor = self.q_proj(x).view(batch, length, self.num_heads, self.head_dim)
        if self.qk_norm:
            q = self.q_norm(q)
        k, v = self._keys_values(context, height, width)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if need_weights:
            scores: Tensor = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            weights: Tensor = scores.softmax(dim=-1)
            attn: Tensor = torch.matmul(weights, v).transpose(1, 2)
            return self.resid_dropout(self.out_proj(attn.reshape(batch, length, dim))), weights.mean(dim=1)

        if FLASH_AVAILABLE and q.dtype in (torch.float16, torch.bfloat16):
            attn = flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                dropout_p=self.dropout_p if self.training else 0.0, causal=False,
            )
        else:
            attn = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout_p if self.training else 0.0, is_causal=False,
            ).transpose(1, 2)
        return self.resid_dropout(self.out_proj(attn.reshape(batch, length, dim))), None

    def forward_cached(self, x: Tensor, cache: KVCache) -> Tensor:
        batch, length, dim = x.shape
        q: Tensor = self.q_proj(x).view(batch, length, self.num_heads, self.head_dim)
        if self.qk_norm:
            q = self.q_norm(q)
        k, v = cache.get(batch)
        k = _cast(k, q.dtype)
        v = _cast(v, q.dtype)

        if FLASH_AVAILABLE and q.dtype in (torch.float16, torch.bfloat16):
            attn: Tensor = flash_attn_func(q, k, v, causal=False)
        else:
            attn = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False,
            ).transpose(1, 2)
        return self.out_proj(attn.reshape(batch, length, dim))

    def fill_cache(self, context: Tensor, cache: KVCache, height: int, width: int) -> None:
        k, v = self._keys_values(context, height, width)
        batch, length = k.shape[0], k.shape[1]
        cache.k[:batch, :length] = _cast(k, cache.k.dtype)
        cache.v[:batch, :length] = _cast(v, cache.v.dtype)
        cache.seq_len = length

    def create_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> KVCache:
        shape: Tuple[int, int, int, int] = (batch_size, self.max_ctx_len, self.num_heads, self.head_dim)
        return KVCache(k=torch.zeros(shape, device=device, dtype=dtype),
                       v=torch.zeros(shape, device=device, dtype=dtype))


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dim: int = 128, dropout: float = 0.1,
                 dropblock_conv2: float = 0.10, dropblock_conv3: float = 0.10,
                 dropblock_size: int = 3) -> None:
        super().__init__()
        self.embed_dim: int = embed_dim
        stem_channels: List[int] = [min(32, embed_dim // 4), min(64, embed_dim // 2), embed_dim]
        groups: List[int] = [min(8, stem_channels[0] // 4), min(8, stem_channels[1] // 8),
                             min(8, stem_channels[2] // 16)]

        self.stem: nn.Sequential = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(in_channels * 4, stem_channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(groups[0], stem_channels[0]),
            nn.SiLU(),
            DropBlock2D(dropblock_conv2, dropblock_size),
            nn.Conv2d(stem_channels[0], stem_channels[1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(groups[1], stem_channels[1]),
            nn.SiLU(),
            DropBlock2D(dropblock_conv3, dropblock_size),
            nn.PixelUnshuffle(2),
            nn.Conv2d(stem_channels[1] * 4, stem_channels[2], kernel_size=1, bias=False),
            nn.GroupNorm(groups[2], stem_channels[2]),
            nn.SiLU(),
        )
        self.proj: nn.Linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm: RMSNorm = RMSNorm(embed_dim)
        self.dropout: nn.Module = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tuple[Tensor, int, int]:
        patches: Tensor = self.stem(x.to(memory_format=torch.channels_last))
        grid_h: int = patches.shape[2]
        grid_w: int = patches.shape[3]
        tokens: Tensor = patches.flatten(2).transpose(1, 2)
        return self.dropout(self.norm(self.proj(tokens))), grid_h, grid_w


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float = 0.1,
                 drop_path_rate: float = 0.0, qk_norm: bool = True, n_registers: int = 0,
                 value_residual: bool = False, rope_base: float = 64.0) -> None:
        super().__init__()
        self.norm1_in: RMSNorm = RMSNorm(dim)
        self.norm1_out: RMSNorm = RMSNorm(dim)
        self.attn: RoPE2DSelfAttention = RoPE2DSelfAttention(
            dim, num_heads, qk_norm=qk_norm, attn_dropout=0.0, resid_dropout=dropout,
            n_prefix=n_registers, value_residual=value_residual, rope_base=rope_base,
        )
        self.norm2_in: RMSNorm = RMSNorm(dim)
        self.norm2_out: RMSNorm = RMSNorm(dim)
        self.ffn: XSwiGLU = XSwiGLU(dim, int(dim * mlp_ratio))
        self.drop_path1: nn.Module = StochasticDepth(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.drop_path2: nn.Module = StochasticDepth(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.dropout: nn.Module = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor, height: int, width: int,
                v_first: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        attn_out, v_source = self.attn(self.norm1_in(x), height, width, v_first)
        x = x + self.drop_path1(self.norm1_out(attn_out))
        x = x + self.drop_path2(self.norm2_out(self.dropout(self.ffn(self.norm2_in(x)))))
        return x, v_source


class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, max_seq_len: int = MAX_SEQ_LEN,
                 max_ctx_len: int = MAX_TOKENS, mlp_ratio: float = 2.66, self_attn_dropout: float = 0.1,
                 cross_attn_dropout: float = 0.1, ffn_dropout: float = 0.1, qk_norm: bool = True,
                 n_registers: int = 0, rope_base: float = 64.0) -> None:
        super().__init__()
        self.norm1_in: RMSNorm = RMSNorm(dim)
        self.norm1_out: RMSNorm = RMSNorm(dim)
        self.self_attn: RoPESelfAttention = RoPESelfAttention(
            dim, num_heads, max_seq_len, qk_norm=qk_norm, attn_dropout=0.0, resid_dropout=self_attn_dropout,
        )
        self.norm2_in: RMSNorm = RMSNorm(dim)
        self.norm2_out: RMSNorm = RMSNorm(dim)
        self.cross_attn: CrossAttention2D = CrossAttention2D(
            dim, num_heads, max_ctx_len, qk_norm=qk_norm, attn_dropout=0.0,
            resid_dropout=cross_attn_dropout, n_prefix=n_registers, rope_base=rope_base,
        )
        self.norm3_in: RMSNorm = RMSNorm(dim)
        self.norm3_out: RMSNorm = RMSNorm(dim)
        self.ffn: XSwiGLU = XSwiGLU(dim, int(dim * mlp_ratio))
        self.dropout: nn.Module = nn.Dropout(ffn_dropout) if ffn_dropout > 0 else nn.Identity()

    def forward(self, x: Tensor, memory: Tensor, height: int, width: int,
                need_weights: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        x = x + self.norm1_out(self.self_attn(self.norm1_in(x), is_causal=True))
        cross_out, weights = self.cross_attn(self.norm2_in(x), memory, height, width, need_weights)
        x = x + self.norm2_out(cross_out)
        x = x + self.norm3_out(self.dropout(self.ffn(self.norm3_in(x))))
        return x, weights

    def forward_cached(self, x: Tensor, layer_cache: DecoderLayerCache) -> Tensor:
        x = x + self.norm1_out(self.self_attn.forward_cached(self.norm1_in(x), layer_cache.self_cache))
        x = x + self.norm2_out(self.cross_attn.forward_cached(self.norm2_in(x), layer_cache.cross_cache))
        x = x + self.norm3_out(self.ffn(self.norm3_in(x)))
        return x

    def create_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> DecoderLayerCache:
        return DecoderLayerCache(
            self_cache=self.self_attn.create_cache(batch_size, device, dtype),
            cross_cache=self.cross_attn.create_cache(batch_size, device, dtype),
        )

    def fill_cross_cache(self, memory: Tensor, layer_cache: DecoderLayerCache, height: int, width: int) -> None:
        self.cross_attn.fill_cache(memory, layer_cache.cross_cache, height, width)


class PlateOCR(nn.Module):
    def __init__(self, dim: int = 128, n_heads: int = 4, in_channels: int = 3, n_encoder_layers: int = 6,
                 n_decoder_layers: int = 2, mlp_ratio: float = 2.66, dropout: Union[float, Dict[str, float]] = 0.1,
                 drop_path_rate: float = 0.1, qk_norm: bool = True, n_registers: int = 4,
                 value_residual: bool = True, tie_embeddings: bool = True, rope_base: float = 64.0,
                 stem_dropblock: float = 0.10, bias_suppress_value: float = -100.0,
                 bias_eos_value: float = -1.0) -> None:
        super().__init__()
        self.embed_dim: int = dim
        self.mlp_ratio: float = mlp_ratio
        self.n_encoder_layers: int = n_encoder_layers
        self.n_decoder_layers: int = n_decoder_layers
        self.n_registers: int = n_registers
        self.vocab_size: int = num_classes
        self.max_seq_len: int = MAX_SEQ_LEN
        self.tie_embeddings: bool = tie_embeddings
        self.bias_suppress_value: float = bias_suppress_value
        self.bias_eos_value: float = bias_eos_value
        self.max_ctx_len: int = MAX_TOKENS + n_registers

        if isinstance(dropout, dict):
            d_encoder: float = dropout.get('encoder', 0.1)
            d_self: float = dropout.get('decoder_self_attn', 0.1)
            d_cross: float = dropout.get('decoder_cross_attn', 0.1)
            d_ffn: float = dropout.get('decoder_ffn', 0.1)
            d_embed: float = dropout.get('embed', 0.1)
            d_patch: float = dropout.get('patch_embed', 0.1)
        else:
            d_encoder = d_self = d_cross = d_ffn = d_embed = d_patch = dropout

        self.patch_embed: PatchEmbedding = PatchEmbedding(
            in_channels, dim, d_patch, stem_dropblock, stem_dropblock,
        )
        self.register_tokens: Optional[Parameter] = (
            nn.Parameter(torch.zeros(1, n_registers, dim)) if n_registers > 0 else None
        )

        encoder_dpr: List[float] = [
            drop_path_rate * (i / max(1, n_encoder_layers - 1)) ** 2 for i in range(n_encoder_layers)
        ]
        self.encoder_layers: nn.ModuleList = nn.ModuleList([
            TransformerEncoderLayer(
                dim, n_heads, mlp_ratio, d_encoder, encoder_dpr[i], qk_norm, n_registers,
                value_residual and i > 0, rope_base,
            )
            for i in range(n_encoder_layers)
        ])
        self.encoder_norm: RMSNorm = RMSNorm(dim)

        self.text_embed: nn.Embedding = nn.Embedding(num_classes, dim, padding_idx=PAD_IDX)
        self.text_scale: Parameter = nn.Parameter(torch.tensor(6.8))
        self.embed_dropout: nn.Module = nn.Dropout(d_embed) if d_embed > 0 else nn.Identity()

        self.decoder_layers: nn.ModuleList = nn.ModuleList([
            TransformerDecoderLayer(
                dim, n_heads, MAX_SEQ_LEN, self.max_ctx_len, mlp_ratio, d_self, d_cross, d_ffn,
                qk_norm, n_registers, rope_base,
            )
            for _ in range(n_decoder_layers)
        ])
        self.decoder_norm: RMSNorm = RMSNorm(dim)
        self.output_proj: nn.Linear = nn.Linear(dim, num_classes, bias=True)

        self.glyph_head: nn.Linear = nn.Linear(dim, 4, bias=True)
        self.type_head: nn.Linear = nn.Linear(dim, num_subtypes, bias=True)
        self.corner_head: nn.Linear = nn.Linear(dim, 8, bias=True)
        self.summary_norm: RMSNorm = RMSNorm(dim)

        self.register_buffer('position_masks', build_position_masks(), persistent=False)
        self.register_buffer('union_mask', build_union_position_mask(), persistent=False)
        region_pos, region_max_len = build_region_gate()
        self.register_buffer('region_pos', region_pos, persistent=False)
        self.register_buffer('region_max_len', region_max_len, persistent=False)
        self.register_buffer('region3_ids', torch.tensor(REGION3_PREFIX_IDS, dtype=torch.long), persistent=False)

        self._init_weights()
        if tie_embeddings:
            self.output_proj.weight = self.text_embed.weight

    def _init_weights(self) -> None:
        embed_std: float = 0.02
        ffn_dim: int = int(self.embed_dim * self.mlp_ratio)
        encoder_scale: float = (2.0 * self.n_encoder_layers) ** -0.5
        decoder_scale: float = (2.0 * self.n_encoder_layers + 3.0 * self.n_decoder_layers) ** -0.5
        gate_up_std: float = 0.5 * (2.0 / self.embed_dim) ** 0.5
        down_std: float = (2.0 / ffn_dim) ** 0.5
        qkv_std: float = self.embed_dim ** -0.5
        proj_std: float = (self.embed_dim * 2.0) ** -0.5
        output_std: float = self.embed_dim ** -0.5
        stem_scale: float = (2.0 * 3) ** -0.5

        head_modules = (self.glyph_head, self.type_head, self.corner_head)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if module is self.output_proj:
                    nn.init.normal_(module.weight, mean=0.0, std=output_std)
                    nn.init.zeros_(module.bias)
                    nn.init.constant_(module.bias[EOS_IDX], self.bias_eos_value)
                    nn.init.constant_(module.bias[PAD_IDX], self.bias_suppress_value)
                    nn.init.constant_(module.bias[BOS_IDX], self.bias_suppress_value)
                elif module in head_modules:
                    nn.init.trunc_normal_(module.weight, mean=0.0, std=0.01)
                    nn.init.zeros_(module.bias)
                elif hasattr(module, '_is_gate_up_x'):
                    nn.init.trunc_normal_(module.weight, mean=0.0, std=gate_up_std)
                elif hasattr(module, '_is_down'):
                    scale: float = down_std * (encoder_scale if 'encoder' in name else decoder_scale)
                    nn.init.trunc_normal_(module.weight, mean=0.0, std=scale)
                elif 'v_gate' in name:
                    nn.init.zeros_(module.weight)
                elif 'qkv_proj' in name or 'q_proj' in name or 'kv_proj' in name:
                    nn.init.trunc_normal_(module.weight, mean=0.0, std=qkv_std)
                elif 'out_proj' in name:
                    scale = proj_std * (encoder_scale if 'encoder' in name else decoder_scale)
                    nn.init.trunc_normal_(module.weight, mean=0.0, std=scale)
                elif 'patch_embed.proj' in name:
                    nn.init.trunc_normal_(module.weight, mean=0.0, std=self.embed_dim ** -0.5)
                else:
                    std: float = (2.0 / (module.weight.shape[0] + module.weight.shape[1])) ** 0.5
                    nn.init.trunc_normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None and module is not self.output_proj and module not in head_modules:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=embed_std)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
            elif isinstance(module, nn.Conv2d):
                if 'patch_embed.stem' in name:
                    fan_in: int = module.weight.shape[1] * module.weight.shape[2] * module.weight.shape[3]
                    std = stem_scale * (2.0 / (fan_in * 0.84)) ** 0.5
                else:
                    fan_out: int = (module.kernel_size[0] * module.kernel_size[1]
                                    * module.out_channels // max(1, module.groups))
                    std = (2.0 / fan_out) ** 0.5
                nn.init.trunc_normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

        if self.register_tokens is not None:
            nn.init.trunc_normal_(self.register_tokens, mean=0.0, std=embed_std)

    def encode(self, images: Tensor) -> Tuple[Tensor, Tensor, int, int]:
        features, grid_h, grid_w = self.patch_embed(images)
        if self.register_tokens is not None:
            registers: Tensor = self.register_tokens.expand(features.shape[0], -1, -1)
            features = torch.cat([_cast(registers, features.dtype), features], dim=1)

        v_first: Optional[Tensor] = None
        for layer in self.encoder_layers:
            features, v_source = layer(features, grid_h, grid_w, v_first)
            if v_first is None:
                v_first = v_source

        encoded: Tensor = self.encoder_norm(features)
        summary: Tensor = self.summary_norm(
            encoded[:, :self.n_registers].mean(dim=1) if self.n_registers > 0 else encoded.mean(dim=1)
        )
        return encoded, summary, grid_h, grid_w

    def decode(self, memory: Tensor, tokens: Tensor, grid_h: int, grid_w: int,
               need_weights: bool = False) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        x: Tensor = self.embed_dropout(self.text_embed(tokens) * self.text_scale)
        maps: List[Tensor] = []
        for layer in self.decoder_layers:
            x, weights = layer(x, memory, grid_h, grid_w, need_weights)
            if weights is not None:
                maps.append(weights)
        hidden: Tensor = self.decoder_norm(x)
        attention: Optional[Tensor] = torch.stack(maps, dim=1) if maps else None
        return self.output_proj(hidden), hidden, attention

    def forward(self, images: Tensor, tokens: Optional[Tensor] = None,
                need_weights: bool = False) -> Dict[str, Optional[Tensor]]:
        memory, summary, grid_h, grid_w = self.encode(images)
        type_logits: Tensor = self.type_head(summary)

        if tokens is None:
            return self.predict(memory, summary, type_logits, grid_h, grid_w)

        logits, hidden, attention = self.decode(memory, tokens, grid_h, grid_w, need_weights)
        return {
            'logits': logits,
            'glyph_boxes': self.glyph_head(hidden).sigmoid(),
            'type_logits': type_logits,
            'corners': self.corner_head(summary).view(-1, 4, 2),
            'attention': attention,
            'grid': torch.tensor([grid_h, grid_w], device=images.device),
        }

    def create_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> FullCache:
        return FullCache(
            layers=[layer.create_cache(batch_size, device, dtype) for layer in self.decoder_layers],
            output_tokens=torch.zeros(batch_size, MAX_SEQ_LEN, device=device, dtype=torch.long),
        )

    def prepare_inference(self) -> None:
        for layer in self.encoder_layers:
            layer.attn.rope_2d.precompute()
        for layer in self.decoder_layers:
            layer.cross_attn.rope_2d.precompute()

    @torch.no_grad()
    def predict(self, memory: Tensor, summary: Tensor, type_logits: Tensor,
                grid_h: int, grid_w: int, cache: Optional[FullCache] = None) -> Dict[str, Tensor]:
        batch: int = memory.shape[0]
        device: torch.device = memory.device
        dtype: torch.dtype = memory.dtype

        type_probs: Tensor = type_logits.softmax(dim=-1)
        subtype: Tensor = type_probs.argmax(dim=-1)
        masks: Tensor = self.position_masks[subtype]

        if cache is None:
            cache = self.create_cache(batch, device, dtype)
        else:
            cache.reset()
        for layer, layer_cache in zip(self.decoder_layers, cache.layers):
            layer.fill_cross_cache(memory, layer_cache, grid_h, grid_w)

        eos_only_mask: Tensor = torch.zeros_like(masks[:, 0])
        eos_only_mask[:, EOS_IDX] = True
        region_pos: Tensor = self.region_pos[subtype]
        region_max: Tensor = self.region_max_len[subtype]
        region_choice: Tensor = torch.zeros(batch, dtype=torch.long, device=device)
        char_probs: Tensor = torch.ones(batch, device=device)
        finished: Tensor = torch.zeros(batch, dtype=torch.bool, device=device)
        current: Tensor = torch.full((batch, 1), BOS_IDX, device=device, dtype=torch.long)
        step_probs: Tensor = torch.zeros(batch, MAX_SEQ_LEN, masks.shape[-1],
                                         device=device, dtype=dtype)

        for step in range(MAX_SEQ_LEN):
            x: Tensor = _cast(self.text_embed(current) * self.text_scale, dtype)
            for layer, layer_cache in zip(self.decoder_layers, cache.layers):
                x = layer.forward_cached(x, layer_cache)
            logits: Tensor = self.output_proj(self.decoder_norm(x)[:, -1])

            short_region: Tensor = (region_pos >= 0) & (step == region_max - 1) & ~torch.isin(
                region_choice, self.region3_ids
            )
            step_mask: Tensor = torch.where(short_region.unsqueeze(-1), eos_only_mask, masks[:, step])

            probs: Tensor = logits.masked_fill(~step_mask, float('-inf')).softmax(dim=-1)
            step_probs[:, step] = probs
            top_prob, next_token = probs.max(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, PAD_IDX), next_token)

            char_probs = torch.where(finished | (next_token == EOS_IDX), char_probs,
                                     torch.minimum(char_probs, top_prob))
            cache.output_tokens[:, step] = next_token
            region_choice = torch.where((region_pos >= 0) & (step == region_pos), next_token, region_choice)
            finished = finished | (next_token == EOS_IDX) | (next_token == PAD_IDX)
            current = next_token.unsqueeze(1)

        subtype_prob: Tensor = type_probs.gather(1, subtype.unsqueeze(1)).squeeze(1)
        confidence: Tensor = torch.where(
            subtype == UNKNOWN_SUBTYPE_IDX, subtype_prob, subtype_prob * char_probs,
        )

        return {
            'tokens': cache.output_tokens[:, :MAX_SEQ_LEN],
            'subtype': subtype,
            'type_probs': type_probs,
            'confidence': confidence,
            'corners': self.corner_head(summary).view(-1, 4, 2),
            'probs': step_probs,
        }
