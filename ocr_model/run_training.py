import argparse
import datetime
import inspect
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nn.model import PlateOCR
from utils.alphabet import idx_to_subtype
from utils.dataset import (BucketBatchSampler, PhotometricAugmenter, PlateDataset, SampleTable, collate,
                           load_annotations_jsonl, load_meta_csv, load_plates_tsv, load_sharded_annotations)
from utils.ema import EMAModel, use_ema_for_eval
from utils.optimizer import MultiScheduler, build_hybrid_optimizer
from utils.train import (LossWeights, PrefetchLoader, WarmupStableDecayLR, compute_losses, evaluate,
                         train_one_epoch)

try:
    import mlflow

    MLFLOW_AVAILABLE: bool = True
except ImportError:
    MLFLOW_AVAILABLE: bool = False


@dataclass
class TrainingConfig:
    data_roots: List[str] = field(default_factory=list)
    val_roots: List[str] = field(default_factory=list)
    output_dir: str = 'checkpoints'

    num_epochs: int = 30
    warmup_epochs: int = 4
    decay_epochs: int = 15
    decay_shape: str = 'sqrt'
    batch_size: int = 256

    muon_lr: float = 5e-3
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.01
    muon_ns_steps: int = 5

    adamw_lr: float = 1e-4
    adamw_weight_decay: float = 0.1
    adamw_betas: Tuple[float, float] = (0.9, 0.999)

    eta_min: float = 1e-6
    grad_clip: float = 15.0

    dropout: Dict[str, float] = field(default_factory=lambda: {
        'encoder': 0.06,
        'decoder_self_attn': 0.05,
        'decoder_cross_attn': 0.05,
        'decoder_ffn': 0.08,
        'embed': 0.10,
        'patch_embed': 0.05,
    })
    drop_path_rate: float = 0.10
    stem_dropblock: float = 0.10

    dim: int = 128
    n_heads: int = 4
    n_encoder_layers: int = 6
    n_decoder_layers: int = 2
    mlp_ratio: float = 2.66
    n_registers: int = 4
    rope_base: float = 64.0
    value_residual: bool = True
    tie_embeddings: bool = True

    maxsup_alpha: float = 0.002
    maxsup_ramp: bool = False
    loss: LossWeights = field(default_factory=LossWeights)

    use_ema: bool = True
    ema_decay: float = 0.9995
    update_after_step: int = 0
    ema_warmup_ratio: float = 0.10
    ema_update_every: int = 10

    base_pad: float = 0.04
    jitter: float = 0.20
    val_ratio: float = 0.03
    val_limit: int = 10000
    num_workers: int = 12
    val_workers: int = 4
    seed: int = 1337

    amp_dtype: str = 'bfloat16'
    dry_run: bool = False
    bench_steps: int = 0
    continue_training: bool = False
    checkpoint_path: Optional[str] = None

    mlflow_tracking_uri: str = 'http://localhost:5089'
    mlflow_experiment: str = 'PlateOCR_Training'


def set_seed(seed: int = 1337) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def setup_distributed_env() -> None:
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12355')
    os.environ.setdefault('NCCL_TIMEOUT', '1800')

    os.environ.setdefault('NCCL_NVLS_ENABLE', '0')
    os.environ.setdefault('NCCL_P2P_LEVEL', 'SYS')
    os.environ.setdefault('NCCL_BUFS_PER_THREAD', '8')

    os.environ.setdefault('NCCL_SOCKET_NTHREADS', '8')
    os.environ.setdefault('NCCL_NSOCKS_PERTHREAD', '8')


def discover_samples(roots: List[str]) -> SampleTable:
    tables: List[SampleTable] = []
    for raw_root in roots:
        root: Path = Path(raw_root)
        if root.is_file():
            if root.suffix == '.jsonl':
                tables.append(load_annotations_jsonl(root))
            elif root.suffix == '.tsv':
                tables.append(load_plates_tsv(root))
            else:
                tables.append(load_meta_csv(root))
            continue
        if any(root.glob('shard_*')):
            tables.append(load_sharded_annotations(root))
            continue
        splits: List[Path] = [root / name for name in ('train.tsv', 'val.tsv')]
        present: List[Path] = [path for path in splits if path.exists()]
        if present:
            for path in present:
                tables.append(load_plates_tsv(path, root / path.stem / 'images'))
            continue
        annotations: Path = root / 'annotations.jsonl'
        if annotations.exists():
            tables.append(load_annotations_jsonl(annotations))
            continue
        meta: Path = root / 'meta.csv'
        if meta.exists():
            tables.append(load_meta_csv(meta, root / 'images' if (root / 'images').exists() else root))
            continue
        raise FileNotFoundError(f'no annotations.jsonl or meta.csv under {root}')
    return SampleTable.concat(tables)


def split_samples(table: SampleTable, val_ratio: float,
                  seed: int) -> Tuple[np.ndarray, np.ndarray]:
    subtypes: np.ndarray = np.asarray(table.subtype)
    rng: np.random.Generator = np.random.default_rng(seed)
    val_mask: np.ndarray = np.zeros(subtypes.shape[0], dtype=bool)
    for value in np.unique(subtypes):
        order: np.ndarray = rng.permutation(np.flatnonzero(subtypes == value))
        val_mask[order[:max(1, int(round(order.size * val_ratio)))]] = True
    return np.flatnonzero(~val_mask), np.flatnonzero(val_mask)


def subsample_stratified(table: SampleTable, indices: np.ndarray, limit: int,
                         seed: int) -> np.ndarray:
    subtypes: np.ndarray = np.asarray(table.subtype)[indices]
    rng: np.random.Generator = np.random.default_rng(seed)
    groups: np.ndarray = np.unique(subtypes)
    per_type: int = max(1, limit // max(1, int(groups.size)))
    keep: List[np.ndarray] = [rng.permutation(indices[subtypes == value])[:per_type]
                              for value in groups]
    return np.sort(np.concatenate(keep)) if keep else indices[:0]


def create_model(cfg: TrainingConfig, device: torch.device) -> PlateOCR:
    return PlateOCR(
        dim=cfg.dim,
        n_heads=cfg.n_heads,
        n_encoder_layers=cfg.n_encoder_layers,
        n_decoder_layers=cfg.n_decoder_layers,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        drop_path_rate=cfg.drop_path_rate,
        n_registers=cfg.n_registers,
        rope_base=cfg.rope_base,
        value_residual=cfg.value_residual,
        tie_embeddings=cfg.tie_embeddings,
        stem_dropblock=cfg.stem_dropblock,
    ).to(device)


def create_dataloaders(cfg: TrainingConfig, rank: int, world_size: int
                       ) -> Tuple[DataLoader, DataLoader, PlateDataset, BucketBatchSampler]:
    table: SampleTable = discover_samples(cfg.data_roots)
    if len(table) == 0:
        raise RuntimeError('dataset is empty')

    if cfg.val_roots:
        val_table: SampleTable = discover_samples(cfg.val_roots)
        held_out: set = set(val_table.paths())
        train_indices: np.ndarray = np.fromiter(
            (i for i in range(len(table)) if table.path(i) not in held_out),
            dtype=np.int64, count=-1)
        val_indices: np.ndarray = np.arange(len(val_table), dtype=np.int64)
    else:
        val_table = table
        train_indices, val_indices = split_samples(table, cfg.val_ratio, cfg.seed)

    if cfg.val_limit and val_indices.size > cfg.val_limit:
        val_indices = subsample_stratified(val_table, val_indices, cfg.val_limit, cfg.seed)

    train_dataset: PlateDataset = PlateDataset(
        table, train_indices, train=True, base_pad=cfg.base_pad, jitter=cfg.jitter,
        augmenter=PhotometricAugmenter(), seed=cfg.seed,
    )
    val_dataset: PlateDataset = PlateDataset(
        val_table, val_indices, train=False, base_pad=cfg.base_pad, seed=cfg.seed)

    train_sampler: BucketBatchSampler = BucketBatchSampler(
        train_dataset.buckets, cfg.batch_size, shuffle=True, drop_last=True,
        seed=cfg.seed, rank=rank, world_size=world_size,
    )
    val_sampler: BucketBatchSampler = BucketBatchSampler(
        val_dataset.buckets, cfg.batch_size, shuffle=False, drop_last=False,
        seed=cfg.seed, rank=rank, world_size=world_size,
    )

    train_common: Dict = {
        'collate_fn': collate,
        'num_workers': cfg.num_workers,
        'pin_memory': True,
        'persistent_workers': cfg.num_workers > 0,
    }
    if cfg.num_workers > 0:
        train_common['prefetch_factor'] = 4

    val_workers: int = min(cfg.val_workers, cfg.num_workers)
    val_common: Dict = {
        'collate_fn': collate,
        'num_workers': val_workers,
        'pin_memory': True,
        'persistent_workers': False,
    }
    if val_workers > 0:
        val_common['prefetch_factor'] = 2

    train_loader: DataLoader = DataLoader(train_dataset, batch_sampler=train_sampler, **train_common)
    val_loader: DataLoader = DataLoader(val_dataset, batch_sampler=val_sampler, **val_common)

    if rank == 0:
        counts: np.ndarray = np.bincount(np.asarray(table.subtype)[train_indices],
                                         minlength=len(idx_to_subtype))
        named: List[str] = [f'{idx_to_subtype[i]}={int(v)}'
                            for i, v in enumerate(counts) if v]
        print(f'train: {train_indices.size} samples, {len(train_sampler)} batches/rank | '
              f'val: {val_indices.size} samples, {len(val_sampler)} batches/rank | '
              f'table {table.nbytes() / 1e9:.2f} GB', flush=True)
        print('  types: ' + ' '.join(sorted(named)), flush=True)

    return train_loader, val_loader, train_dataset, train_sampler


def build_hybrid(model: torch.nn.Module, cfg: TrainingConfig, rank: int):
    optimizer, n_muon, n_decay, n_no_decay = build_hybrid_optimizer(
        model=model,
        muon_lr=cfg.muon_lr,
        muon_momentum=cfg.muon_momentum,
        muon_weight_decay=cfg.muon_weight_decay,
        muon_ns_steps=cfg.muon_ns_steps,
        adamw_lr=cfg.adamw_lr,
        adamw_betas=cfg.adamw_betas,
        adamw_weight_decay=cfg.adamw_weight_decay,
    )
    if rank == 0:
        print(f'Muon params: {n_muon:,} | AdamW decay: {n_decay:,} | AdamW no_decay: {n_no_decay:,}', flush=True)

    warmup_epochs: int = min(cfg.warmup_epochs, max(1, cfg.num_epochs // 5))
    decay_epochs: int = min(cfg.decay_epochs, max(1, cfg.num_epochs - warmup_epochs - 1))
    if rank == 0 and (warmup_epochs, decay_epochs) != (cfg.warmup_epochs, cfg.decay_epochs):
        print(f'schedule clamped to fit {cfg.num_epochs} epochs: '
              f'warmup {warmup_epochs}, stable {cfg.num_epochs - warmup_epochs - decay_epochs}, '
              f'decay {decay_epochs}', flush=True)

    muon_opt, adamw_opt = optimizer.optimizers
    scheduler: MultiScheduler = MultiScheduler([
        WarmupStableDecayLR(
            opt, warmup_epochs=warmup_epochs, total_epochs=cfg.num_epochs,
            decay_epochs=decay_epochs, eta_min=cfg.eta_min, decay_shape=cfg.decay_shape,
        )
        for opt in (muon_opt, adamw_opt)
    ])
    return optimizer, scheduler


def setup_mlflow(cfg: TrainingConfig, model: torch.nn.Module, run_id: Optional[str]) -> None:
    if not MLFLOW_AVAILABLE:
        print('mlflow is not installed, skipping experiment tracking', flush=True)
        return

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)

    run_name: str = (f'L{cfg.n_encoder_layers}/{cfg.n_decoder_layers}_H{cfg.n_heads}'
                     f'_E{cfg.dim}_R{cfg.mlp_ratio}_B{cfg.batch_size}')

    if cfg.continue_training and run_id:
        mlflow.start_run(run_id=run_id)
        print(f'resumed mlflow run: {run_id}', flush=True)
        return

    mlflow.start_run(run_name=run_name)
    print(f'new mlflow run: {mlflow.active_run().info.run_id}', flush=True)
    for artifact in ('nn/model.py', 'utils/train.py', 'utils/optimizer.py', 'utils/dataset.py'):
        if Path(artifact).exists():
            mlflow.log_artifact(artifact)

    params: Dict = asdict(cfg)
    for key, value in params.pop('dropout').items():
        params[f'dropout_{key}'] = value
    for key, value in params.pop('loss').items():
        params[f'loss_{key}'] = value
    params.pop('adamw_betas', None)
    total_params: int = sum(p.numel() for p in model.parameters())
    params.update({
        'optimizer': 'Muon+AdamW',
        'adamw_betas': f'{cfg.adamw_betas[0]},{cfg.adamw_betas[1]}',
        'total_params': total_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),
    })
    mlflow.log_params(params)


def load_checkpoint(cfg: TrainingConfig, model: torch.nn.Module, device: torch.device,
                    rank: int) -> Tuple[int, float, Optional[Dict], Optional[str], int]:
    if not cfg.continue_training or not cfg.checkpoint_path:
        return 0, float('-inf'), None, None, 0

    checkpoint: Dict = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    if rank == 0:
        print(f'loaded checkpoint {cfg.checkpoint_path} @ epoch {checkpoint["epoch"]}', flush=True)
    return (
        checkpoint['epoch'] + 1,
        checkpoint.get('best_accuracy', float('-inf')),
        checkpoint.get('ema'),
        checkpoint.get('mlflow_run_id'),
        checkpoint.get('global_step', 0),
    )


def train(rank: int, gpu_ids: List[int], num_gpus: int, cfg: TrainingConfig) -> None:
    set_seed(cfg.seed + rank)
    distributed: bool = num_gpus > 1

    gpu_id: int = gpu_ids[rank]
    device: torch.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if distributed:
        dist.init_process_group(backend='nccl', world_size=num_gpus, rank=rank,
                                timeout=datetime.timedelta(minutes=30))

    model: PlateOCR = create_model(cfg, device)
    start_epoch, best_accuracy, ema_state, mlflow_run_id, global_step = load_checkpoint(cfg, model, device, rank)

    if rank == 0:
        print(f'model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params | device {device}', flush=True)
        if not cfg.dry_run:
            setup_mlflow(cfg, model, mlflow_run_id)

    if distributed:
        ddp_options: Dict = dict(
            device_ids=[gpu_id], output_device=gpu_id, gradient_as_bucket_view=True,
            bucket_cap_mb=25, find_unused_parameters=False, static_graph=True,
        )
        if 'forward_sync_buffers' in inspect.signature(DDP.__init__).parameters:
            ddp_options['forward_sync_buffers'] = False
        else:
            ddp_options['broadcast_buffers'] = False
        model_ddp: torch.nn.Module = DDP(model, **ddp_options)
    else:
        model_ddp = model

    train_loader, val_loader, train_dataset, train_sampler = create_dataloaders(cfg, rank, num_gpus)
    optimizer, scheduler = build_hybrid(model_ddp, cfg, rank)

    planned_steps: int = cfg.num_epochs * len(train_sampler)
    ema_warmup: int = cfg.update_after_step or max(
        1, int(planned_steps / max(1, cfg.ema_update_every) * cfg.ema_warmup_ratio)
    )

    ema: Optional[EMAModel] = None
    if cfg.use_ema:
        ema = EMAModel(model_ddp, decay=cfg.ema_decay, warmup_steps=ema_warmup,
                       update_every=cfg.ema_update_every)
        if ema_state is not None:
            ema.load_state_dict(ema_state, device)
        if rank == 0:
            print(f'EMA: decay={cfg.ema_decay} update_every={cfg.ema_update_every} '
                  f'warmup={ema_warmup} updates (of {planned_steps // cfg.ema_update_every} planned)', flush=True)

    amp_dtype: Optional[torch.dtype] = {
        'bfloat16': torch.bfloat16, 'float16': torch.float16, 'none': None,
    }[cfg.amp_dtype]

    prefetch_train: PrefetchLoader = PrefetchLoader(train_loader, device)
    prefetch_val: PrefetchLoader = PrefetchLoader(val_loader, device)

    if cfg.dry_run:
        started: float = time.perf_counter()
        iterator = iter(prefetch_train)
        batch = next(iterator)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype else nullcontext():
            outputs = model_ddp(batch['image'], batch['tokens'], need_weights=True)
            parts = compute_losses(outputs, batch, cfg.loss, cfg.n_registers, cfg.maxsup_alpha)
        parts['total'].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model_ddp.parameters(), cfg.grad_clip)
        optimizer.step()
        if rank == 0:
            shapes: str = ' '.join(f'{k}{tuple(v.shape)}' for k, v in list(batch.items())[:3])
            print(f'dry-run OK in {time.perf_counter() - started:.1f}s | batch {shapes}', flush=True)
            print('  losses: ' + ' '.join(f'{k}={v.detach().item():.4f}' for k, v in parts.items()), flush=True)
            print(f'  grad_norm {grad_norm.detach().item():.3f} | steps/epoch {len(train_sampler)} '
                  f'| planned total {cfg.num_epochs * len(train_sampler)}', flush=True)
        if cfg.bench_steps > 0:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            bench_started: float = time.perf_counter()
            seen: int = 0
            for _ in range(cfg.bench_steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type,
                                        dtype=amp_dtype) if amp_dtype else nullcontext():
                    outputs = model_ddp(batch['image'], batch['tokens'], need_weights=True)
                    parts = compute_losses(outputs, batch, cfg.loss, cfg.n_registers, cfg.maxsup_alpha)
                parts['total'].backward()
                torch.nn.utils.clip_grad_norm_(model_ddp.parameters(), cfg.grad_clip)
                optimizer.step()
                seen += 1
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed: float = time.perf_counter() - bench_started
            if rank == 0 and seen:
                per_step: float = elapsed / seen
                epoch_minutes: float = per_step * len(train_sampler) / 60.0
                print(f'  bench: {seen} steps in {elapsed:.1f}s -> {per_step * 1000:.0f} ms/step, '
                      f'{cfg.batch_size * num_gpus / per_step:.0f} samples/s total, '
                      f'~{epoch_minutes:.1f} min/epoch, ~{epoch_minutes * cfg.num_epochs / 60.0:.1f} h '
                      f'for {cfg.num_epochs} epochs', flush=True)
        if rank == 0:
            print('dry run complete, training not started', flush=True)
        if distributed:
            dist.destroy_process_group()
        return

    total_steps: int = cfg.num_epochs * len(train_sampler)
    maxsup_ramp_steps: int = total_steps if cfg.maxsup_ramp else 0
    output_dir: Path = Path(cfg.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, cfg.num_epochs):
        train_dataset.set_epoch(epoch)
        train_sampler.set_epoch(epoch)
        if rank == 0:
            print('')
            print(f'Epoch {epoch + 1}/{cfg.num_epochs}', flush=True)

        stats, global_step = train_one_epoch(
            model_ddp, prefetch_train, optimizer, device, cfg.loss, cfg.n_registers, epoch, rank,
            global_step, cfg.maxsup_alpha, maxsup_ramp_steps, cfg.grad_clip, amp_dtype, distributed, ema,
        )
        scheduler.step()

        raw_result = evaluate(model_ddp, prefetch_val, device, rank, amp_dtype=amp_dtype,
                              distributed=distributed, desc='Validating')
        ema_result = None
        if ema is not None:
            with use_ema_for_eval(model_ddp, ema):
                ema_result = evaluate(model_ddp, prefetch_val, device, rank, amp_dtype=amp_dtype,
                                      distributed=distributed, desc='Validating EMA')

        report = ema_result if ema_result is not None else raw_result
        score: float = report.exact_target

        if rank == 0:
            learning_rates: List[float] = scheduler.get_last_lr()
            maxsup_term: float = cfg.maxsup_alpha * stats['maxsup_reg']
            if score > best_accuracy:
                print(f'🔥 Best target_exact improved from {best_accuracy:.4f} to {score:.4f}', flush=True)
            print(f'Train Loss: {stats["total"]:.4f}, LR muon: {learning_rates[0]:.9f}, '
                  f'adamw: {learning_rates[-1]:.9f} ({stats["seconds"]:.0f}s)', flush=True)
            print(f'CE pure: {stats["ce_pure"]:.4f} | MaxSup reg: {stats["maxsup_reg"]:.3f} | '
                  f'term: {maxsup_term:.4f} (a={cfg.maxsup_alpha:.4f}) | '
                  f'ratio: {maxsup_term / max(stats["ce_pure"], 1e-8):.3f}', flush=True)
            print(f'Aux: glyph {stats["glyph"]:.4f} | attn {stats["attention"]:.4f} | '
                  f'corner {stats["corner"]:.4f} | type {stats["subtype"]:.4f} | '
                  f'grad {stats["grad_norm"]:.2f} '
                  f'(clipped {stats["clip_rate"] * 100:.0f}%)', flush=True)
            print(f'Val: target_exact {raw_result.exact_target:.4f} | char {raw_result.char_accuracy:.4f} | '
                  f'subtype {raw_result.subtype_accuracy:.4f} | '
                  f'unknown_rej {raw_result.unknown_rejection:.4f}', flush=True)
            if ema_result is not None:
                print(f'Val EMA: target_exact {ema_result.exact_target:.4f} '
                      f'({ema_result.exact_target - raw_result.exact_target:+.4f}) | '
                      f'char {ema_result.char_accuracy:.4f} | '
                      f'subtype {ema_result.subtype_accuracy:.4f} | '
                      f'unknown_rej {ema_result.unknown_rejection:.4f}', flush=True)
            for reference, hypothesis, true_type, guess_type in report.errors[:5]:
                print(f'    {reference:>10} -> {hypothesis:<10} [{true_type} -> {guess_type}]', flush=True)

            if MLFLOW_AVAILABLE and mlflow.active_run() is not None:
                metrics: Dict[str, float] = {f'train_{k}': v for k, v in stats.items()}
                metrics.update({
                    'val_exact_target': raw_result.exact_target,
                    'val_char_accuracy': raw_result.char_accuracy,
                    'val_subtype_accuracy': raw_result.subtype_accuracy,
                    'val_unknown_rejection': raw_result.unknown_rejection,
                    'lr_muon': scheduler.get_last_lr()[0],
                    'lr_adamw': scheduler.get_last_lr()[-1],
                })
                if ema_result is not None:
                    metrics.update({
                        'val_ema_exact_target': ema_result.exact_target,
                        'val_ema_char_accuracy': ema_result.char_accuracy,
                        'val_ema_subtype_accuracy': ema_result.subtype_accuracy,
                    })
                mlflow.log_metrics(metrics, step=epoch)

            state: Dict = {
                'model': (model_ddp.module if distributed else model_ddp).state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'ema': ema.state_dict() if ema is not None else None,
                'epoch': epoch,
                'global_step': global_step,
                'best_accuracy': max(best_accuracy, score),
                'config': asdict(cfg),
                'mlflow_run_id': mlflow.active_run().info.run_id if (
                        MLFLOW_AVAILABLE and mlflow.active_run() is not None) else None,
            }
            torch.save(state, output_dir / 'last.pt')
            if score > best_accuracy:
                torch.save(state, output_dir / 'best.pt')

        best_accuracy = max(best_accuracy, score)
        if distributed:
            dist.barrier()

    if rank == 0:
        print(f'done. best target_exact {best_accuracy:.4f}', flush=True)
        if MLFLOW_AVAILABLE and mlflow.active_run() is not None:
            mlflow.end_run()
    if distributed:
        dist.destroy_process_group()


def parse_args() -> Tuple[TrainingConfig, List[int]]:
    parser = argparse.ArgumentParser(description='train license plate OCR')
    parser.add_argument('--data', nargs='+', required=True)
    parser.add_argument('--val-data', nargs='*', default=[])
    parser.add_argument('--output', default='checkpoints')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--warmup-epochs', type=int, default=4)
    parser.add_argument('--decay-epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--val-limit', type=int, default=6000)
    parser.add_argument('--gpus', default=None, help='comma-separated gpu ids, default: all visible')
    parser.add_argument('--amp', default='bfloat16', choices=('bfloat16', 'float16', 'none'))
    parser.add_argument('--dry-run', action='store_true', help='load data, run one step, exit')
    parser.add_argument('--bench-steps', type=int, default=0, help='extra timed steps during dry run')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--config', default=None)
    args = parser.parse_args()

    cfg: TrainingConfig = TrainingConfig(
        data_roots=args.data, val_roots=args.val_data, output_dir=args.output, num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs, decay_epochs=args.decay_epochs,
        batch_size=args.batch_size, num_workers=args.workers, amp_dtype=args.amp, val_limit=args.val_limit,
        continue_training=args.resume is not None, checkpoint_path=args.resume, dry_run=args.dry_run,
        bench_steps=args.bench_steps,
    )
    if args.config:
        overrides: Dict = json.loads(Path(args.config).read_text(encoding='utf-8'))
        loss_overrides: Dict = overrides.pop('loss', {})
        dropout_overrides: Dict = overrides.pop('dropout', {})
        for key, value in overrides.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        for key, value in loss_overrides.items():
            if hasattr(cfg.loss, key):
                setattr(cfg.loss, key, value)
        cfg.dropout.update(dropout_overrides)

    gpu_ids: List[int] = (
        [int(v) for v in args.gpus.split(',')] if args.gpus
        else list(range(max(1, torch.cuda.device_count())))
    )
    return cfg, gpu_ids


def main() -> None:
    cfg, gpu_ids = parse_args()
    num_gpus: int = len(gpu_ids)

    if num_gpus > 1:
        setup_distributed_env()
        mp.spawn(train, args=(gpu_ids, num_gpus, cfg), nprocs=num_gpus, join=True)
    else:
        train(0, gpu_ids, 1, cfg)


if __name__ == '__main__':
    main()
