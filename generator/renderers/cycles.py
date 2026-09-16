from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from generator.config import Config
from generator.layout import Layout
from generator.materials import Surface
from generator.scene import FramePlan

SCRIPT = Path(__file__).with_name("blender_scene.py")


def find_blender(configured: str = "") -> Path:
    if configured:
        path = Path(configured)
        if not path.is_file():
            raise FileNotFoundError(f"Blender executable not found: {path}")
        return path
    root = Path(__file__).resolve().parents[2] / ".tools"
    candidates = sorted(root.glob("blender-*/blender.exe")) + sorted(root.glob("blender-*/blender"))
    if not candidates:
        raise FileNotFoundError(
            "Blender not found; run python -m generator.setup --blender or set render.blender")
    return candidates[-1]


@dataclass
class Job:
    plan: FramePlan
    layout: Layout
    surface: Surface


def _surface_payload(surface: Surface) -> dict[str, np.ndarray]:
    return {
        "albedo": surface.albedo.astype(np.float32),
        "height_mm": surface.height_mm.astype(np.float32),
        "roughness": surface.roughness.astype(np.float32),
        "metallic": surface.metallic.astype(np.float32),
        "alpha": surface.alpha.astype(np.float32),
        "nir": surface.nir.astype(np.float32),
    }


def _job_entry(job: Job, workdir: Path) -> dict:
    plan = job.plan
    surface_path = workdir / f"surface_{plan.index:06d}.npz"
    np.savez(surface_path, **_surface_payload(job.surface))
    retro_gain = 1.0 if job.surface.parameters["retroreflective"] else 0.25
    return {
        "index": plan.index,
        "seed": (plan.seed + plan.index * 7919) % (2 ** 31 - 1),
        "surface": str(surface_path),
        "output": str(workdir / f"frame_{plan.index:06d}.npy"),
        "size_mm": [job.layout.width_mm, job.layout.height_mm],
        "width": plan.render_size[0], "height": plan.render_size[1],
        "focal_px": plan.focal_px, "depth": plan.depth_mm / 1000.0,
        "centre": [float(plan.centre[0]), float(plan.centre[1])],
        "sensor": list(plan.sensor), "window": list(plan.window),
        "angles": list(plan.angles), "lighting": plan.lighting,
        "light_direction": list(plan.light_direction),
        "background_color": list(plan.background_color),
        "is_vehicle": bool(plan.is_vehicle), "holder": bool(plan.holder),
        "parameters": {key: (float(value) if isinstance(value, (int, float)) else value)
                       for key, value in job.surface.parameters.items()},
        "retro_gain": retro_gain,
    }


def render_batch(jobs: list[Job], config: Config, workdir: Path) -> dict[int, np.ndarray]:
    if not jobs:
        return {}
    workdir.mkdir(parents=True, exist_ok=True)
    entries = [_job_entry(job, workdir) for job in jobs]
    payload = {
        "render": {
            "samples": config.render.samples,
            "denoise": config.render.denoise,
            "device": config.render.device,
            "supersampling": config.render.supersampling,
            "threads": max(1, config.workers),
        },
        "jobs": entries,
    }
    manifest = workdir / "job.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    executable = find_blender(config.render.blender)
    command = [str(executable), "--background", "--factory-startup", "--python", str(SCRIPT),
               "--", str(manifest)]
    completed = subprocess.run(command, capture_output=True, text=True,
                               timeout=config.render.timeout_seconds)
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.strip().splitlines()[-25:])
        raise RuntimeError(f"Blender failed ({completed.returncode}):\n{tail}\n{completed.stderr}")
    frames: dict[int, np.ndarray] = {}
    for entry in entries:
        path = Path(entry["output"])
        if not path.is_file():
            raise RuntimeError(f"Blender produced no frame for index {entry['index']}")
        frames[entry["index"]] = np.load(path).astype(np.float32)
        path.unlink(missing_ok=True)
        Path(entry["surface"]).unlink(missing_ok=True)
    manifest.unlink(missing_ok=True)
    return frames


def temporary_workdir() -> Path:
    return Path(tempfile.mkdtemp(prefix="plate_cycles_"))
