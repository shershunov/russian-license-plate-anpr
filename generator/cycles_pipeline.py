from __future__ import annotations

import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from generator.config import Config, from_dict
from generator.geometry import homography
from generator.layout import build_layout
from generator.materials import make_surface
from generator.pipeline import surface_config
from generator.renderers.cycles import Job
from generator.scene import make_plan, random_stream


def prepare(values: dict, index: int, workdir: str) -> dict:
    cv2.setNumThreads(1)
    config = from_dict(values)
    plan = make_plan(config, index)
    material = surface_config(config, plan)
    layout = build_layout(plan.identity, material)
    surface = make_surface(layout, material,
                           random_stream(config.seed, index, 300),
                           plan.difficulty, plan.weather)
    folder = Path(workdir)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / f"chars_{index:06d}.npz",
             char_ids=layout.char_ids.astype(np.uint16))
    from generator.renderers.cycles import _job_entry

    return _job_entry(Job(plan, layout, surface), folder)


def warp_characters(index: int, workdir: Path, plan, shape: tuple[int, int]) -> np.ndarray:
    payload = np.load(workdir / f"chars_{index:06d}.npz")
    char_ids = payload["char_ids"]
    height, width = char_ids.shape
    matrix = homography((width, height), np.asarray(plan.quad, np.float32))
    warped = cv2.warpPerspective(char_ids, matrix, (shape[1], shape[0]),
                                 flags=cv2.INTER_NEAREST)
    (workdir / f"chars_{index:06d}.npz").unlink(missing_ok=True)
    return warped.astype(np.uint16)


def cleanup(workdir: Path) -> None:
    shutil.rmtree(workdir, ignore_errors=True)


def batches(count: int, size: int) -> list[list[int]]:
    return [list(range(start, min(start + size, count))) for start in range(0, count, size)]


def prepare_batch(config: Config, indices: list[int], workdir: Path,
                  pool: ProcessPoolExecutor | None = None) -> list[dict]:
    values = config.to_dict()
    if pool is None:
        return [prepare(values, index, str(workdir)) for index in indices]
    return list(pool.map(prepare, [values] * len(indices), indices,
                         [str(workdir)] * len(indices), chunksize=4))


def render_prepared(entries: list[dict], config: Config, workdir: Path) -> dict[int, str]:
    return {} if not entries else _render(entries, config, workdir)


def _blender_environment(executable: Path, slot: int, config: Config) -> dict[str, str]:
    import os

    environment = dict(os.environ)
    extra = executable.resolve().parents[1] / "syslibs" / "pkg" / "usr" / "lib" / "x86_64-linux-gnu"
    if extra.is_dir():
        previous = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = f"{extra}:{previous}" if previous else str(extra)
    if config.render.device != "CPU" and config.render.gpus > 1:
        environment["CUDA_VISIBLE_DEVICES"] = str(slot % config.render.gpus)
    return environment


def _render(entries: list[dict], config: Config, workdir: Path) -> dict[int, str]:
    import json
    import os
    import subprocess

    from generator.renderers.cycles import SCRIPT, find_blender

    slots = max(1, min(config.render.processes, len(entries)))
    settings = {
        "samples": config.render.samples,
        "denoise": config.render.denoise,
        "device": config.render.device,
        "supersampling": config.render.supersampling,
        "threads": max(1, (os.cpu_count() or 8) // slots),
    }
    executable = find_blender(config.render.blender)
    running: list[tuple[subprocess.Popen, Path, Path]] = []
    for slot in range(slots):
        shard = entries[slot::slots]
        if not shard:
            continue
        manifest = workdir / f"job_{slot}.json"
        manifest.write_text(json.dumps({"render": settings, "jobs": shard}), encoding="utf-8")
        log = workdir / f"job_{slot}.log"
        handle = log.open("wb")
        command = [str(executable), "--background", "--factory-startup", "--python", str(SCRIPT),
                   "--", str(manifest)]
        running.append((subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                         env=_blender_environment(executable, slot, config)),
                        manifest, log))
        handle.close()
    verbose = bool(os.environ.get("PLATE_PROFILE"))
    for process, manifest, log in running:
        code = process.wait(timeout=config.render.timeout_seconds)
        manifest.unlink(missing_ok=True)
        output = log.read_text("utf-8", "replace")
        log.unlink(missing_ok=True)
        if code != 0:
            tail = "\n".join(output.strip().splitlines()[-30:])
            raise RuntimeError(f"Blender failed ({code}):\n{tail}")
        if verbose:
            for line in output.splitlines():
                if line.startswith(("PLATE_PROFILE", "DEBUG_")):
                    print(line, flush=True)
    outputs: dict[int, str] = {}
    for entry in entries:
        path = Path(entry["output"])
        if not path.is_file():
            raise RuntimeError(f"Blender produced no frame for index {entry['index']}")
        outputs[entry["index"]] = str(path)
        Path(entry["surface"]).unlink(missing_ok=True)
    return outputs
