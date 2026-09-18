from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from generator.config import Config, from_dict
from generator.layout import build_layout
from generator.materials import make_surface
from generator.pipeline import GlyphContext, surface_config
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
    stencil = folder / f"chars_{index:06d}.npz"
    np.savez(stencil, char_ids=layout.char_ids.astype(np.uint16), alpha=layout.alpha)
    from generator.renderers.cycles import _job_entry

    return {
        "job": _job_entry(Job(plan, layout, surface), folder),
        "finish": {
            "stencil": str(stencil),
            "context": GlyphContext.of(layout),
            "parameters": surface.parameters,
        },
    }


def cleanup(workdir: Path) -> None:
    shutil.rmtree(workdir, ignore_errors=True)


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

    slots = max(1, min(max(config.render.processes, config.render.gpus), len(entries)))
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
    transcript: list[str] = []
    for process, manifest, log in running:
        code = process.wait(timeout=config.render.timeout_seconds)
        manifest.unlink(missing_ok=True)
        output = log.read_text("utf-8", "replace")
        log.unlink(missing_ok=True)
        transcript.extend(output.strip().splitlines()[-30:])
        if code != 0:
            raise RuntimeError(f"Blender failed ({code}):\n" + "\n".join(transcript))
        if verbose:
            for line in output.splitlines():
                if line.startswith(("PLATE_PROFILE", "DEBUG_")):
                    print(line, flush=True)
    outputs: dict[int, str] = {}
    for entry in entries:
        path = Path(entry["output"])
        if not path.is_file():
            raise RuntimeError(f"Blender produced no frame for index {entry['index']}:\n"
                               + "\n".join(transcript))
        outputs[entry["index"]] = str(path)
        Path(entry["surface"]).unlink(missing_ok=True)
    return outputs
