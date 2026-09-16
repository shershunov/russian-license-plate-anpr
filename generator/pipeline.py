from __future__ import annotations

import os
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from generator.camera import Optics, capture
from generator.catalog import CATALOG
from generator.config import Config
from generator.export import (
    DatasetWriter,
    Record,
    conditions_for,
    plate_type_of,
    quad_bbox,
    visible_fraction,
)
from generator.geometry import homography, perimeter, transform_points
from generator.layout import Layout, build_layout
from generator.materials import make_surface
from generator.renderers.cpu import render_frame
from generator.scene import FramePlan, make_plan, random_stream

OVERSHOOT_LIMIT = 2.5
READABLE_MIN_HEIGHT = 6.5
READABLE_MIN_AREA = 26.0
READABLE_MIN_CONTRAST = 16.0
FRAME_MIN_READABLE = 0.34
CONTOUR_TOLERANCE_PX = 0.3


def _project(points: np.ndarray, matrix: np.ndarray, optics: Optics,
             offset: float) -> np.ndarray:
    return optics.forward(transform_points(points, matrix)) - offset


def _simplify(points: np.ndarray) -> list[list[float]]:
    if len(points) < 8:
        return points.round(1).tolist()
    reduced = cv2.approxPolyDP(points.astype(np.float32).reshape(-1, 1, 2),
                               CONTOUR_TOLERANCE_PX, True)
    return reduced.reshape(-1, 2).round(1).tolist()


def _label_means(labels: np.ndarray, gray: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    flat = labels.ravel()
    count = np.bincount(flat, minlength=size)[:size].astype(np.float32)
    total = np.bincount(flat, weights=gray.ravel(), minlength=size)[:size].astype(np.float32)
    return total, count


def glyph_contrasts(image: np.ndarray, characters: np.ndarray, plate_px: float) -> np.ndarray:
    size = int(characters.max()) + 1
    if size < 2:
        return np.zeros(1, np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    labels = characters.astype(np.uint16)
    stroke = max(1.0, plate_px / 52.0)
    gap = 2 * max(1, int(round(0.7 * stroke))) + 1
    span = gap + 2 * max(1, int(round(1.3 * stroke)))
    near = cv2.dilate(labels, np.ones((gap, gap), np.uint8))
    far = cv2.dilate(labels, np.ones((span, span), np.uint8))
    ring = np.where((labels == 0) & (near == 0), far, 0).astype(np.intp)
    erosion = 2 * max(1, int(round(0.35 * stroke))) + 1
    eroded = cv2.erode((labels > 0).astype(np.uint8), np.ones((erosion, erosion), np.uint8))
    core = np.where(eroded > 0, labels, 0).astype(np.intp)
    whole = labels.astype(np.intp)
    core_total, core_count = _label_means(core, gray, size)
    whole_total, whole_count = _label_means(whole, gray, size)
    ring_total, ring_count = _label_means(ring, gray, size)
    ink = np.where(core_count >= 4, core_total / np.maximum(core_count, 1),
                   whole_total / np.maximum(whole_count, 1))
    field = ring_total / np.maximum(ring_count, 1)
    contrast = np.abs(ink - field)
    contrast[(whole_count < 4) | (ring_count < 4)] = 0.0
    return contrast


def annotate(plan: FramePlan, layout: Layout, image: np.ndarray, characters: np.ndarray,
             optics: Optics) -> tuple[np.ndarray, list[dict], str, np.ndarray, list]:
    scale = layout.pixels_per_mm
    height_px, width_px = layout.ink.shape
    matrix = homography((width_px, height_px), np.asarray(plan.quad, np.float32))
    corners = np.array([[0, 0], [width_px - 1, 0], [width_px - 1, height_px - 1],
                        [0, height_px - 1]], np.float32)
    offset = float(plan.margin)
    quad = _project(corners, matrix, optics, offset)
    contour = _project(perimeter(width_px, height_px), matrix, optics, offset)
    outline_plate = _simplify(contour)
    glyphs: list[dict] = []
    observed = []
    contrasts = glyph_contrasts(image, characters, plan.plate_px)
    for glyph in layout.glyphs:
        x, y, w, h = glyph.box
        box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32) * scale
        projected = _project(box, matrix, optics, offset)
        visible = int((characters == glyph.index + 1).sum())
        side = float(np.linalg.norm(projected[3] - projected[0]))
        area = float(cv2.contourArea(projected.astype(np.float32)))
        contrast = float(contrasts[glyph.index + 1]) if glyph.index + 1 < len(contrasts) else 0.0
        readable = (side >= READABLE_MIN_HEIGHT and area >= READABLE_MIN_AREA
                    and visible >= 6 and contrast >= READABLE_MIN_CONTRAST)
        glyphs.append({
            "char": glyph.char, "index": glyph.index,
            "quad": projected.round(1).tolist(),
            "pixels": visible, "height_px": round(side, 2),
            "contrast": round(contrast, 1), "readable": readable,
        })
        observed.append(glyph.char if readable else "#")
    return quad, glyphs, "".join(observed), contour, outline_plate


def surface_config(config: Config, plan: FramePlan):
    spec = CATALOG[plan.identity.subtype]
    wanted = config.material.pixels_per_plate_pixel * plan.plate_px / spec.width_mm
    scale = min(config.material.pixels_per_mm,
                max(config.material.min_pixels_per_mm, wanted))
    return replace(config.material, pixels_per_mm=round(scale, 3))


def render_one(config: Config, index: int, root: Path,
               frame_path: str | None = None) -> dict | None:
    plan = make_plan(config, index)
    material = surface_config(config, plan)
    layout = build_layout(plan.identity, material)
    surface_rng = random_stream(config.seed, index, 300)
    surface = make_surface(layout, material, surface_rng, plan.difficulty, plan.weather)
    if frame_path is None:
        image, characters, info = render_frame(plan, layout, surface, config)
    else:
        image, characters, info = load_cycles_frame(plan, layout, Path(frame_path))
    camera_rng = random_stream(config.seed, index, 400)
    image, characters, optics, camera_parameters = capture(image, characters, plan,
                                                           config.camera, camera_rng)
    quad, glyphs, observed, contour, outline_plate = annotate(plan, layout, image,
                                                              characters, optics)
    height, width = image.shape[:2]
    coverage = visible_fraction(quad, width, height)
    readable_share = (sum(1 for glyph in glyphs if glyph["readable"]) / len(glyphs)
                      if glyphs else 0.0)
    slack_x, slack_y = width * 0.28, height * 0.28
    outside = (quad[:, 0].min() < -slack_x or quad[:, 0].max() > width + slack_x
               or quad[:, 1].min() < -slack_y or quad[:, 1].max() > height + slack_y)
    if coverage < 0.55 or readable_share < FRAME_MIN_READABLE or outside:
        if os.environ.get("PLATE_DEBUG_REJECTS"):
            folder = root / "rejects"
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / f"rej_{index:06d}.png"),
                        cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            np.save(folder / f"rej_{index:06d}_mask.npy", characters.astype(np.uint16))
        heights = [glyph["height_px"] for glyph in glyphs] or [0.0]
        contrasts = [glyph["contrast"] for glyph in glyphs] or [0.0]
        return {"reject": ("outside" if outside else
                           "coverage" if coverage < 0.55 else "readability"),
                "index": index, "plate_px": round(plan.plate_px, 1),
                "coverage": round(coverage, 3), "readable_share": round(readable_share, 3),
                "median_height": round(float(np.median(heights)), 1),
                "median_contrast": round(float(np.median(contrasts)), 1),
                "lighting": plan.lighting, "weather": plan.weather,
                "difficulty": round(plan.difficulty, 2)}
    writer = DatasetWriter(root, config.format, config.save_masks)
    name = f"syn_{index:06d}"
    relative = writer.write_image(name, image)
    writer.write_mask(name, characters)
    bbox = quad_bbox(quad, width, height)
    plate_type = plate_type_of(plan.identity.subtype)
    record = Record(
        image=relative,
        plate_num=observed,
        plate_type=plate_type,
        bbox=bbox,
        quad=quad.round(1).tolist(),
        is_vehicle=int(plan.is_vehicle),
        conditions=conditions_for(plan, surface.parameters, camera_parameters),
        payload={
            "image": relative, "index": index, "seed": config.seed,
            "subtype": plan.identity.subtype, "plate_type": plate_type,
            "plate_num": observed, "plate_num_true": plan.identity.text,
            "serial": plan.identity.serial, "region": plan.identity.region,
            "profile": plan.identity.profile,
            "width": width, "height": height,
            "bbox": list(bbox), "quad": quad.round(2).tolist(),
            "contour": outline_plate,
            "glyphs": glyphs,
            "plate_px": round(plan.plate_px, 2), "angles": [round(a, 2) for a in plan.angles],
            "camera": plan.camera_profile, "lighting": plan.lighting, "weather": plan.weather,
            "difficulty": plan.difficulty, "is_vehicle": bool(plan.is_vehicle),
            "holder": bool(plan.holder),
            "material": {key: (round(value, 4) if isinstance(value, float) else value)
                         for key, value in surface.parameters.items()},
            "camera_parameters": camera_parameters,
            "renderer": info.get("backend", "cpu"),
            "coverage": round(coverage, 4), "readable_share": round(readable_share, 3),
        },
    )
    writer.write_label(name, [record.yolo_row(width, height)])
    return {"record": record.__dict__, "csv": record.csv_row(),
            "yolo": record.yolo_row(width, height), "payload": record.payload}


def load_cycles_frame(plan: FramePlan, layout: Layout, path: Path
                      ) -> tuple[np.ndarray, np.ndarray, dict]:
    frame = np.load(path).astype(np.float32)
    path.unlink(missing_ok=True)
    render_width, render_height = plan.render_size
    if frame.shape[0] != render_height or frame.shape[1] != render_width:
        frame = cv2.resize(frame, (render_width, render_height), interpolation=cv2.INTER_AREA)
    height_px, width_px = layout.char_ids.shape
    matrix = homography((width_px, height_px), np.asarray(plan.quad, np.float32))
    characters = cv2.warpPerspective(layout.char_ids, matrix, (render_width, render_height),
                                     flags=cv2.INTER_NEAREST)
    alpha = cv2.warpPerspective(layout.alpha, matrix, (render_width, render_height),
                                flags=cv2.INTER_LINEAR)
    characters = np.where(alpha > 0.5, characters, 0).astype(np.uint16)
    return frame, characters, {"backend": "cycles"}


def _worker(args: tuple) -> dict | None:
    from generator.config import from_dict

    cv2.setNumThreads(1)
    values, index, root = args[0], args[1], args[2]
    frame_path = args[3] if len(args) > 3 else None
    return render_one(from_dict(values), index, Path(root), frame_path)


def drop_extra(root: Path, results: list[dict], image_format: str) -> None:
    keep = {item["payload"]["image"] for item in results}
    folder = root / "images" / "synthetic"
    labels = root / "labels"
    for path in folder.glob(f"*.{image_format}"):
        relative = path.relative_to(root).as_posix()
        if relative not in keep:
            path.unlink(missing_ok=True)
            (labels / f"{path.stem}.txt").unlink(missing_ok=True)
            (root / "masks" / f"{path.stem}.png").unlink(missing_ok=True)


def reject_summary(rejects: list[dict]) -> dict:
    if not rejects:
        return {}
    reasons = Counter(item["reject"] for item in rejects)
    readability = [item for item in rejects if item["reject"] == "readability"]
    summary: dict = {"reasons": dict(reasons.most_common())}
    if readability:
        summary["readability"] = {
            "median_plate_px": round(float(np.median([r["plate_px"] for r in readability])), 1),
            "median_glyph_px": round(float(np.median([r["median_height"] for r in readability])), 1),
            "median_contrast": round(float(np.median([r["median_contrast"] for r in readability])), 1),
            "median_readable_share": round(float(np.median([r["readable_share"]
                                                            for r in readability])), 2),
            "lighting": dict(Counter(r["lighting"] for r in readability).most_common(5)),
        }
    return summary


def summarise(results: list[dict], skipped: int, elapsed: float, config: Config) -> dict:
    types = Counter(item["payload"]["subtype"] for item in results)
    return {
        "generated": len(results), "skipped": skipped, "seconds": round(elapsed, 2),
        "images_per_second": round(len(results) / max(elapsed, 1e-6), 2),
        "backend": config.render.backend, "workers": config.workers,
        "types": dict(sorted(types.items())),
        "plate_types": dict(Counter(item["payload"]["plate_type"] for item in results)),
        "unreadable_chars": sum(item["payload"]["plate_num"].count("#") for item in results),
        "total_chars": sum(len(item["payload"]["plate_num"]) for item in results),
        "regions": len({item["payload"]["region"] for item in results}),
        "unique_plates": len({item["payload"]["plate_num_true"] for item in results}),
        "conditions": dict(Counter(tag for item in results
                                   for tag in item["record"]["conditions"])),
    }


def generate_cycles(config: Config, root: Path, progress: bool = True) -> dict:
    from generator.cycles_pipeline import cleanup, prepare_batch, render_prepared

    root.mkdir(parents=True, exist_ok=True)
    writer = DatasetWriter(root, config.format, config.save_masks)
    workdir = Path(tempfile.mkdtemp(prefix="plate_cycles_"))
    started = time.perf_counter()
    values = config.to_dict()
    results: list[dict] = []
    rejects: list[dict] = []
    skipped = 0
    stage_times = {"prepare": 0.0, "render": 0.0, "finish": 0.0}
    attempts = 0
    limit = int(config.count * OVERSHOOT_LIMIT)
    try:
        pool = None if config.workers == 1 else ProcessPoolExecutor(max_workers=config.workers)
        while len(results) < config.count and attempts < limit:
            overshoot = (min(1.6, attempts / max(len(results), 1) * 1.04)
                         if results else 1.10)
            size = min(config.render.batch_size,
                       max(8, int(round((config.count - len(results)) * overshoot))))
            chunk = list(range(attempts, min(attempts + size, limit)))
            attempts += len(chunk)
            if not chunk:
                break
            mark = time.perf_counter()
            entries = prepare_batch(config, chunk, workdir, pool)
            stage_times["prepare"] += time.perf_counter() - mark
            mark = time.perf_counter()
            outputs = render_prepared(entries, config, workdir)
            stage_times["render"] += time.perf_counter() - mark
            mark = time.perf_counter()
            payload = [(values, index, str(root), outputs[index]) for index in chunk]
            if pool is None:
                finished = [_worker(item) for item in payload]
            else:
                finished = list(pool.map(_worker, payload, chunksize=4))
            stage_times["finish"] += time.perf_counter() - mark
            for outcome in finished:
                if outcome is None or "reject" in outcome:
                    skipped += 1
                    if outcome is not None:
                        rejects.append(outcome)
                else:
                    results.append(outcome)
            for index in chunk:
                (workdir / f"chars_{index:06d}.npz").unlink(missing_ok=True)
            if progress:
                print(f"  {len(results)}/{config.count} kept "
                      f"({attempts} rendered)", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
        cleanup(workdir)
    results = sorted(results, key=lambda item: item["payload"]["index"])[:config.count]
    drop_extra(root, results, config.format)
    records = [Record(**item["record"]) for item in results]
    report = summarise(results, skipped, time.perf_counter() - started, config)
    report["stages_seconds"] = {key: round(value, 1) for key, value in stage_times.items()}
    report["rendered"] = attempts
    report["rejected"] = reject_summary(rejects)
    writer.finalise(records, values, report)
    return report


def generate(config: Config, root: Path, progress: bool = True) -> dict:
    if config.render.backend == "cycles":
        return generate_cycles(config, root, progress)
    root.mkdir(parents=True, exist_ok=True)
    writer = DatasetWriter(root, config.format, config.save_masks)
    started = time.perf_counter()
    values = config.to_dict()
    results: list[dict] = []
    rejects: list[dict] = []
    skipped = 0
    limit = int(config.count * OVERSHOOT_LIMIT)
    attempts = 0
    if config.workers == 1:
        while len(results) < config.count and attempts < limit:
            outcome = render_one(config, attempts, root)
            attempts += 1
            if outcome is None or "reject" in outcome:
                skipped += 1
                if outcome is not None:
                    rejects.append(outcome)
            else:
                results.append(outcome)
            if progress and attempts % 50 == 0:
                print(f"  {len(results)}/{config.count} kept ({attempts} rendered)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as pool:
            while len(results) < config.count and attempts < limit:
                overshoot = (min(1.6, attempts / max(len(results), 1) * 1.04)
                             if results else 1.10)
                size = min(max(64, config.workers * 8),
                           max(8, int(round((config.count - len(results)) * overshoot))))
                chunk = list(range(attempts, min(attempts + size, limit)))
                attempts += len(chunk)
                futures = [pool.submit(_worker, (values, index, str(root), None))
                           for index in chunk]
                for future in as_completed(futures):
                    outcome = future.result()
                    if outcome is None or "reject" in outcome:
                        skipped += 1
                        if outcome is not None:
                            rejects.append(outcome)
                    else:
                        results.append(outcome)
                if progress:
                    print(f"  {len(results)}/{config.count} kept ({attempts} rendered)",
                          flush=True)
    results = sorted(results, key=lambda item: item["payload"]["index"])[:config.count]
    drop_extra(root, results, config.format)
    records = [Record(**item["record"]) for item in results]
    report = summarise(results, skipped, time.perf_counter() - started, config)
    report["rendered"] = attempts
    report["rejected"] = reject_summary(rejects)
    writer.finalise(records, values, report)
    return report
