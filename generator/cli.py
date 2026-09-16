from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from generator import __version__
from generator.config import Config, load_config
from generator.preview import atlas, gallery


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plate-generator",
        description="Synthetic Russian licence plate generator (GOST R 50577-2018)")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("generate", help="render a synthetic dataset")
    run.add_argument("output", type=Path)
    run.add_argument("--config", type=Path)
    run.add_argument("--count", type=int)
    run.add_argument("--seed", type=int)
    run.add_argument("--workers", type=int)
    run.add_argument("--backend", choices=("cpu", "cycles"))
    run.add_argument("--device", choices=("CPU", "CUDA", "OPTIX"))
    run.add_argument("--samples", type=int)
    run.add_argument("--batch-size", type=int)
    run.add_argument("--processes", type=int)
    run.add_argument("--gpus", type=int)
    run.add_argument("--format", choices=("png", "jpg"))
    run.add_argument("--profile", choices=("gost", "competition"))
    run.add_argument("--types", nargs="+")
    run.add_argument("--target-share", type=float)
    run.add_argument("--save-masks", action="store_true")
    run.add_argument("--preview", action="store_true")

    catalog = sub.add_parser("catalog", help="print the plate catalogue")
    catalog.add_argument("--json", action="store_true")

    sheet = sub.add_parser("atlas", help="render one plate of every type")
    sheet.add_argument("output", type=Path)
    sheet.add_argument("--profile", default="gost", choices=("gost", "competition"))
    sheet.add_argument("--region", default="77")

    preview = sub.add_parser("preview", help="build a gallery for an existing dataset")
    preview.add_argument("dataset", type=Path)
    preview.add_argument("--limit", type=int, default=96)

    validate = sub.add_parser("validate", help="check dataset contracts")
    validate.add_argument("dataset", type=Path)
    validate.add_argument("--strict", action="store_true")

    setup = sub.add_parser("setup", help="fetch external assets")
    setup.add_argument("--blender", action="store_true")
    return parser


def apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    render = config.render
    if args.backend:
        render = replace(render, backend=args.backend)
    if args.device:
        render = replace(render, device=args.device)
    if args.samples:
        render = replace(render, samples=args.samples)
    if args.batch_size:
        render = replace(render, batch_size=args.batch_size)
    if getattr(args, "processes", None):
        render = replace(render, processes=args.processes)
    if getattr(args, "gpus", None):
        render = replace(render, gpus=args.gpus)
    updates: dict = {"render": render}
    for name in ("count", "seed", "workers", "profile", "target_share"):
        value = getattr(args, name, None)
        if value is not None:
            updates[name] = value
    if args.format:
        updates["format"] = args.format
    if args.types:
        updates["types"] = tuple(args.types)
    if args.save_masks:
        updates["save_masks"] = True
    config = replace(config, **updates)
    config.validate()
    return config


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        from generator.pipeline import generate

        config = apply_overrides(load_config(args.config), args)
        report = generate(config, args.output)
        print(json.dumps(report, ensure_ascii=False, indent=1))
        if args.preview:
            print(gallery(args.output))
        return 0
    if args.command == "catalog":
        from generator.catalog import CATALOG

        if args.json:
            print(json.dumps({key: value.__dict__ for key, value in CATALOG.items()},
                             ensure_ascii=False, indent=1))
        else:
            for key, spec in CATALOG.items():
                print(f"{key:8s} {spec.plate_type:7s} "
                      f"{spec.width_mm:5.0f}x{spec.height_mm:<5.0f} {spec.pattern:8s} "
                      f"{spec.palette:7s} {spec.material:9s} {spec.title}")
        return 0
    if args.command == "atlas":
        print(atlas(args.output, args.profile, args.region))
        return 0
    if args.command == "preview":
        print(gallery(args.dataset, limit=args.limit))
        return 0
    if args.command == "validate":
        from generator.validate import check

        report = check(args.dataset, args.strict)
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return 0 if report["ok"] else 1
    if args.command == "setup":
        from generator.setup import main as setup_main

        setup_main()
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
