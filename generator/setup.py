from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = Path(__file__).resolve().parent / "assets"
FONT_BASE = "https://raw.githubusercontent.com/stanlapru/rus-carplates-font/main"
BLENDER_VERSION = "4.5.3"
BLENDER_BASE = "https://download.blender.org/release/Blender4.5"


def download(url: str, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_suffix(destination.suffix + ".download")
    print(f"Downloading {url}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "PlateGenerator/0.1"})
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=90) as response, pending.open("wb") as target:
        while chunk := response.read(1024 * 1024):
            target.write(chunk)
            digest.update(chunk)
    pending.replace(destination)
    return digest.hexdigest()


def setup_font() -> None:
    directory = ASSETS / "fonts"
    manifest = directory / "source.json"
    if manifest.exists():
        records = json.loads(manifest.read_text(encoding="utf-8"))
        if all(
                (directory / item["file"]).is_file()
                and hashlib.sha256((directory / item["file"]).read_bytes()).hexdigest()
                == item["sha256"]
                for item in records
        ):
            print("Font assets already verified", flush=True)
            return
        raise RuntimeError("Font asset checksum mismatch; inspect existing files before replacing")
    records = []
    for filename in ("GOST-R-50577-93.ttf", "LICENSE"):
        destination = directory / filename
        if destination.exists():
            raise FileExistsError(destination)
        url = f"{FONT_BASE}/{filename}"
        sha256 = download(url, destination)
        records.append({"file": filename, "url": url, "sha256": sha256})
    manifest.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


def setup_blender() -> Path:
    directory = ROOT / ".tools"
    name = f"blender-{BLENDER_VERSION}-windows-x64"
    executable = directory / name / "blender.exe"
    if executable.is_file():
        print(f"Blender already installed: {executable}", flush=True)
        return executable
    filename = f"{name}.zip"
    archive = directory / filename
    checksums = directory / f"blender-{BLENDER_VERSION}.sha256"
    download(f"{BLENDER_BASE}/{checksums.name}", checksums)
    expected = next(
        line.split()[0]
        for line in checksums.read_text().splitlines()
        if line.rstrip().endswith(filename)
    )
    if archive.is_file():
        with archive.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
    else:
        actual = download(f"{BLENDER_BASE}/{filename}", archive)
    if actual != expected:
        raise RuntimeError(f"Blender SHA256 mismatch: {actual} != {expected}")
    resolved = directory.resolve()
    print("Verified official SHA256; extracting Blender", flush=True)
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            destination = (resolved / member.filename).resolve()
            if not destination.is_relative_to(resolved):
                raise ValueError(f"Unsafe archive path: {member.filename}")
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as incoming, destination.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
    if not executable.is_file():
        raise RuntimeError("Blender archive does not contain the expected executable")
    return executable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender", action="store_true")
    args = parser.parse_args()
    setup_font()
    if args.blender:
        print(setup_blender(), flush=True)


if __name__ == "__main__":
    main()
