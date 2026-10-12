#!/usr/bin/env python3
"""
Fetch and unpack the AVVG imagery that pairs with refGeo/metainfo/avvg_*.jsonl.

Why AVVG
--------
It is the only refGeo subset that is both genuinely oriented and ships its own
images (train splits):

  avvg_train       26,482 records, 97.85% oriented, off-axis angle mean 20.39 deg,
                   aspect median 2.14 with 99.6% above the 1.1 direction
                   stability threshold. Images: refGeo/images/avvg.zip, 1.03 GB.
  dior_rsvg_train  27,133 records, ids resolve 100% against the DIOR-RSVG images
                   already on disk, but 100.00% axis-aligned -- no orientation.
  geochat_train    63,883 records, oriented, images are a separate large download.
                   Pairing these records with DIOR-RSVG filenames silently
                   mispairs every target.

The container has no curl and no guarantee of unzip, so both steps go through the
standard library. Resumable via an HTTP Range request against the partial file.

    python scripts/fetch_avvg_images.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

URL = ("https://huggingface.co/datasets/erenzhou/refGeo/resolve/main/"
       "refGeo/images/avvg.zip")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or unit == "GiB":
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} GiB"


def download(url: str, target: Path, chunk: int = 1 << 20) -> bool:
    """Stream to ``target``, resuming from whatever is already there."""
    offset = target.stat().st_size if target.exists() else 0
    headers = {"User-Agent": "obb-pbd/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        print(f"  resuming from {human(offset)}", flush=True)

    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and offset:      # already complete
            print("  server reports the range is satisfied; treating as complete")
            return True
        print(f"  HTTP {exc.code}: {exc.reason}")
        return False

    total = response.headers.get("Content-Length")
    total = (int(total) + offset) if total else None
    mode = "ab" if offset and response.status == 206 else "wb"
    if mode == "wb":
        offset = 0

    written = offset
    started = time.time()
    last = 0.0
    with response, target.open(mode) as handle:
        while True:
            block = response.read(chunk)
            if not block:
                break
            handle.write(block)
            written += len(block)
            now = time.time()
            if now - last > 5.0:
                rate = (written - offset) / max(1e-6, now - started)
                share = f" ({100.0 * written / total:.1f}%)" if total else ""
                print(f"  {human(written)}{share} at {human(rate)}/s", flush=True)
                last = now

    print(f"  downloaded {human(written)} in {time.time() - started:.0f}s",
          flush=True)
    return total is None or written >= total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="/workspace/data/refgeo")
    parser.add_argument("--keep-archive", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    dest = Path(args.dest)
    images_root = dest / "images"
    images_root.mkdir(parents=True, exist_ok=True)
    archive = dest / "avvg.zip"

    existing = [p for p in images_root.rglob("*") if p.suffix in IMAGE_SUFFIXES]
    if existing and not args.force:
        print(f"  {len(existing):,} image files already under {images_root}, "
              "nothing to do (use --force to redo)")
    else:
        free = shutil.disk_usage(dest).free
        print(f"  free space: {human(free)}")
        print(f"  downloading {URL}")
        if not download(URL, archive):
            print("  download failed")
            return 1
        print(f"  archive: {human(archive.stat().st_size)}")

        print("  extracting")
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
            print(f"    {len(names):,} entries, first few: {names[:3]}")
            zf.extractall(images_root)
        print("  extracted")
        if not args.keep_archive:
            archive.unlink(missing_ok=True)
            print("  removed the archive")

    directories = sorted({p.parent for p in images_root.rglob("*")
                          if p.suffix in IMAGE_SUFFIXES})
    files = [p for p in images_root.rglob("*") if p.suffix in IMAGE_SUFFIXES]
    print()
    print(f"  image files      : {len(files):,}")
    print(f"  directories      : {[str(d.relative_to(images_root)) for d in directories[:6]]}")
    if files:
        sample = sorted(files)[:5]
        print("  sample names     :")
        for path in sample:
            print(f"    {path.relative_to(images_root)}  "
                  f"{human(path.stat().st_size)}")
    return 0 if files else 1


if __name__ == "__main__":
    raise SystemExit(main())
