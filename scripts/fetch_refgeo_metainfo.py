#!/usr/bin/env python3
"""
Fetch the RefGeo metainfo split that matches the imagery we actually have.

The problem this fixes
----------------------
An early setup downloaded ``geochat_train.jsonl`` / ``geochat_test.jsonl`` and paired
them with DIOR-RSVG imagery. Those two files belong to a different image pool.
RefGeo's published layout is::

    refGeo/
      images/{avvg,rsvg,dior_rsvg,geochat,vrsbench}/
      metainfo/{avvg,rsvg,dior_rsvg,geochat,vrsbench,rrsisd}_{train,val,test}.jsonl

so ``geochat_*.jsonl`` requires ``images/geochat/`` (the MBZUAI GeoChat_Instruct
pool), while the DIOR-RSVG images on disk belong with ``dior_rsvg_*.jsonl``.

``RefGeoOBBDataset`` bridged the gap by rewriting the RefGeo image id
``train_14034_0001.jpg`` into ``14034.jpg``. That filename exists in DIOR-RSVG but
is an unrelated picture: rendering the polygon on it shows "silver boeing737
airplane" landing on empty grass beside a football field, and "1 silver a220
airplane" landing on a highway overpass. Quantitatively, the polygons of 500
already-accepted records have a mean best IoU of 0.0197 against the object boxes
in the annotation XML of the image they were paired with, with 499 of 500 below
0.5. Every target paired this way was wrong.

    python scripts/fetch_refgeo_metainfo.py --subset dior_rsvg
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/erenzhou/refGeo/resolve/main"
TREE = "https://huggingface.co/api/datasets/erenzhou/refGeo/tree/main"
SPLITS = ("train", "val", "test")


def list_repo(path: str = "") -> list[dict]:
    """Directory listing from the HF dataset tree API."""
    url = f"{TREE}/{path}".rstrip("/")
    request = urllib.request.Request(url, headers={"User-Agent": "obb-pbd/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def walk(path: str = "", depth: int = 0, limit: int = 2) -> list[dict]:
    entries = []
    for entry in list_repo(path):
        entries.append(entry)
        if entry.get("type") == "directory" and depth < limit:
            entries += walk(entry["path"], depth + 1, limit)
    return entries


def fetch(url: str, target: Path) -> int:
    request = urllib.request.Request(url, headers={"User-Agent": "obb-pbd/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    target.write_bytes(payload)
    return len(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="dior_rsvg")
    parser.add_argument("--dest", default="/workspace/data/refgeo/metainfo")
    parser.add_argument("--list", action="store_true",
                        help="print the repository tree and exit")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if args.list:
        for entry in walk():
            size = entry.get("size")
            print(f"  {entry.get('type', '?'):<10} {entry['path']:<52} "
                  f"{'' if size is None else f'{size:,} B'}")
        return 0

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    # locate the files rather than assuming a path: the 404s showed the layout in
    # the dataset card is not literally the repository layout
    available = [e["path"] for e in walk() if e.get("type") == "file"
                 and e["path"].endswith(".jsonl")]
    wanted = [p for p in available
              if Path(p).name.startswith(f"{args.subset}_")]
    if not wanted:
        print(f"  no {args.subset}_*.jsonl in the repository. Available jsonl:")
        for path in sorted(available):
            print(f"    {path}")
        return 1

    fetched = 0
    for path in sorted(wanted):
        name = Path(path).name
        target = dest / name
        try:
            size = fetch(f"{BASE}/{path}", target)
        except urllib.error.HTTPError as exc:
            print(f"  {name:<28} HTTP {exc.code} (skipped)")
            target.unlink(missing_ok=True)
            continue
        except Exception as exc:
            print(f"  {name:<28} {type(exc).__name__}: {exc}")
            target.unlink(missing_ok=True)
            continue

        lines = target.read_text(encoding="utf-8").strip().splitlines()
        print(f"  {name:<28} {size:>12,} B  {len(lines):>8,} records")
        if lines:
            first = json.loads(lines[0])
            print(f"    keys      : {sorted(first.keys())}")
            print(f"    image_id  : {first.get('image_id')!r}")
            print(f"    dataset   : {first.get('dataset')!r}")
            print(f"    question  : {str(first.get('question'))[:80]!r}")
            print(f"    poly      : {first.get('poly')}")
        fetched += 1

    print(f"\n  {fetched} file(s) written to {dest}")
    return 0 if fetched else 1


if __name__ == "__main__":
    raise SystemExit(main())
