from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from PIL import Image
from torch.utils.data import Dataset

from obb_pbd.geometry import polygon_to_obb, OrientedBox, InvalidPolygonError
from obb_pbd.metrics import obb_to_aabb
from obb_pbd.tokens import OBBTokenCodec

#: ``target_mode`` values ``RefGeoOBBDataset`` accepts.
TARGET_MODES = ("obb", "hbb_envelope", "hbb_corner")


#: RefGeo consolidates five sources, each with its own image pool. The published
#: layout is::
#:
#:     refGeo/
#:       images/{avvg,rsvg,dior_rsvg,geochat,vrsbench}/
#:       metainfo/{subset}_{split}.jsonl
#:
#: so a metainfo file is only meaningful next to *its own* images. Candidates are
#: tried in order; the first directory that exists is indexed.
SUBSET_IMAGE_DIRS: Dict[str, Sequence[str]] = {
    "avvg": ("refgeo/images/avvg",),
    "dior_rsvg": ("refgeo/images/dior_rsvg",
                  "DIOR-RSVG/JPEGImages/JPEGImages",
                  "DIOR-RSVG/JPEGImages"),
    "geochat": ("refgeo/images/geochat",),
    "rsvg": ("refgeo/images/rsvg",),
    "vrsbench": ("refgeo/images/vrsbench",),
}

#: splits a subset actually publishes; anything else falls back to test
SUBSET_SPLITS: Dict[str, Sequence[str]] = {
    "avvg": ("train", "test"),
    "dior_rsvg": ("train", "val", "test"),
    "geochat": ("train", "test"),
    "rsvg": ("train", "val", "test"),
    "vrsbench": ("train", "test"),
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

#: Longest side an image is scaled to before it reaches the processor.
#:
#: AVVG frames are 4000x2250 DJI stills and go through the processor as 6,639
#: tokens; DIOR-RSVG at 800x800 produced ~890, which is the scale every VRAM and
#: throughput figure in this project was measured at. 1024 brings AVVG back to 828
#: tokens so those figures still hold, and
#: it costs nothing in target precision: coordinates are normalised by the frame,
#: so the encoded bins are identical at every scale -- measured 61.7 / 53.3 for
#: w / h at native, 1600, 1024 and 800 alike. A car stays roughly 60 px wide.
#:
#: Pass ``max_image_side=None`` for native resolution.
DEFAULT_MAX_IMAGE_SIDE = 1024


class ImagePairingError(RuntimeError):
    """Raised when too few records resolve to an image on disk.

    This exists because of a specific failure. Earlier revisions paired
    ``geochat_*.jsonl`` with DIOR-RSVG imagery by rewriting the RefGeo image id
    ``train_14034_0001.jpg`` into ``14034.jpg``. That filename exists in
    DIOR-RSVG, so nothing raised and 30% of records "loaded" -- but it is an
    unrelated picture. Rendering the polygon on it put "silver boeing737
    airplane" on bare grass beside a football field, and over 500 accepted
    records the polygon's AABB had a mean best IoU of 0.0197 against the object
    boxes annotated for the resolved image. Every such target was mispaired, and
    the only symptom was a low score.

    Silence is the danger, so a low resolve rate is now fatal by default.
    """


class RefGeoOBBDataset(Dataset):
    """RefGeo oriented-box grounding records paired with their own imagery.

    Image ids are used **verbatim**. There is no filename rewriting: if a record's
    image is not present the record is skipped and counted, and an implausible
    overall resolve rate aborts construction rather than training on nonsense.

    Subset choice decides what can be learned at all
    (train splits):

        avvg       26,482 records, 97.85% genuinely oriented, off-axis angle mean
                   20.39 deg, aspect median 2.14 with 99.6% above the 1.1
                   direction-stability threshold. Images ship with RefGeo.
        dior_rsvg  27,133 records, ids match the DIOR-RSVG files exactly, but
                   100.00% of the polygons are axis-aligned: no orientation to
                   learn.
        geochat    63,883 records, oriented, images are a separate large download.

    ``target_mode`` controls what geometry ``__getitem__`` encodes as the training
    target, independent of the OBB-vs-HBB decision that lives in ``model_config``:

        obb            (default) the polygon's minimum-rotated-rectangle OBB.
        hbb_envelope    the OBB's axis-aligned envelope (``metrics.obb_to_aabb``),
                        with ``theta_deg`` forced to 0 or 90 by construction. This
                        exists for the fine-tuned-HBB control: same images, same
                        prompts, same LoRA recipe, same 6-geometry-slot block and
                        collator as the OBB run, but the model can never express an
                        angle other than axis-aligned. It isolates "the run learned
                        the parallel-block coordinate format" (present in both arms)
                        from "the run learned oriented geometry" (present only in
                        the ``obb`` arm): part of any trained gain is "learned to
                        fill the block" rather than "learned OBB geometry"
                        specifically. Frame-containment
                        filtering still runs on the polygon's OBB (identical sample
                        set to the ``obb`` arm), and the envelope substitution
                        happens after that check, so envelope containment follows
                        for free and this arm never needs its own filtering pass.
        hbb_corner      the same axis-aligned envelope geometry as ``hbb_envelope``,
                        but encoded and tokenized as the *original* upstream
                        LocateAnything 4-slot corner block ``<box> x1 y1 x2 y2
                        </box>`` (``model_config.HBB_BLOCK_SIZE = 6``) instead of
                        the 6-slot ``cx cy w h sin cos`` block both other modes
                        use. This is the matched 4-slot HBB-PBD baseline: same
                        data, same LoRA recipe, same fine-tuning, but the
                        *pretrained* coordinate interface, so it isolates the
                        six-slot block-format step from the fine-tuning step in
                        the format/orientation decomposition (see
                        ``collators.py``'s ``n_geometry_slots`` and
                        ``tokens.py``'s ``encode_corners``/``decode_corners``).

                        NOT the same sample set as ``obb``/``hbb_envelope``: a
                        corner block can only represent an axis-aligned box whose
                        own extremes (x1, y1, x2, y2) are all in [0, frame], and
                        the envelope of an in-frame *rotated* box can extend past
                        the frame even when the rotated box's own cx/cy/w/h
                        normalise fine (``_encodable`` never computes a corner
                        coordinate, so it cannot catch this -- found empirically
                        2026-08-08, see ``_build_dataset``'s corner-frame check).
                        ~9.8% of AVVG's training records are additionally dropped
                        for this reason. Use ``require_corner_encodable=True`` on
                        an ``obb``-mode dataset to get the *same* filtered sample
                        set with the *true* rotated geometry still stored, e.g.
                        for scoring an ``hbb_corner`` checkpoint against the
                        record set it was actually trained/evaluated on.
    """

    def __init__(
        self,
        data_dir: str | Path = "/workspace/data",
        split: str = "train",
        subset: str = "avvg",
        max_samples: Optional[int] = None,
        bins: int = 1000,
        verbose: bool = True,
        max_image_side: Optional[int] = DEFAULT_MAX_IMAGE_SIDE,
        min_resolve_rate: float = 0.5,
        require_polygon_in_frame: bool = True,
        target_mode: str = "obb",
        require_corner_encodable: bool = False,
    ):
        super().__init__()
        if subset not in SUBSET_IMAGE_DIRS:
            raise ValueError(
                f"unknown subset {subset!r}; expected one of "
                f"{sorted(SUBSET_IMAGE_DIRS)}"
            )
        if target_mode not in TARGET_MODES:
            raise ValueError(
                f"unknown target_mode {target_mode!r}; expected one of {TARGET_MODES}"
            )
        self.data_dir = Path(data_dir)
        self.split = split
        self.subset = subset
        self.max_samples = max_samples
        self.codec = OBBTokenCodec(bins=bins)
        self.max_image_side = max_image_side
        self.min_resolve_rate = min_resolve_rate
        self.require_polygon_in_frame = require_polygon_in_frame
        self.target_mode = target_mode
        self.require_corner_encodable = require_corner_encodable
        self.image_root: Optional[Path] = None
        self.samples: List[Dict[str, Any]] = []
        self.stats: Dict[str, int] = {}

        self._build_dataset(verbose=verbose)

    # ------------------------------------------------------------------ paths --

    def _metainfo_path(self) -> Path:
        available = SUBSET_SPLITS.get(self.subset, ("train", "test"))
        split = self.split if self.split in available else "test"
        return (self.data_dir / "refgeo" / "metainfo"
                / f"{self.subset}_{split}.jsonl")

    def _index_images(self, verbose: bool) -> Dict[str, Path]:
        """Filename -> path for the subset's image directory.

        Indexed rather than probed per record so nested archive layouts work and
        so the resolve rate can be reported without 20k filesystem round trips.
        """
        for relative in SUBSET_IMAGE_DIRS[self.subset]:
            root = self.data_dir / relative
            if not root.is_dir():
                continue
            index: Dict[str, Path] = {}
            for path in root.rglob("*"):
                if path.name.startswith("._"):
                    continue           # AppleDouble stubs from a macOS-made zip
                if "__MACOSX" in path.parts:
                    continue
                if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
                    index.setdefault(path.name, path)
            if index:
                if verbose:
                    print(f"[{self.subset}] {len(index):,} images indexed under "
                          f"{root}")
                self.image_root = root
                return index
        return {}

    # ------------------------------------------------------------------ build --

    def _build_dataset(self, verbose: bool):
        jsonl_path = self._metainfo_path()
        if not jsonl_path.exists():
            raise FileNotFoundError(
                f"metainfo not found: {jsonl_path}. Fetch it with "
                f"scripts/fetch_refgeo_metainfo.py --subset {self.subset}"
            )

        images = self._index_images(verbose=verbose)
        if not images:
            raise ImagePairingError(
                f"no images found for subset {self.subset!r} under "
                f"{[str(self.data_dir / r) for r in SUBSET_IMAGE_DIRS[self.subset]]}. "
                "RefGeo ships metainfo and imagery separately; pairing a metainfo "
                "file with another subset's images silently mispairs every target."
            )

        counts = {"total": 0, "no question or poly": 0, "image missing": 0,
                  "bad polygon": 0, "outside frame": 0, "outside frame (corner)": 0,
                  "loaded": 0}
        size_cache: Dict[str, tuple] = {}

        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_idx, line in enumerate(handle, start=1):
                if self.max_samples is not None and len(self.samples) >= self.max_samples:
                    break
                if not line.strip():
                    continue
                counts["total"] += 1
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    counts["bad polygon"] += 1
                    continue

                question = str(item.get("question", "")).strip()
                raw_poly = item.get("poly")
                if not question or not raw_poly:
                    counts["no question or poly"] += 1
                    continue

                image_id = str(item.get("image_id", ""))
                img_path = images.get(image_id)
                if img_path is None:
                    counts["image missing"] += 1
                    continue

                candidate = raw_poly
                if (isinstance(raw_poly, list) and len(raw_poly) == 1
                        and isinstance(raw_poly[0], list)):
                    candidate = raw_poly[0]
                if not isinstance(candidate, list) or len(candidate) != 4:
                    counts["bad polygon"] += 1
                    continue

                try:
                    obb = polygon_to_obb(candidate)
                except (InvalidPolygonError, ValueError, TypeError):
                    counts["bad polygon"] += 1
                    continue

                # A polygon reaching outside the frame cannot be encoded: the
                # codec rejects normalised values below 0 or above 1, and it would
                # raise inside __getitem__ mid-training rather than here. AVVG has
                # such records; the very first train polygon has y = -55.
                scale = 1.0
                # hbb_corner (and require_corner_encodable) always needs width/
                # height below (the corner-frame re-check), regardless of
                # require_polygon_in_frame/max_image_side.
                needs_corner_check = (self.target_mode == "hbb_corner"
                                      or self.require_corner_encodable)
                if self.require_polygon_in_frame or self.max_image_side or needs_corner_check:
                    key = str(img_path)
                    if key not in size_cache:
                        with Image.open(img_path) as probe:
                            size_cache[key] = probe.size
                    width, height = size_cache[key]
                    if (self.require_polygon_in_frame
                            and not self._encodable(obb, width, height)):
                        counts["outside frame"] += 1
                        continue
                    if self.max_image_side and max(width, height) > self.max_image_side:
                        scale = self.max_image_side / max(width, height)

                if scale != 1.0:
                    # Store the box in the frame the model will actually see.
                    # Evaluation reads samples[i]["obb"] as ground truth, so a
                    # scale applied only in __getitem__ would silently shift every
                    # target by the resize factor.
                    obb = OrientedBox(obb.cx * scale, obb.cy * scale,
                                      obb.w * scale, obb.h * scale, obb.theta_deg)

                # Envelope substitution and the corner-frame check are computed
                # independently of each other: ``target_mode`` decides what gets
                # STORED as this sample's geometry, ``needs_corner_check`` decides
                # whether this record gets INCLUDED at all. They used to be the
                # same condition (target_mode == "hbb_corner"), which is what
                # created the bug below describes.
                #
                # ``_encodable`` (the 6-slot ``codec.encode`` check, above) only
                # verifies cx/cy/w/h/sin/cos each individually normalize into
                # [0,1] -- it never computes a corner coordinate, so it CANNOT
                # catch a box whose own cx/w combination puts an edge outside the
                # frame (e.g. cx=30, w=100 in an 800-wide image: cx/width and
                # w/width both individually valid, but x1 = cx - w/2 = -20). The
                # ``hbb_envelope`` arm never notices, because its 6-slot block
                # never encodes x1 as its own token. ``hbb_corner`` does, and hits
                # this for real (found empirically 2026-08-08: a smoke-test batch
                # raised "normalized value -0.0555 out of range" here on a record
                # this exact check had waved through) -- about 9.8% of AVVG train
                # records fail this stricter check, so hbb_corner's sample set is
                # NOT the same as obb's/hbb_envelope's.
                #
                # A second, independently-discovered bug this decoupling exists
                # to fix: eval_pbd_direct.py always scores against a target_mode=
                # "obb" dataset (ground truth must be the TRUE rotated box, not
                # the envelope -- see run_hbb_control_eval.sh). Before this
                # decoupling there was no way to build an "obb"-mode dataset that
                # ALSO applied hbb_corner's stricter filter, so evaluating an
                # hbb_corner checkpoint against a plain "obb" dataset silently
                # compared "record N of the corner-filtered training set" against
                # "record N of the unfiltered eval set" -- different underlying
                # records whenever any of the earlier ~9.8% got dropped, which
                # is almost always. Found via scripts/debug_hbb_corner_positions.py:
                # a checkpoint that decoded its OWN (envelope) training target
                # exactly (rotIoU 0.99 against the envelope) scored rotIoU 0.0000
                # against the "true" GT read from the wrong record (a box with a
                # completely different center). ``require_corner_encodable=True``
                # on an "obb"-mode dataset reproduces hbb_corner's exact filtered
                # record sequence while keeping the true rotated geometry stored.
                if needs_corner_check:
                    envelope = obb_to_aabb(obb)
                    scaled_w = max(1, round(width * scale))
                    scaled_h = max(1, round(height * scale))
                    try:
                        self.codec.encode_corners(envelope, (scaled_w, scaled_h))
                    except (ValueError, TypeError):
                        counts["outside frame (corner)"] += 1
                        continue
                elif self.target_mode == "hbb_envelope":
                    envelope = obb_to_aabb(obb)

                if self.target_mode in ("hbb_envelope", "hbb_corner"):
                    obb = envelope

                self.samples.append({
                    "img_path": str(img_path),
                    "image_id": image_id,
                    "question": question,
                    "obb": obb,
                    "scale": scale,
                    "record_id": item.get("question_id", line_idx),
                })
                counts["loaded"] += 1

        self.stats = counts
        resolvable = counts["total"] - counts["image missing"]
        rate = resolvable / max(1, counts["total"])

        if verbose:
            print(f"[{self.subset}/{self.split}] {counts['loaded']:,} samples "
                  f"from {counts['total']:,} records in {jsonl_path.name}")
            for key in ("image missing", "no question or poly", "bad polygon",
                        "outside frame", "outside frame (corner)"):
                if counts[key]:
                    print(f"    skipped, {key:<22} {counts[key]:>8,} "
                          f"({100.0 * counts[key] / max(1, counts['total']):5.2f}%)")
            print(f"    image resolve rate      {100.0 * rate:5.2f}%")
            print(f"    max_image_side          {self.max_image_side}")

        if rate < self.min_resolve_rate:
            raise ImagePairingError(
                f"only {100.0 * rate:.2f}% of {counts['total']:,} "
                f"{self.subset}/{self.split} records resolve to an image under "
                f"{self.image_root} (minimum {100.0 * self.min_resolve_rate:.0f}%). "
                "That usually means the metainfo and the imagery come from "
                "different RefGeo subsets. Pass min_resolve_rate=0 to override "
                "deliberately."
            )

    def _encodable(self, obb: OrientedBox, width: int, height: int) -> bool:
        try:
            self.codec.encode(obb, (width, height))
        except (ValueError, TypeError):
            return False
        return True

    # -------------------------------------------------------------- accessors --

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        img = Image.open(sample["img_path"]).convert("RGB")
        scale = sample.get("scale", 1.0)
        if scale != 1.0:
            img = img.resize((max(1, round(img.width * scale)),
                              max(1, round(img.height * scale))),
                             Image.BILINEAR)

        # sample["obb"] is already expressed in this frame, so the tokens the model
        # is trained on and the ground truth evaluation reads are the same object.
        obb = sample["obb"]
        width, height = img.size
        if self.target_mode == "hbb_corner":
            tokens = self.codec.encode_corners(obb, (width, height))
        else:
            tokens = self.codec.encode(obb, (width, height))
        obb_str = "<box>" + "".join(f"<{t}>" for t in tokens) + "</box>"
        prompt = ("Locate a single instance that matches the following "
                  f"description: {sample['question']}.")

        return {
            "image": img,
            "prompt": prompt,
            "target_obb_str": obb_str,
            "record_id": sample["record_id"],
            "img_path": sample["img_path"],
            "obb": obb,
        }
