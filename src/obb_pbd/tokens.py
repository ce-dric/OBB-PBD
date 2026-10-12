from __future__ import annotations

from math import atan2, cos, degrees, radians, sin

import numpy as np

from .geometry import OrientedBox, obb_to_polygon

BlockToken = str | int


class OBBTokenCodec:
    def __init__(self, bins: int = 1000):
        if bins < 2:
            raise ValueError("bins must be at least 2")
        self.bins = bins

    def _encode_unit(self, value: float) -> int:
        # Gracefully handle floating point epsilon inaccuracies at image boundaries
        if -0.01 <= value < 0.0:
            value = 0.0
        elif 1.0 < value <= 1.01:
            value = 1.0
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"normalized value {value} out of range [0.0, 1.0]")
        return int(np.rint(value * self.bins))

    def _decode_unit(self, token: int) -> float:
        if not 0 <= token <= self.bins:
            raise ValueError(f"coordinate token must be in [0, {self.bins}]")
        return token / self.bins

    def _encode_signed(self, value: float) -> int:
        return self._encode_unit((value + 1.0) / 2.0)

    def _decode_signed(self, token: int) -> float:
        return self._decode_unit(token) * 2.0 - 1.0

    def encode(self, box: OrientedBox, image_size: tuple[int, int]) -> tuple[int, ...]:
        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        angle = radians(2.0 * box.theta_deg)
        return (
            self._encode_unit(box.cx / width),
            self._encode_unit(box.cy / height),
            self._encode_unit(box.w / width),
            self._encode_unit(box.h / height),
            self._encode_signed(sin(angle)),
            self._encode_signed(cos(angle)),
        )

    def decode(self, tokens: tuple[int, ...], image_size: tuple[int, int]) -> OrientedBox:
        if len(tokens) != 6:
            raise ValueError("OBB requires exactly six parameter tokens")
        width, height = image_size
        cx = self._decode_unit(tokens[0]) * width
        cy = self._decode_unit(tokens[1]) * height
        w = self._decode_unit(tokens[2]) * width
        h = self._decode_unit(tokens[3]) * height
        s = self._decode_signed(tokens[4])
        c = self._decode_signed(tokens[5])
        if s == 0 and c == 0:
            raise ValueError("quantized direction vector cannot be zero")
        theta = (degrees(atan2(s, c)) / 2.0) % 180.0
        if w < h:
            w, h = h, w
            theta = (theta + 90.0) % 180.0
        return OrientedBox(cx, cy, w, h, theta)

    def encode_corners(self, box: OrientedBox, image_size: tuple[int, int]) -> tuple[int, ...]:
        """4-slot corner encoding ``(x1, y1, x2, y2)``, matching the original
        upstream LocateAnything HBB block (``model_config.HBB_BLOCK_SIZE = 6``:
        ``<box>`` + 4 coordinate tokens + ``</box>``).

        Only meaningful for an axis-aligned box (``theta_deg`` 0 or 90) -- there
        is no slot for orientation, unlike :meth:`encode`. Reads the extremes off
        ``box``'s own polygon rather than assuming ``theta_deg in (0, 90)``, so a
        genuinely rotated box is silently reduced to its axis-aligned envelope
        instead of raising; callers that need to enforce axis-alignment upstream
        (``dataset.py``'s ``hbb_corner``/``hbb_envelope`` target modes both call
        ``metrics.obb_to_aabb`` first) should keep doing so.
        """
        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        points = obb_to_polygon(box)
        x1 = float(points[:, 0].min())
        y1 = float(points[:, 1].min())
        x2 = float(points[:, 0].max())
        y2 = float(points[:, 1].max())
        return (
            self._encode_unit(x1 / width),
            self._encode_unit(y1 / height),
            self._encode_unit(x2 / width),
            self._encode_unit(y2 / height),
        )

    def decode_corners(self, tokens: tuple[int, ...], image_size: tuple[int, int]) -> OrientedBox:
        """Inverse of :meth:`encode_corners`. Always returns ``theta_deg`` 0 or
        90 -- corner form cannot express any other orientation."""
        if len(tokens) != 4:
            raise ValueError("corner HBB requires exactly four coordinate tokens")
        width, height = image_size
        x1 = self._decode_unit(tokens[0]) * width
        y1 = self._decode_unit(tokens[1]) * height
        x2 = self._decode_unit(tokens[2]) * width
        y2 = self._decode_unit(tokens[3]) * height
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        if w >= h:
            return OrientedBox(cx, cy, w, h, 0.0)
        return OrientedBox(cx, cy, h, w, 90.0)


def format_obb_block(tokens: tuple[int, ...]) -> tuple[BlockToken, ...]:
    if len(tokens) != 6:
        raise ValueError("OBB block requires six coordinate tokens")
    return ("<box>", *tokens, "</box>")


def format_hbb_block(tokens: tuple[int, ...]) -> tuple[BlockToken, ...]:
    if len(tokens) != 4:
        raise ValueError("HBB block requires four coordinate tokens")
    return ("<box>", *tokens, "</box>", "<null>", "<null>")


def format_point_block(tokens: tuple[int, ...]) -> tuple[BlockToken, ...]:
    if len(tokens) != 2:
        raise ValueError("point block requires two coordinate tokens")
    return ("<box>", *tokens, "</box>", "<null>", "<null>", "<null>", "<null>")
