import pytest

from obb_pbd.geometry import OrientedBox
from obb_pbd.tokens import OBBTokenCodec, format_hbb_block, format_obb_block, format_point_block


def test_obb_codec_round_trip_at_angle_boundary():
    codec = OBBTokenCodec(bins=1000)
    original = OrientedBox(400, 300, 200, 80, 179.9)
    encoded = codec.encode(original, image_size=(800, 600))
    decoded = codec.decode(encoded, image_size=(800, 600))
    assert len(encoded) == 6
    assert decoded.cx == pytest.approx(original.cx, abs=0.8)
    assert decoded.cy == pytest.approx(original.cy, abs=0.6)
    assert decoded.w == pytest.approx(original.w, abs=0.8)
    assert decoded.h == pytest.approx(original.h, abs=0.6)
    circular_error = abs(((decoded.theta_deg - original.theta_deg + 90) % 180) - 90)
    assert circular_error < 0.2


def test_blocks_always_have_length_eight():
    assert len(format_obb_block((1, 2, 3, 4, 5, 6))) == 8
    assert format_hbb_block((1, 2, 3, 4)) == (
        "<box>", 1, 2, 3, 4, "</box>", "<null>", "<null>"
    )
    assert format_point_block((1, 2)) == (
        "<box>", 1, 2, "</box>", "<null>", "<null>", "<null>", "<null>"
    )


def test_codec_rejects_out_of_image_box():
    codec = OBBTokenCodec()
    with pytest.raises(ValueError, match="normalized"):
        codec.encode(OrientedBox(900, 300, 200, 80, 0), image_size=(800, 600))
