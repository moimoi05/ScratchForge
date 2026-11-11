import numpy as np
import pytest
from PIL import Image, ImageDraw

from scratchforge.warping import warp_stamp


def stamp():
    pixels = np.zeros((96, 28, 3), dtype=np.uint8)
    pixels[..., 0] = np.arange(96)[:, None] + 80
    pixels[..., 1:] = 60
    mask = Image.new('L', (28, 96)); ImageDraw.Draw(mask).line([(14, 7), (14, 88)], fill=255, width=7)
    return Image.fromarray(pixels), mask


@pytest.mark.parametrize('points', [[20, 20, 115, 70], [10, 65, 40, 25, 80, 30, 120, 65],
                                   [-10, 40, 70, 40], [20, 20, 90, 80, 20, 80, 90, 20]])
def test_tps_warps_native_texture_along_polyline(points):
    rgb, source_mask = stamp()
    warped, mask, box, metadata = warp_stamp(rgb, source_mask, points, 9, (140, 100))
    assert warped.size == mask.size == (box[2]-box[0], box[3]-box[1])
    assert mask.getbbox() and set(np.unique(mask)) <= {0, 255}
    assert metadata['warp'] == 'thin_plate_spline' and metadata['vae_used'] is False
    assert 0 <= box[0] < box[2] <= 140 and 0 <= box[1] < box[3] <= 100
    full = Image.new('L', (140, 100)); full.paste(mask, box[:2])
    allowed = Image.new('L', full.size)
    ImageDraw.Draw(allowed).line(list(zip(points[::2], points[1::2])), fill=255, width=9, joint='curve')
    assert not np.any((np.asarray(full) > 0) & (np.asarray(allowed) == 0))
    assert len(np.unique(np.asarray(warped)[np.asarray(mask) > 0], axis=0)) > 4


def test_bad_centerlines_and_shapes_have_clear_errors():
    rgb, mask = stamp()
    for points in ([3, 3, 3, 3], [3, 3, 4], [float('nan'), 3, 4, 4]):
        with pytest.raises(ValueError):
            warp_stamp(rgb, mask, points, 9, (140, 100))
    with pytest.raises(ValueError, match='skeleton|axis|centerline'):
        warp_stamp(rgb, Image.new('L', rgb.size), [10, 10, 40, 40], 9, (140, 100))
    with pytest.raises(ValueError, match='dimensions'):
        warp_stamp(rgb, Image.new('L', (2, 2)), [10, 10, 40, 40], 9, (140, 100))
