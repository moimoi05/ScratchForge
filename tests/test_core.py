"""Core contracts use fixtures; real data is verified by verify_dataset.py."""
import numpy as np
import pytest
from PIL import Image

from scratchforge.dataset import copy_checked, crop_box, mask_for, translate, valid_polygons
from scratchforge.engine import composite


def test_copy_collision_and_idempotence(tmp_path):
    source, destination = tmp_path/'source', tmp_path/'destination'
    source.write_bytes(b'original')
    copy_checked(source, destination)
    copy_checked(source, destination)
    source.write_bytes(b'changed')
    with pytest.raises(ValueError):
        copy_checked(source, destination)
    assert destination.read_bytes() == b'original'


@pytest.mark.parametrize('seg', [[[1, 2, 3]], [[0, 0, 1, 1, 2, 2]],
                                  [[0, 0, 999, 0, 1, 2]], {}])
def test_invalid_polygon(seg):
    assert not valid_polygons(seg, 20, 20)


def test_complete_long_instance_and_coordinate_roundtrip():
    polygons = [[300, 300, 310, 300, 310, 1700, 300, 1700]]
    x, y, right, bottom = crop_box(polygons, (2000, 2000))
    assert right-x >= 256 and bottom-y >= 1432
    local = translate(polygons, x, y)
    assert translate(local, -x, -y) == polygons
    assert mask_for(local, (right-x, bottom-y)).getpixel((305-x, 500-y)) == 255
    assert crop_box([[0, 0, 10, 0, 10, 10]], (100, 100), 512) == (0, 0, 100, 100)


def test_composite_preserves_fullframe_and_exact_background():
    original = Image.new('RGB', (60, 40), (13, 23, 33))
    generated = Image.new('RGB', original.size, 'white')
    mask = mask_for([[10, 10, 20, 10, 20, 20]], original.size)
    output = composite(original, generated, mask)
    outside = np.asarray(mask) == 0
    assert output.size == original.size
    assert np.array_equal(np.asarray(output)[outside], np.asarray(original)[outside])
    with pytest.raises(ValueError):
        composite(original, Image.new('RGB', (1, 1)), mask)
