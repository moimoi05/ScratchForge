"""Automatic labels retain pixel support rather than the AI guidance rectangle."""
import numpy as np
import pytest
from PIL import Image, ImageDraw
from pycocotools import mask as coco_tools

from scratchforge.dataset import mask_for
from scratchforge.labels import coco_document, difference_mask, mask_to_polygons


@pytest.mark.parametrize('shape', ['rectangle', 'hole', 'diagonal', 'thin', 'islands'])
def test_mask_polygon_roundtrip_keeps_holes_and_small_components(shape):
    mask = Image.new('L', (45, 35)); draw = ImageDraw.Draw(mask)
    if shape in ('rectangle', 'hole'):
        draw.rectangle((2, 2, 40, 30), fill=255)
        if shape == 'hole':
            draw.rectangle((12, 9, 24, 19), fill=0)
    elif shape == 'diagonal':
        draw.line([(3, 3), (20, 27), (40, 6)], fill=255, width=5)
    elif shape == 'thin':
        draw.line([(0, 0), (44, 34)], fill=255, width=1)
    else:
        draw.rectangle((1, 1, 3, 8), fill=255); draw.point((44, 34), fill=255)
    polygons = mask_to_polygons(mask)
    restored = mask_for(polygons, mask.size)
    assert np.array_equal(np.asarray(restored), np.asarray(mask))
    coco = coco_document(polygons, mask.size)
    assert coco['images'][0]['width'] == 45
    assert sum(row['area'] for row in coco['annotations']) == np.count_nonzero(np.asarray(mask))
    encoded = coco_tools.frPyObjects([polygon for row in coco['annotations'] for polygon in row['segmentation']],
                                    mask.height, mask.width)
    decoded = coco_tools.decode(coco_tools.merge(encoded))*255
    assert np.array_equal(decoded, np.asarray(mask))
    for row in coco['annotations']:
        encoded = coco_tools.merge(coco_tools.frPyObjects(row['segmentation'], mask.height, mask.width))
        assert int(coco_tools.area(encoded)) == row['area']
        assert coco_tools.toBbox(encoded).tolist() == row['bbox']


def test_difference_restricts_support_and_filters_noise():
    before = Image.new('RGB', (20, 20), (50, 50, 50))
    after = before.copy(); draw = ImageDraw.Draw(after)
    draw.rectangle((0, 0, 3, 3), fill='white')
    draw.rectangle((8, 8, 12, 9), fill=(80, 50, 50)); draw.point((15, 15), fill='white')
    support = Image.new('L', before.size); ImageDraw.Draw(support).rectangle((5, 5, 18, 18), fill=255)
    mask = difference_mask(before, after, support, threshold=8, min_area=3)
    assert np.count_nonzero(mask) == 10
    assert mask.getpixel((0, 0)) == mask.getpixel((15, 15)) == 0
    with pytest.raises(ValueError, match='dimensions'):
        difference_mask(before, Image.new('RGB', (2, 2)), support)
    with pytest.raises(ValueError):
        difference_mask(before, after, support, threshold=-1)


def test_empty_mask_and_nonbinary_rejected():
    assert mask_to_polygons(Image.new('L', (8, 8))) == []
    with pytest.raises(ValueError, match='binary'):
        mask_to_polygons(Image.new('L', (8, 8), 42))
    one_bit = Image.new('1', (8, 8), 1)
    assert mask_to_polygons(one_bit)


def test_coco_uses_local_rasterization_and_enforces_label_limits(monkeypatch):
    allocate = Image.new; sizes = []
    def record_new(mode, size, *args, **kwargs):
        sizes.append(size)
        return allocate(mode, size, *args, **kwargs)
    monkeypatch.setattr('scratchforge.labels.Image.new', record_new)
    polygon = [5000., 3000., 5005., 3000., 5005., 3002., 5000., 3002.]
    document = coco_document([polygon], (5861, 3421))
    assert sizes == [(6, 3)]
    assert document['annotations'][0]['area'] == 18
    assert document['annotations'][0]['bbox'] == [5000, 3000, 6, 3]
    with pytest.raises(ValueError, match='limit'):
        coco_document([polygon]*10001, (5861, 3421))
    with pytest.raises(ValueError, match='polygon'):
        coco_document([[float('nan'), 1, 2, 2, 3, 1]], (30, 30))


def test_noisy_mask_limits_fail_before_contour_allocation(monkeypatch):
    mask = Image.new('L', (16, 16)); draw = ImageDraw.Draw(mask)
    for x in (1, 5, 9):
        draw.rectangle((x, 1, x+1, 3), fill=255)
    monkeypatch.setattr('scratchforge.labels.MAX_POLYGONS', 2)
    def forbidden(*args, **kwargs):
        raise AssertionError('Allocated contours after known component limit failure')
    monkeypatch.setattr('scratchforge.labels.cv2.findContours', forbidden)
    with pytest.raises(ValueError, match='limit'):
        mask_to_polygons(mask)
