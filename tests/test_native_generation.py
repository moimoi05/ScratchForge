from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from backend.images import save_tiff, validate_tiff
from scratchforge.dataset import mask_for, sha
from scratchforge.generation import generate_scene
import pytest


def native_stamp(tmp_path):
    rgb = Image.new('RGB', (24, 80), (140, 90, 60))
    mask = Image.new('L', rgb.size); ImageDraw.Draw(mask).line([(12, 6), (12, 73)], fill=255, width=7)
    save_tiff(rgb, tmp_path/'stamp.tif'); mask.save(tmp_path/'mask.png')
    return {'id': 'native', 'split': 'train', 'rgb': 'stamp.tif', 'mask': 'mask.png',
            'rgb_sha256': sha(tmp_path/'stamp.tif'), 'mask_sha256': sha(tmp_path/'mask.png')}


def test_stamp_brush_v2_keeps_tiff_color_and_preserves_pixels_outside(tmp_path):
    stamp = native_stamp(tmp_path)
    original = Image.new('RGB', (160, 110), (40, 40, 40))
    scene = {'brush_method': 'stamp_brush_v2', 'brush_stamp_id': 'native',
             'brushes': [{'points': [20, 65, 50, 42, 90, 45, 130, 74], 'width': 31}]}
    def forbidden(*args):
        pytest.fail('V2 must not transfer background color into the scratch')
    result = generate_scene(original, scene, [stamp], tmp_path, harmonize=forbidden)
    pixels = np.asarray(result.image)
    inside = np.asarray(result.label_mask) > 0
    assert inside.any() and np.all(pixels[inside] == [140, 90, 60])
    assert np.all(pixels[~inside] == 40)
    assert result.layers[0]['effective_method'] == 'stamp_brush_v2'


def test_stamp_brush_v2_has_asymmetric_ends_and_is_reproducible(tmp_path):
    stamp = native_stamp(tmp_path)
    original = Image.new('RGB', (160, 100), (40, 40, 40))
    scene = {'brush_method': 'stamp_brush_v2', 'brush_stamp_id': 'native', 'seed': 42,
             'brushes': [{'points': [20, 50, 130, 50], 'width': 31}]}
    result = generate_scene(original, scene, [stamp], tmp_path)
    mask = np.asarray(result.label_mask) > 0
    assert abs(int(mask[:, 23:32].sum()) - int(mask[:, 118:127].sum())) > 10
    assert np.all(np.asarray(result.image)[~mask] == 40)
    assert np.array_equal(np.asarray(result.image), np.asarray(generate_scene(original, scene, [stamp], tmp_path).image))
    assert result.layers[0]['effective_method'] == 'stamp_brush_v2'


def test_tiff_is_rgb_lzw_300dpi_without_pixel_changes(tmp_path):
    pixels = np.arange(30*40*3, dtype=np.uint8).reshape(30, 40, 3)
    path = save_tiff(Image.fromarray(pixels), tmp_path/'industrial.tif')
    with Image.open(path) as saved:
        validate_tiff(saved)
        assert saved.tag_v2[259] == 5
        assert tuple(saved.info['dpi']) == (300, 300)
        assert np.array_equal(np.asarray(saved), pixels)


def test_tiff_validation_and_preview_boundaries(tmp_path, monkeypatch):
    from backend.images import preview_png, save_image
    path = save_tiff(Image.new('RGB', (40, 30), (10, 20, 30)), tmp_path/'source.tif')
    preview = preview_png(path, tmp_path/'preview.png', (20, 20))
    assert Image.open(preview).size == (20, 15)
    assert preview_png(path, preview) == preview
    with pytest.raises(ValueError, match='24-bit RGB'):
        save_tiff(Image.new('RGBA', (3, 3)), tmp_path/'wrong.tif')
    with pytest.raises(ValueError, match='TIFF RGB'):
        validate_tiff(Image.new('RGB', (3, 3)))
    Image.new('L', (3, 3)).save(tmp_path/'gray.tif')
    with Image.open(tmp_path/'gray.tif') as image, pytest.raises(ValueError, match='TIFF RGB'):
        validate_tiff(image)
    with Image.open(path) as image:
        monkeypatch.setattr('backend.images.MAX_TIFF_PIXELS', 1)
        with pytest.raises(ValueError, match='megapixels'):
            validate_tiff(image)
    monkeypatch.setattr('backend.images.MAX_TIFF_PIXELS', 64_000_000)
    Image.new('RGB', (3, 3)).save(tmp_path/'multi.tif', save_all=True, append_images=[Image.new('RGB', (3, 3))])
    with Image.open(tmp_path/'multi.tif') as image, pytest.raises(ValueError, match='single-page'):
        validate_tiff(image)
    Image.new('RGB', (3, 3)).save(tmp_path/'rotated.tif', tiffinfo={274: 6})
    with Image.open(tmp_path/'rotated.tif') as image, pytest.raises(ValueError, match='orientation'):
        validate_tiff(image)
    def fail_save(*args, **kwargs):
        raise OSError('Disk unavailable')
    monkeypatch.setattr(Image.Image, 'save', fail_save)
    with pytest.raises(OSError, match='Disk unavailable'):
        save_image(Image.new('RGB', (3, 3)), tmp_path/'failed.png', 'PNG')
    assert not list(tmp_path.glob('*.tmp'))


def test_upload_limit_accepts_uncompressed_industrial_rgb_frame():
    from backend.images import MAX_TIFF_PIXELS, MAX_TIFF_UPLOAD
    assert MAX_TIFF_UPLOAD > 5861*3421*3
    assert MAX_TIFF_UPLOAD >= MAX_TIFF_PIXELS*3


def test_ai_labels_detect_changed_scratches_and_clip_to_exact_stroke():
    original = Image.new('RGB', (150, 100), (40, 40, 40))
    scene = {'brushes': [{'points': [20, 40, 120, 40], 'width': 9}],
             'brush_method': 'sdxl', 'edit_margin': 24, 'seed': 1}
    def infer(rgb, mask, seed, strength):
        out = rgb.copy(); ImageDraw.Draw(out).rectangle((45, 38, 55, 42), fill=(100, 40, 40))
        ImageDraw.Draw(out).rectangle((0, 0, 8, 8), fill='white')
        return out
    output = generate_scene(original, scene, [], None, infer)
    assert output.polygons and output.label_mask.getbbox() == (45, 38, 56, 43)
    assert np.array_equal(np.asarray(mask_for(output.polygons, original.size)), np.asarray(output.label_mask))
    outside = np.asarray(output.guidance) == 0
    assert np.array_equal(np.asarray(output.image)[outside], np.asarray(original)[outside])
    assert np.array_equal(np.asarray(output.edit), np.asarray(output.guidance))


def test_stamp_brush_uses_tiff_harmonizer_and_never_vae(tmp_path):
    stamp = native_stamp(tmp_path); seen = []
    def forbidden(*args):
        raise AssertionError('TIFF stamp entered VAE')
    def harmonize(rgb, mask, background, method):
        seen.append(method)
        return SimpleNamespace(image=rgb, requested_method=method, effective_method=method,
                               warnings=(), metadata={'vae_used': False})
    scene = {'brushes': [{'points': [15, 60, 65, 30, 120, 60], 'width': 9, 'stamp_id': 'native'}],
             'brush_method': 'stamp_brush', 'stamp_method': 'classical_laplacian', 'seed': 4}
    original = Image.new('RGB', (140, 90), (40, 40, 40))
    output = generate_scene(original, scene, [stamp], tmp_path, forbidden, harmonize)
    assert seen == ['classical_laplacian'] and output.polygons
    assert output.layers[0]['stamp_id'] == 'native'
    assert output.layers[0]['effective_method'] == 'stamp_brush'
    assert output.layers[0]['metadata']['harmonization']['effective_method'] == 'classical_laplacian'
    outside = np.asarray(output.edit) == 0
    assert np.array_equal(np.asarray(output.image)[outside], np.asarray(original)[outside])


def test_stamp_brush_random_and_invalid_selection(tmp_path):
    stamp = native_stamp(tmp_path)
    original = Image.new('RGB', (120, 90), (40, 40, 40))
    scene = {'brushes': [{'points': [20, 20, 100, 70], 'width': 9}],
             'brush_method': 'stamp_brush', 'brush_stamp_mode': 'random', 'seed': 4}
    output = generate_scene(original, scene, [stamp], tmp_path)
    repeat = generate_scene(original, scene, [stamp], tmp_path)
    assert output.layers[0]['stamp_id'] == 'native'
    assert np.array_equal(np.asarray(output.image), np.asarray(repeat.image))
    with pytest.raises(ValueError, match='known stamp'):
        generate_scene(original, scene, [], tmp_path)
    selected = {**scene, 'brush_stamp_id': 'native'}
    with pytest.raises(ValueError, match='train'):
        generate_scene(original, selected, [{**stamp, 'split': 'test'}], tmp_path)
    with pytest.raises(ValueError, match='dimensions'):
        generate_scene(original, selected, [stamp], tmp_path, harmonize=lambda *args: SimpleNamespace(image=Image.new('RGB', (2, 2))))


def test_unchanged_ai_result_stays_without_fake_labels():
    original = Image.new('RGB', (30, 30), (40, 40, 40))
    scene = {'brushes': [{'points': [5, 5, 20, 20], 'width': 5}], 'brush_method': 'sdxl'}
    output = generate_scene(original, scene, [], None, lambda rgb, *args: rgb)
    assert output.polygons == [] and output.warnings
    with pytest.raises(ValueError, match='Unknown brush'):
        generate_scene(original, {**scene, 'brush_method': 'wrong'}, [], None)


def test_native_stamp_reader_rejects_invalid_rgb_and_nonbinary_masks(tmp_path):
    stamp = native_stamp(tmp_path)
    scene = {'placements': [{'stamp_id': 'native', 'x': 40, 'y': 40}], 'brushes': []}
    Image.new('L', (24, 80), 255).save(tmp_path/'stamp.tif')
    invalid = {**stamp, 'rgb_sha256': sha(tmp_path/'stamp.tif')}
    with pytest.raises(ValueError, match='RGB'):
        generate_scene(Image.new('RGB', (100, 100)), scene, [invalid], tmp_path)
    stamp = native_stamp(tmp_path)
    Image.new('L', (24, 80), 42).save(tmp_path/'mask.png')
    invalid = {**stamp, 'mask_sha256': sha(tmp_path/'mask.png')}
    with pytest.raises(ValueError, match='binary'):
        generate_scene(Image.new('RGB', (100, 100)), scene, [invalid], tmp_path)
