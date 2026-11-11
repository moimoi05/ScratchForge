"""Immutable, native-pixel stamp bundles and synchronized geometric augmentation."""
import json
import re
import shutil
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from backend.images import MAX_TIFF_PIXELS, save_image, save_tiff
from scratchforge.dataset import mask_for, sha, valid_polygons


ASSET_KEYS = ('rgb', 'mask', 'thumbnail', 'preview_alpha', 'polygon', 'metadata')


def validate_pair(rgb, mask):
    if rgb.mode != 'RGB' or mask.mode != 'L' or rgb.size != mask.size:
        raise ValueError('Stamp requires matching RGB image and grayscale mask')
    if rgb.width * rgb.height > MAX_TIFF_PIXELS:
        raise ValueError('Stamp exceeds the supported pixel limit')
    values = np.unique(np.asarray(mask))
    if not set(values.tolist()) <= {0, 255} or 255 not in values:
        raise ValueError('Stamp mask must be nonempty binary black/white')


def write_bundle(directory, identifier, rgb, mask, *, polygons=None, metadata=None):
    from scratchforge.labels import coco_document, mask_to_polygons

    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,149}', identifier):
        raise ValueError('Invalid stamp identifier')
    validate_pair(rgb, mask)
    polygons = mask_to_polygons(mask) if polygons is None else polygons
    if not valid_polygons(polygons, *rgb.size):
        raise ValueError('Invalid stamp polygons')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / identifier
    stage = directory / ('.stage-' + uuid.uuid4().hex)
    stage.mkdir()
    try:
        save_tiff(rgb, stage / 'stamp_crop.tif')
        save_image(mask, stage / 'mask.png', 'PNG')
        alpha = rgb.convert('RGBA')
        alpha.putalpha(mask)
        save_image(alpha, stage / 'preview_alpha.png', 'PNG')
        polygon_path = stage / 'polygon.json'
        polygon_path.write_text(json.dumps(coco_document(polygons, rgb.size, 'stamp_crop.tif'),
                                           ensure_ascii=False, indent=2), encoding='utf-8')
        assets = {'rgb': 'stamp_crop.tif', 'mask': 'mask.png', 'preview_alpha': 'preview_alpha.png',
                  'thumbnail': 'preview_alpha.png', 'polygon': 'polygon.json', 'metadata': 'metadata.json'}
        record = {**(metadata or {}), 'id': identifier, 'width': rgb.width, 'height': rgb.height,
                  'segmentation': polygons, **{key: f'{identifier}/{name}' for key, name in assets.items()},
                  **{key + '_sha256': sha(stage / assets[key])
                     for key in ('rgb', 'mask', 'preview_alpha', 'polygon')}}
        record.setdefault('origin', 'original')
        record.setdefault('split', 'train')
        record.setdefault('review_status', 'PENDING_VISUAL_REVIEW')
        (stage / 'metadata.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        record = {**record, 'metadata_sha256': sha(stage / 'metadata.json')}
        if destination.exists():
            previous = json.loads((destination / 'metadata.json').read_text(encoding='utf-8'))
            previous = {**previous, 'metadata_sha256': sha(destination / 'metadata.json')}
            if previous != record or any(sha(destination / assets[key]) != record[key + '_sha256']
                                         for key in ('rgb', 'mask', 'preview_alpha', 'polygon')):
                raise ValueError('Stamp bundles are immutable; use a new revision')
            return previous
        stage.replace(destination)
        return record
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def augment_pair(rgb, mask, seed):
    """One expanded inverse map drives texture and binary geometry together."""
    import cv2

    validate_pair(rgb, mask)
    rng = np.random.default_rng(seed)
    angle, scale = float(rng.uniform(-28, 28)), float(rng.uniform(.9, 1.1))
    pad = int(np.ceil(max(rgb.size) * .35)) + 32
    width, height = rgb.width + 2 * pad, rgb.height + 2 * pad
    if width * height > MAX_TIFF_PIXELS:
        raise ValueError('Expanded augmentation exceeds the supported pixel limit')
    transform = cv2.getRotationMatrix2D((rgb.width / 2, rgb.height / 2), angle, scale)
    transform[:, 2] += pad
    affine_rgb = cv2.warpAffine(np.asarray(rgb), transform, (width, height),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    affine_mask = cv2.warpAffine(np.asarray(mask), transform, (width, height), flags=cv2.INTER_NEAREST)
    xx, yy = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    amplitude = min(rgb.size) * .035
    elastic_x = cv2.resize(rng.normal(0, amplitude, (5, 5)).astype('float32'),
                          (width, height), interpolation=cv2.INTER_CUBIC)
    elastic_y = cv2.resize(rng.normal(0, amplitude, (5, 5)).astype('float32'),
                          (width, height), interpolation=cv2.INTER_CUBIC)
    controls = rng.uniform(-min(rgb.size) * .08, min(rgb.size) * .08, 4)
    t = np.clip((yy - pad) / max(1, rgb.height - 1), 0, 1)
    bend = ((1-t)**3*controls[0] + 3*(1-t)**2*t*controls[1]
            + 3*(1-t)*t*t*controls[2] + t**3*controls[3])
    map_x, map_y = (xx + elastic_x + bend).astype('float32'), (yy + elastic_y).astype('float32')
    warped_rgb = cv2.remap(affine_rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    warped_mask = cv2.remap(affine_mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    output_mask = Image.fromarray(warped_mask)
    bounds = output_mask.getbbox()
    if not bounds or bounds[0] == 0 or bounds[1] == 0 or bounds[2] == width or bounds[3] == height:
        raise ValueError('Augmentation lost or clipped the scratch instance')
    x, y, right, bottom = bounds
    crop = (max(0, x-16), max(0, y-16), min(width, right+16), min(height, bottom+16))
    return (Image.fromarray(warped_rgb).crop(crop), output_mask.crop(crop),
            {'seed': int(seed), 'rotation': angle, 'scale': scale, 'elastic_grid': [5, 5],
             'elastic_amplitude': amplitude, 'spline_controls': controls.tolist(), 'crop': list(crop)})
