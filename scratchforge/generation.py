"""Full-frame composition; temporary inference regions never truncate instances."""
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from scratchforge.ai import procedural_scratch

from scratchforge.engine import composite
from scratchforge.dataset import sha


@dataclass(frozen=True)
class SceneOutput:
    image: Image.Image
    guidance: Image.Image
    edit: Image.Image
    polygons: list
    regions: list
    layers: list
    warnings: tuple

    def __iter__(self):
        return iter((self.image, self.guidance, self.edit, self.polygons))


def build_regions(boxes, size, context=64):
    width, height = size
    pending = [(max(0, x-context), max(0, y-context), min(width, xx+context), min(height, yy+context))
               for x, y, xx, yy in boxes]
    merged = []
    while pending:
        box = pending.pop(0)
        overlaps = [other for other in merged if box[0] <= other[2] and other[0] <= box[2]
                    and box[1] <= other[3] and other[1] <= box[3]]
        if overlaps:
            group = [box, *overlaps]
            merged = [other for other in merged if other not in overlaps]
            pending.insert(0, (min(b[0] for b in group), min(b[1] for b in group),
                               max(b[2] for b in group), max(b[3] for b in group)))
        else:
            merged.append(box)
    return sorted(merged)


def safe_asset(directory, name):
    directory = Path(directory).resolve()
    path = (directory / name).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError('Invalid library asset')
    return path


def _place(original, placement, stamp, directory):
    rgb_path = safe_asset(directory, stamp['rgb'])
    mask_path = safe_asset(directory, stamp['mask'])
    for key, path in [('rgb_sha256', rgb_path), ('mask_sha256', mask_path)]:
        if not stamp.get(key) or sha(path) != stamp[key]:
            raise ValueError('Stamp asset checksum missing or changed; rebuild the scratch library')
    with Image.open(rgb_path) as source:
        rgba = source.convert('RGBA')
    with Image.open(mask_path) as source:
        alpha = source.convert('L')
    if rgba.size != alpha.size:
        raise ValueError('Stamp mask dimensions mismatch')
    x, y = placement['x'], placement['y']
    rgba.putalpha(alpha)
    theta = math.radians(placement.get('rotation', 0)); scale = placement.get('scale', 1)
    c, sn = math.cos(theta), math.sin(theta); flip = -1 if placement.get('flip', False) else 1
    a, b, d, e = flip*c/scale, flip*sn/scale, -sn/scale, c/scale
    coefficients = (a, b, rgba.width/2-a*x-b*y, d, e, rgba.height/2-d*x-e*y)
    warped = rgba.transform(original.size, Image.Transform.AFFINE, coefficients, Image.Resampling.BILINEAR)
    mask = warped.getchannel('A').point(lambda value: 255 if value >= 128 else 0)
    return warped.convert('RGB'), mask


def brush_method(scene):
    return scene.get('brush_method') or ('sdxl' if scene.get('engine') == 'sdxl' else 'procedural')


def brush_mask(original, brushes):
    guidance = Image.new('L', original.size)
    boxes = []
    for brush in brushes:
        mask = Image.new('L', original.size)
        ImageDraw.Draw(mask).line(list(zip(brush['points'][::2], brush['points'][1::2])),
                                 fill=255, width=round(brush.get('width', 12)), joint='curve')
        box = mask.getbbox()
        if box:
            boxes.append(box)
            guidance.paste(255, (0, 0), mask)
    return guidance, boxes


def edit_mask(guidance, boxes, margin):
    edit = Image.new('L', guidance.size)
    for box in build_regions(boxes, guidance.size, margin):
        local = guidance.crop(box)
        if margin:
            local = local.filter(ImageFilter.MaxFilter(margin*2+1))
        edit.paste(local, box[:2])
    return edit


def _brush_branch(original, scene, infer):
    guidance, boxes = brush_mask(original, scene.get('brushes', []))
    method = brush_method(scene)
    margin = scene.get('edit_margin', 24)
    edit = guidance.copy() if method == 'procedural' else edit_mask(guidance, boxes, margin)
    regions = build_regions(boxes, original.size, margin+96)
    result = original.copy(); layers = []
    for box in regions:
        adapter = procedural_scratch if method == 'procedural' else infer
        if adapter is None:
            raise ValueError(f'{method} inference is unavailable')
        local_mask = (guidance if method == 'procedural' else edit).crop(box)
        generated = adapter(original.crop(box), local_mask, scene.get('seed', 0), scene.get('strength', .65))
        if generated.size != (box[2]-box[0], box[3]-box[1]):
            raise ValueError('AI returned incorrect region dimensions')
        result.paste(generated, box[:2], local_mask)
        layers.append({'kind': 'brush', 'requested_method': method, 'effective_method': method,
                       'roi': list(box), 'warnings': [], 'metadata': {'native_size': list(local_mask.size)}})
    return result, guidance, edit, regions, layers


def _stamp_branch(original, scene, stamps, directory, harmonize):
    lookup = {stamp['id']: stamp for stamp in stamps}
    result = original.copy(); support = Image.new('L', original.size)
    layers = []; regions = []; warnings = []
    for placement in scene.get('placements', []):
        stamp = lookup.get(placement['stamp_id'])
        if stamp is None:
            raise ValueError('Unknown stamp')
        if stamp.get('split') != 'train':
            raise ValueError('Only train stamps are available')
        warped, mask = _place(original, placement, stamp, directory)
        bounds = mask.getbbox()
        if bounds is None:
            continue
        box = build_regions([bounds], original.size, scene.get('edit_margin', 24))[0]
        background = original.crop(box); local_mask = mask.crop(box)
        placed = Image.composite(warped.crop(box), background, local_mask)
        method = scene.get('stamp_method', 'classical_mixed')
        corrected = harmonize(placed, local_mask, background, method)
        if corrected.image.size != background.size:
            raise ValueError('Stamp harmonization returned incorrect native region dimensions')
        result.paste(corrected.image, box[:2], local_mask)
        support.paste(255, (0, 0), mask)
        regions.append(box); warnings.extend(corrected.warnings)
        layers.append({'kind': 'stamp', 'stamp_id': stamp['id'], 'roi': list(box),
                       'requested_method': corrected.requested_method, 'effective_method': corrected.effective_method,
                       'warnings': list(corrected.warnings), 'metadata': corrected.metadata})
    return result, support, regions, layers, warnings


def generate_scene(original, scene, stamps, directory, infer=None, harmonize=None):
    if harmonize is None and scene.get('placements'):
        from scratchforge.harmonization import StampHarmonizer
        harmonize = StampHarmonizer()
    brush_rgb, brush_guidance, brush_edit, brush_regions, brush_layers = _brush_branch(original, scene, infer)
    stamp_rgb, stamp_mask, stamp_regions, stamp_layers, warnings = _stamp_branch(
        original, scene, stamps, directory, harmonize)
    if not brush_guidance.getbbox() and not stamp_mask.getbbox():
        raise ValueError('Place or draw at least one visible scratch')
    # Each branch sees the original image. Stamp cores win overlaps and never traverse a VAE.
    result = brush_rgb.copy(); result.paste(stamp_rgb, (0, 0), stamp_mask)
    guidance = Image.fromarray(np.maximum(np.asarray(brush_guidance), np.asarray(stamp_mask)))
    edit = Image.fromarray(np.maximum(np.asarray(brush_edit), np.asarray(stamp_mask)))
    return SceneOutput(composite(original, result, edit), guidance, edit, [],
                       [*brush_regions, *stamp_regions], [*brush_layers, *stamp_layers], tuple(dict.fromkeys(warnings)))
