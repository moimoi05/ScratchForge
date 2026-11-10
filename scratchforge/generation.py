"""Full-frame composition; temporary inference regions never truncate instances."""
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from scratchforge.engine import composite
from scratchforge.dataset import sha


@dataclass(frozen=True)
class SceneOutput:
    image: Image.Image
    guidance: Image.Image
    edit: Image.Image
    polygons: list
    regions: list

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
    # Preserve the observed scratch contrast while adjusting the local background tint.
    source_array = np.asarray(rgba.convert('RGB'), dtype=np.float32)
    mask_array = np.asarray(alpha) > 0
    x, y = placement['x'], placement['y']
    sample_x = max(0, min(original.width-1, int(x)))
    sample_y = max(0, min(original.height-1, int(y)))
    target = original.crop((max(0, sample_x-32), max(0, sample_y-32),
                            min(original.width, sample_x+33), min(original.height, sample_y+33)))
    if target.width and target.height and (~mask_array).any():
        offset = np.median(np.asarray(target).reshape(-1, 3), axis=0) - np.median(source_array[~mask_array], axis=0)
        rgba = Image.fromarray(np.clip(source_array + offset, 0, 255).astype('uint8')).convert('RGBA')
    rgba.putalpha(alpha)
    theta = math.radians(placement.get('rotation', 0)); scale = placement.get('scale', 1)
    c, sn = math.cos(theta), math.sin(theta); flip = -1 if placement.get('flip', False) else 1
    a, b, d, e = flip*c/scale, flip*sn/scale, -sn/scale, c/scale
    coefficients = (a, b, rgba.width/2-a*x-b*y, d, e, rgba.height/2-d*x-e*y)
    warped = rgba.transform(original.size, Image.Transform.AFFINE, coefficients, Image.Resampling.BILINEAR)
    mask = warped.getchannel('A').point(lambda value: 255 if value >= 128 else 0)
    return warped.convert('RGB'), mask


def generate_scene(original, scene, stamps, directory, infer=None):
    lookup = {stamp['id']: stamp for stamp in stamps}
    result = original.copy(); boxes = []; guidance = Image.new('L', original.size)
    for placement in scene.get('placements', []):
        if placement['stamp_id'] not in lookup:
            raise ValueError('Unknown stamp')
        stamp = lookup[placement['stamp_id']]
        if stamp.get('split') != 'train':
            raise ValueError('Only train stamps are available')
        warped, mask = _place(original, placement, stamp, directory)
        result.paste(warped, (0, 0), mask)
        if mask.getbbox():
            boxes.append(mask.getbbox())
        guidance.paste(255, (0, 0), mask)
    for brush in scene.get('brushes', []):
        mask = Image.new('L', original.size)
        ImageDraw.Draw(mask).line(list(zip(brush['points'][::2], brush['points'][1::2])),
                                 fill=255, width=round(brush['width']), joint='curve')
        result.paste((180, 155, 115), (0, 0, *original.size), mask)
        if mask.getbbox():
            boxes.append(mask.getbbox())
        guidance.paste(255, (0, 0), mask)
    if not boxes:
        raise ValueError('Place or draw at least one visible scratch')
    margin = scene.get('edit_margin', 24)
    edit = Image.new('L', original.size)
    # Only the bounding region is filtered, avoiding a full-frame large-kernel operation.
    for box in build_regions(boxes, original.size, margin):
        local = guidance.crop(box)
        if margin:
            local = local.filter(ImageFilter.MaxFilter(margin*2+1))
        edit.paste(local, box[:2])
    regions = build_regions(boxes, original.size, margin+96)
    if scene.get('engine', 'baseline') == 'sdxl':
        if infer is None:
            raise ValueError('SDXL inference is unavailable')
        for box in regions:
            generated = infer(result.crop(box), edit.crop(box), scene.get('seed', 0), scene.get('strength', .65))
            if generated.size != (box[2]-box[0], box[3]-box[1]):
                raise ValueError('AI returned incorrect region dimensions')
            result.paste(generated, box[:2], edit.crop(box))
    else:
        edit = guidance.copy()
    return SceneOutput(composite(original, result, edit), guidance, edit, [], regions)
