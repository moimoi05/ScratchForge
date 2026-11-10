"""Build native-pixel complete-instance scratch assets from train frames only."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from scratchforge.dataset import ROOT, crop_box, mask_for, save_json, sha, translate, valid_polygons


def make_stamp(root, row, annotation, output):
    source = root / row['path']
    with Image.open(source) as original:
        if original.size != (row['width'], row['height']) or sha(source) != row['sha256']:
            raise ValueError('Source image changed after import')
        segmentation = annotation['segmentation']
        if not valid_polygons(segmentation, *original.size):
            raise ValueError('Invalid polygon')
        points = np.concatenate([np.array(p).reshape(-1, 2) for p in segmentation])
        if (points.min(0) <= 0).any() or (points.max(0) >= np.array(original.size) - 1).any():
            raise ValueError('Frame boundary: complete scratch extent requires review')
        preferred = 512 if np.ptp(points, axis=0).max() > 224 else 256
        box = crop_box(segmentation, original.size, preferred)
        x, y, x2, y2 = box
        local = translate(segmentation, x, y)
        rgb = original.crop(box)
        mask = mask_for(local, rgb.size)
        stamp_id = f'original-{annotation["id"]}'
        rgb.save(output / f'{stamp_id}.png')
        mask.save(output / f'{stamp_id}-mask.png')
        thumbnail = rgb.convert('RGBA')
        thumbnail.putalpha(mask)
        thumbnail.thumbnail((240, 180))
        thumbnail.save(output / f'{stamp_id}-thumbnail.png')
    return dict(id=stamp_id, rgb=f'{stamp_id}.png', mask=f'{stamp_id}-mask.png',
                rgb_sha256=sha(output / f'{stamp_id}.png'),
                mask_sha256=sha(output / f'{stamp_id}-mask.png'),
                thumbnail=f'{stamp_id}-thumbnail.png', width=x2-x, height=y2-y,
                source_image=row['source_image'], source_image_id=row['source_image_id'],
                source_frame_id=row['id'], source_annotation_id=annotation['id'], split=row['split'],
                source_offset=[x, y], roi=box, segmentation=local,
                review_status='PENDING_VISUAL_REVIEW', source_sha256=row['sha256'])


def build(root=ROOT):
    root = Path(root)
    coco = json.loads((root / 'data/raw/F971/NG/annotations/coco.json').read_text())
    records = json.loads((root / 'data/fullframe/index.json').read_text())
    lookup = {row['source_image_id']: row for row in records if row['status'] == 'NG'}
    output = root / 'data/library'
    output.mkdir(parents=True, exist_ok=True)
    stamps, hold, evaluation = [], [], []
    for annotation in coco['annotations']:
        row = lookup.get(annotation['image_id'])
        if row is None:
            raise ValueError('Annotation has no imported full-frame source')
        if row['split'] != 'train':
            evaluation.append(dict(annotation=annotation['id'], split=row['split']))
        points = np.concatenate([np.array(p).reshape(-1, 2) for p in annotation['segmentation']])
        size = np.array([row['width'], row['height']])
        if (points.min(0) <= 0).any() or (points.max(0) >= size - 1).any():
            hold.append(dict(annotation=annotation['id'], split=row['split'],
                             reason='Frame boundary: complete scratch extent requires review'))
            continue
        if row['split'] != 'train':
            continue
        try:
            stamps.append(make_stamp(root, row, annotation, output))
        except ValueError as error:
            hold.append(dict(annotation=annotation['id'], reason=str(error)))
    save_json(output / 'index.json', stamps)
    save_json(root / 'reports/library.json', dict(
        stamps=len(stamps), source_polygons=len(coco['annotations']), hold=hold,
        evaluation_excluded=evaluation,
        policy='Full-frame F971 sources; train-only complete native-pixel instances; pending visual review'))
    return stamps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest', type=Path, default=ROOT)
    args = parser.parse_args()
    print(f'Library: {len(build(args.dest))} train scratch assets')

