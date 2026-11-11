"""Build native-pixel complete-instance scratch assets from train frames only."""
import argparse
import json
import re
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from backend.images import save_image, validate_tiff
from backend.store import read_json, write_json
from scratchforge.dataset import ROOT, crop_box, mask_for, sha, translate, valid_polygons
from scratchforge.stamps import write_bundle


HOLD_ANNOTATION_IDS = frozenset({53, 194, 203, 233})


def stamp_id_for(row, annotation, revision):
    if type(annotation['id']) is not int or annotation['id'] < 0:
        raise ValueError('Invalid stamp annotation ID')
    if revision is not None:
        if not re.fullmatch(r'[a-f0-9]{32}', revision) or not re.fullmatch(r'f971-[a-f0-9]{16}', row['id']):
            raise ValueError('Invalid annotation revision or source frame')
        return f'review-{row["id"]}-{annotation["id"]}-{revision}'
    identifier = f'original-{annotation["id"]}'
    if Path(row['path']).suffix.lower() in {'.tif', '.tiff'}:
        return f'{identifier}-tiff-{row["sha256"][:16]}-native-v1'
    return identifier


def legacy_stamp(rgb, mask, output, stamp_id, metadata):
    """Keep historical JPEG datasets readable without claiming original TIFF pixels."""
    output.mkdir(parents=True, exist_ok=True)
    save_image(rgb, output / f'{stamp_id}.png', 'PNG')
    save_image(mask, output / f'{stamp_id}-mask.png', 'PNG')
    thumbnail = rgb.convert('RGBA')
    thumbnail.putalpha(mask)
    thumbnail.thumbnail((240, 180))
    save_image(thumbnail, output / f'{stamp_id}-thumbnail.png', 'PNG')
    return {**metadata, 'id': stamp_id, 'rgb': f'{stamp_id}.png', 'mask': f'{stamp_id}-mask.png',
            'thumbnail': f'{stamp_id}-thumbnail.png', 'width': rgb.width, 'height': rgb.height,
            'rgb_sha256': sha(output / f'{stamp_id}.png'),
            'mask_sha256': sha(output / f'{stamp_id}-mask.png'), 'origin': 'legacy',
            'native_parent_tiff': False,
            'warnings': ['Original TIFF parent unavailable; legacy PNG is reference/scene compatibility only.']}


def make_stamp(root, row, annotation, output, *, revision=None):
    if row['status'] != 'NG' or row['split'] != 'train':
        raise ValueError('Stamps require a verified train scratch source')
    if annotation['id'] in HOLD_ANNOTATION_IDS:
        raise ValueError('HOLD: source annotation touches the frame boundary')
    stamp_id = stamp_id_for(row, annotation, revision)
    source = root / row['path']
    with Image.open(source) as original:
        if original.size != (row['width'], row['height']) or sha(source) != row['sha256']:
            raise ValueError('Source image changed after import')
        native_tiff = original.format == 'TIFF'
        if Path(row['path']).suffix.lower() in {'.tif', '.tiff'} and not native_tiff:
            raise ValueError('Original TIFF parent is unavailable or has invalid content')
        if native_tiff:
            validate_tiff(original)
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
        if native_tiff and revision is None:
            from scratchforge.labels import coco_mask, mask_to_polygons

            mask = coco_mask(local, rgb.size)
            internal = mask_to_polygons(mask)
        else:
            mask = mask_for(local, rgb.size)
            internal = local
        metadata = dict(
            source_image=row['source_image'], source_image_id=row['source_image_id'],
            source_frame_id=row['id'], source_annotation_id=annotation['id'], split=row['split'],
            source_offset=[x, y], roi=list(box), origin='original', segmentation=local,
            review_status='APPROVED' if revision else 'PENDING_VISUAL_REVIEW',
            source_sha256=row['sha256'], annotation_revision=revision, native_parent_tiff=native_tiff)
        if not native_tiff:
            return legacy_stamp(rgb, mask, output, stamp_id, metadata)
        return write_bundle(output, stamp_id, rgb, mask, polygons=internal,
                            metadata={**metadata, 'source_segmentation': segmentation})


def upgrade_published_stamps(root, sources, output):
    from backend.segment import read_state, state_path
    from backend.store import Store

    store = Store(root)
    state = read_state(store)
    reviews = dict(state['reviews'])
    for frame_id, review in reviews.items():
        row = sources.get(frame_id)
        if (not row or review['sha256'] != row['sha256']
                or Path(row['path']).suffix.lower() not in {'.tif', '.tiff'}):
            continue
        annotations = {item['id']: item for item in review['annotations']}
        stamps, upgraded = [], []
        for stamp in review['stamps']:
            if Path(stamp['rgb']).suffix.lower() in {'.tif', '.tiff'}:
                stamps.append(stamp)
                continue
            annotation = annotations.get(stamp['source_annotation_id'])
            if annotation is None:
                raise ValueError(f"Published stamp {stamp['id']} has no reviewed source polygon")
            revision = uuid.uuid5(uuid.NAMESPACE_URL, 'scratchforge-native-v1:'+stamp['id']).hex
            native = make_stamp(root, row, annotation, output, revision=revision)
            migrated = {**native, 'segment_revision': stamp['segment_revision'], 'migrated_from': stamp['id']}
            upgraded.append(migrated)
            stamps.append(migrated)
        if upgraded:
            reviews[frame_id] = {**review, 'stamps': stamps, 'history': [*review['history'], *upgraded]}
    if reviews != state['reviews']:
        write_json(state_path(store), {**state, 'reviews': reviews})


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
        if annotation['id'] in HOLD_ANNOTATION_IDS or (points.min(0) <= 0).any() or (points.max(0) >= size - 1).any():
            hold.append(dict(annotation=annotation['id'], split=row['split'],
                             reason='Frame boundary: complete scratch extent requires review'))
            continue
        if row['split'] != 'train':
            continue
        try:
            stamps.append(make_stamp(root, row, annotation, output))
        except ValueError as error:
            hold.append(dict(annotation=annotation['id'], reason=str(error)))
    legacy_index = output / 'legacy-index.json'
    previous = read_json(output / 'index.json', [])
    current_ids = {stamp['id'] for stamp in stamps}
    legacy = {stamp['id']: stamp for stamp in read_json(legacy_index, [])}
    legacy.update({stamp['id']: stamp for stamp in previous if stamp['id'] not in current_ids})
    if legacy:
        write_json(legacy_index, list(legacy.values()))
    upgrade_published_stamps(root, {row['id']: row for row in lookup.values() if row['split'] == 'train'}, output)
    write_json(output / 'index.json', stamps)
    write_json(root / 'reports/library.json', dict(
        stamps=len(stamps), source_polygons=len(coco['annotations']), hold=hold,
        evaluation_excluded=evaluation,
        policy='Full-frame F971 sources; train-only complete native-pixel instances; pending visual review'))
    return stamps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest', type=Path, default=ROOT)
    args = parser.parse_args()
    print(f'Library: {len(build(args.dest))} train scratch assets')

