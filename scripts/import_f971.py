"""Import complete F971 frames without legacy patch dependencies."""
import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from scratchforge.dataset import ROOT, copy_checked, save_json, valid_polygons


def safe_name(name):
    if not isinstance(name, str) or not name or '/' in name or '\\' in name or ':' in name:
        raise ValueError('COCO filename must be a plain filename')
    if name in {'.', '..'} or name.endswith((' ', '.')):
        raise ValueError('Unsafe COCO filename')
    return name


def load_splits(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    splits = {}
    for row in rows:
        if row['product_code'] != 'F971':
            continue
        key = (row['source_status'], safe_name(row['source_image']))
        split = {'validation': 'val'}.get(row['split'], row['split'])
        if key[0] not in {'NG', 'OK'} or split not in {'train', 'val', 'test'}:
            raise ValueError('Unknown F971 status or split')
        if key in splits:
            raise ValueError(f'Duplicate source split: {key}')
        splits[key] = {**row, 'split': split}
    return splits


def validate_coco(coco):
    images = {image['id']: image for image in coco['images']}
    if len(images) != len(coco['images']):
        raise ValueError('Duplicate COCO image ID')
    names = [safe_name(image['file_name']) for image in images.values()]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate COCO filename')
    categories = {category['id']: category['name'] for category in coco['categories']}
    annotation_ids = set()
    for annotation in coco['annotations']:
        image = images.get(annotation['image_id'])
        if annotation['id'] in annotation_ids or image is None:
            raise ValueError('Duplicate annotation ID or unknown image')
        annotation_ids.add(annotation['id'])
        if categories.get(annotation['category_id']) != 'scratch':
            raise ValueError('Annotation must belong to scratch category')
        try:
            valid = valid_polygons(annotation['segmentation'], image['width'], image['height'])
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError(f'Invalid polygon: {annotation["id"]}')
    return images


def inspect_frames(source, coco, splits):
    ng_images = validate_coco(coco)
    by_name = {image['file_name']: image for image in ng_images.values()}
    result = []
    next_id = max(ng_images, default=-1) + 1
    for status in ('NG', 'OK'):
        folder = source / 'F971' / status / 'images'
        files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg'})
        if not files:
            raise ValueError(f'No {status} full-frame images')
        if status == 'NG' and {p.name for p in files} != set(by_name):
            raise ValueError('NG files and COCO inventory differ')
        for path in files:
            row = splits.get((status, path.name))
            if row is None:
                raise ValueError(f'Missing source split: {path.name}')
            image = by_name.get(path.name) if status == 'NG' else None
            with Image.open(path) as rgb:
                rgb.verify()
                width, height = rgb.size
                if rgb.mode != 'RGB':
                    raise ValueError(f'Expected RGB image: {path.name}')
            if image and (width, height) != (image['width'], image['height']):
                raise ValueError(f'COCO dimensions differ: {path.name}')
            if image and row.get('source_image_id') not in {None, '', str(image['id'])}:
                raise ValueError(f'Source image ID mismatch: {path.name}')
            image_id = image['id'] if image else next_id
            if not image:
                next_id += 1
            result.append(dict(source=path, status=status, source_image=path.name,
                               source_image_id=image['id'] if image else None,
                               coco_image_id=image_id, split=row['split'], width=width, height=height))
    if {(r['status'], r['source_image']) for r in result} != set(splits):
        raise ValueError('Split inventory differs from full-frame inventory')
    return result


def import_dataset(source=ROOT.parent / 'GAN_Project', dest=ROOT):
    source, dest = Path(source).resolve(), Path(dest).resolve()
    if source == dest or source in dest.parents or dest in source.parents:
        raise ValueError('Source and destination must not overlap')
    annotation_path = source / 'F971/NG/annotations/coco.json'
    split_path = source / 'reports/source_image_splits.csv'
    coco = json.loads(annotation_path.read_text(encoding='utf-8-sig'))
    frames = inspect_frames(source, coco, load_splits(split_path))
    dest.mkdir(parents=True, exist_ok=True)
    required = sum(frame['source'].stat().st_size for frame in frames
                   if not (dest / 'data/raw/F971' / frame['status'] / 'images' / frame['source_image']).exists())
    if shutil.disk_usage(dest).free < required + 128 * 1024 ** 2:
        raise ValueError('Insufficient space for full-frame import')
    manifest, provenance, images = [], [], []
    for frame in frames:
        relative = Path('data/raw/F971') / frame['status'] / 'images' / frame['source_image']
        digest = copy_checked(frame['source'], dest / relative)
        token = hashlib.sha256(f"{frame['status']}/{frame['source_image']}".encode()).hexdigest()[:16]
        record = {key: value for key, value in frame.items() if key != 'source'}
        manifest.append({**record, 'id': f'f971-{token}', 'path': relative.as_posix(), 'sha256': digest})
        provenance.append(dict(old_path=str(frame['source']), new_path=relative.as_posix(),
                               sha256=digest, bytes=frame['source'].stat().st_size))
        images.append(dict(id=frame['coco_image_id'], file_name=relative.as_posix(),
                           width=frame['width'], height=frame['height'],
                           status=frame['status'], split=frame['split']))
    for metadata, target in [(annotation_path, 'data/raw/F971/NG/annotations/coco.json'),
                             (split_path, 'data/reference/source_image_splits.csv')]:
        digest = copy_checked(metadata, dest / target)
        provenance.append(dict(old_path=str(metadata), new_path=target,
                               sha256=digest, bytes=metadata.stat().st_size))
    save_json(dest / 'data/fullframe/index.json', manifest)
    save_json(dest / 'data/fullframe/provenance.json', provenance)
    save_json(dest / 'data/fullframe/coco.json', {**coco, 'images': images})
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT.parent / 'GAN_Project')
    parser.add_argument('--dest', type=Path, default=ROOT)
    args = parser.parse_args()
    records = import_dataset(args.source, args.dest)
    print(f'Imported {len(records)} complete F971 frames')
