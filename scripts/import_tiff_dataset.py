"""Migrate F971 generation sources and seed Segment from native TIFF files."""
import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image

from backend.images import validate_tiff
from backend.store import read_json, write_json
from scratchforge.dataset import ROOT, copy_checked, sha, valid_polygons
from scripts.import_f971 import load_splits, safe_name


MAX_COCO_ID = (1 << 53) - 1
TIFF_SUFFIXES = {'.tif', '.tiff'}
EXTRA_SCRATCH_STEM = '2026-09-09-14-21-22'
GROUPS = {'NG': 'f971-scratch', 'OK': 'f971-ok', 'S942': 's942-ok'}
IMAGE_FOLDERS = {'NG': 'F971_Scratch', 'OK': '2026-09-14/ORG/MATTRUOC/CAM2/DOME-1', 'S942': 'S942_OK'}
DEFAULT_COCO = ROOT.parent / 'GAN Industry Scratch.v1i.coco-segmentation/train/_annotations.coco.json'
ROBOFLOW_SUFFIX = re.compile(r'_(?:jpe?g|png|tiff?)\.rf\.[a-f\d]+\.(?:jpe?g|png|tiff?)$', re.I)


def inspect_tiff(path):
    try:
        with Image.open(path) as image:
            validate_tiff(image)
            image.load()
            return image.size
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError(f'Invalid or oversized TIFF: {path.name}') from error
    except ValueError as error:
        raise ValueError(f'{error}: {path.name}') from error


def tiff_files(folder):
    paths = {}
    if not folder.is_dir():
        raise ValueError(f'Missing TIFF directory: {folder}')
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in TIFF_SUFFIXES:
            continue
        key = path.stem.casefold()
        if key in paths:
            raise ValueError(f'Duplicate TIFF image stem: {path.name}')
        if not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError(f'TIFF points outside source directory: {path.name}')
        paths[key] = path
    return paths


def _valid_id(value):
    return type(value) is int and 0 <= value <= MAX_COCO_ID


def _objects_by_id(coco, section):
    entries = coco.get(section)
    if not isinstance(entries, list):
        raise ValueError(f'COCO {section} must be a list')
    result = {}
    for entry in entries:
        if not isinstance(entry, dict) or not _valid_id(entry.get('id')):
            raise ValueError(f'Invalid COCO {section} ID')
        if entry['id'] in result:
            raise ValueError(f'Duplicate COCO {section} ID: {entry["id"]}')
        result[entry['id']] = entry
    return result


def _coco_stems(image):
    filename = safe_name(image.get('file_name'))
    extra = image.get('extra')
    hashed = ROBOFLOW_SUFFIX.sub('', filename)
    names = [Path(filename).stem if hashed == filename else hashed]
    if isinstance(extra, dict) and 'name' in extra:
        names.insert(0, Path(safe_name(extra['name'])).stem)
    return {name.casefold() for name in names}


def match_coco(coco, rows, *, ignored_names=()):
    """Match COCO entries to TIFF rows and return unchanged source polygons by frame ID."""
    if not isinstance(coco, dict):
        raise ValueError('Expected a COCO object')
    ignored_names = frozenset(ignored_names)
    categories = _objects_by_id(coco, 'categories')
    images = _objects_by_id(coco, 'images')
    annotations = defaultdict(list)
    for item in _objects_by_id(coco, 'annotations').values():
        image_id, category_id = item.get('image_id'), item.get('category_id')
        if not _valid_id(image_id):
            raise ValueError(f'Invalid COCO image_id for annotation {item["id"]}')
        if not _valid_id(category_id):
            raise ValueError(f'Invalid COCO category_id for annotation {item["id"]}')
        if image_id not in images:
            raise ValueError(f'Unknown COCO image for annotation {item["id"]}')
        if categories.get(category_id, {}).get('name') != 'scratch':
            raise ValueError(f'Expected scratch category for annotation {item["id"]}')
        annotations[image_id].append(item)
    by_stem = {Path(row['file_name']).stem.casefold(): row for row in rows}
    if len(by_stem) != len(rows):
        raise ValueError('Duplicate TIFF image stem in catalog')
    matched = {}
    for image_id, image in images.items():
        stems = _coco_stems(image)
        candidates = stems & by_stem.keys()
        if stems & ignored_names:
            if candidates or annotations[image_id]:
                raise ValueError(f'COCO has annotations or conflicting names for excluded image: {image["file_name"]}')
            continue
        if not candidates:
            raise ValueError(f'Missing TIFF for COCO image: {image["file_name"]}')
        if len(candidates) != 1:
            raise ValueError(f'Ambiguous COCO image names: {image["file_name"]}')
        row = by_stem[next(iter(candidates))]
        if row['id'] in matched:
            raise ValueError(f'Duplicate COCO image match: {row["file_name"]}')
        dimensions = image.get('width'), image.get('height')
        if any(type(value) is not int for value in dimensions) or dimensions != (row['width'], row['height']):
            raise ValueError(f'COCO dimensions differ for {row["file_name"]}')
        matched[row['id']] = [_validate_polygon(item, dimensions) for item in annotations[image_id]]
    return matched


def _validate_polygon(item, dimensions):
    rings = item.get('segmentation')
    try:
        valid = (isinstance(rings, list) and all(isinstance(ring, list) and all(
            type(value) in (int, float) for value in ring) for ring in rings) and valid_polygons(rings, *dimensions))
    except (OverflowError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f'Invalid polygon: COCO annotation {item["id"]}')
    return {key: item[key] for key in ('id', 'category_id', 'segmentation')}


def _frame_records(files, splits, original, prior):
    ng_images = _objects_by_id(original, 'images')
    ng_by_name = {safe_name(row.get('file_name')): row for row in ng_images.values()}
    if len(ng_by_name) != len(ng_images) or set(ng_by_name) != {name for status, name in splits if status == 'NG'}:
        raise ValueError('Source NG COCO and split inventory differ')
    next_id = max(ng_images, default=-1) + 1
    ok_count = sum(status == 'OK' for status, _ in splits)
    if next_id + ok_count - 1 > MAX_COCO_ID:
        raise ValueError('Not enough safe COCO image IDs for F971 OK frames')
    prior_by_id = {row['id']: row for row in prior}
    records = []
    for status in ('NG', 'OK'):
        names = sorted(name for kind, name in splits if kind == status)
        expected = {Path(name).stem.casefold() for name in names}
        if not names or len(expected) != len(names):
            raise ValueError(f'Missing or duplicate {status} source stems in split inventory')
        if expected - files[status].keys():
            raise ValueError(f'Missing TIFF for {status}: {", ".join(sorted(expected - files[status].keys()))}')
        if files[status].keys() - expected:
            raise ValueError(f'TIFF outside verified {status} split inventory')
        for name in names:
            split = splits[status, name]
            path = files[status][Path(name).stem.casefold()]
            metadata = ng_by_name.get(name) if status == 'NG' else None
            image_id = metadata['id'] if metadata else next_id
            next_id += int(metadata is None)
            if metadata and split.get('source_image_id') not in (None, '', str(image_id)):
                raise ValueError(f'Source image ID differs for {name}')
            identifier = 'f971-' + hashlib.sha256(f'{status}/{name}'.encode()).hexdigest()[:16]
            width, height = inspect_tiff(path)
            reference = metadata or prior_by_id.get(identifier)
            if reference and (width, height) != (reference['width'], reference['height']):
                raise ValueError(f'TIFF dimensions differ for {name}')
            records.append(dict(id=identifier, source_image=name, file_name=path.name, status=status,
                                split=split['split'], source_image_id=image_id if metadata else None,
                                coco_image_id=image_id, width=width, height=height, source=path))
    return records


def _validate_geometry(records, original, coco):
    scratch = [row for row in records if row['status'] == 'NG']
    matched = match_coco(coco, scratch, ignored_names={EXTRA_SCRATCH_STEM})
    expected = match_coco(original, scratch)
    for row in scratch:
        identifier = row['id']
        if identifier not in matched:
            raise ValueError(f'COCO missing verified TIFF image: {row["file_name"]}')
        if {item['id']: item for item in matched[identifier]} != {item['id']: item for item in expected[identifier]}:
            raise ValueError(f'Roboflow geometry differs from verified source for {row["file_name"]}')
    return matched


def _external_rows(paths):
    rows = []
    for key, path in paths.items():
        width, height = inspect_tiff(path)
        identifier = 's942-' + hashlib.sha256(key.encode()).hexdigest()[:16]
        rows.append(dict(id=identifier, file_name=path.name, source_image=path.name, status='OK',
                         split=None, source_image_id=None, width=width, height=height,
                         path=str(path.resolve()), sha256=sha(path), group=GROUPS['S942'], annotations=[]))
    return rows


def _copy_with_provenance(source, dest, relative):
    return dict(old_path=str(source), new_path=relative.as_posix(),
                sha256=copy_checked(source, dest / relative), bytes=source.stat().st_size)


def migrate(source, image_root, coco_path, dest=ROOT):
    source, image_root, coco_path, dest = (Path(path).resolve() for path in (source, image_root, coco_path, dest))
    for folder in (source, image_root):
        if folder == dest or folder in dest.parents or dest in folder.parents:
            raise ValueError('TIFF sources and destination must not overlap')
    annotation_path = source / 'F971/NG/annotations/coco.json'
    split_path = source / 'reports/source_image_splits.csv'
    original = json.loads(annotation_path.read_text(encoding='utf-8-sig'))
    if not isinstance(original, dict):
        raise ValueError('Expected a COCO object for authoritative F971 annotations')
    roboflow = json.loads(coco_path.read_text(encoding='utf-8-sig'))
    files = {key: tiff_files(image_root / folder) for key, folder in IMAGE_FOLDERS.items()}
    files['NG'].pop(EXTRA_SCRATCH_STEM, None)
    fullframe = dest / 'data/fullframe'
    previous = read_json(fullframe / 'index.json', [])
    records = _frame_records(files, load_splits(split_path), original, previous)
    matched = _validate_geometry(records, original, roboflow)
    catalog = _external_rows(files['S942'])
    active, images, provenance = _copy_f971(records, dest)
    for path, relative in ((annotation_path, 'data/raw/F971/NG/annotations/coco.json'),
                           (split_path, 'data/reference/source_image_splits.csv')):
        provenance.append(_copy_with_provenance(path, dest, Path(relative)))
    catalog = [{**row, 'group': GROUPS[row['status']], 'annotations': matched.get(row['id'], [])}
               for row in active] + catalog
    legacy = [row for row in previous if Path(row['path']).suffix.lower() in {'.jpg', '.jpeg'}]
    if legacy and not (fullframe / 'legacy-index.json').exists():
        write_json(fullframe / 'legacy-index.json', legacy)
    write_json(fullframe / 'coco.json', {**original, 'images': images})
    write_json(fullframe / 'provenance.json', provenance)
    write_json(dest / 'data/segment/index.json', catalog)
    write_json(fullframe / 'index.json', active)
    return active


def _copy_f971(records, dest):
    active, images, provenance = [], [], []
    paths = [(record, Path('data/raw/F971') / record['status'] / 'images' / record['file_name']) for record in records]
    dest.mkdir(parents=True, exist_ok=True)
    required = sum(record['source'].stat().st_size for record, relative in paths if not (dest / relative).exists())
    if shutil.disk_usage(dest).free < required + 128 * 1024 ** 2:
        raise ValueError('Insufficient disk space for original TIFF files')
    for record, relative in paths:
        entry = _copy_with_provenance(record['source'], dest, relative)
        provenance.append(entry)
        row = {key: value for key, value in record.items() if key != 'source'}
        active.append({**row, 'path': entry['new_path'], 'sha256': entry['sha256']})
        images.append(dict(id=row['coco_image_id'], file_name=entry['new_path'], width=row['width'],
                           height=row['height'], status=row['status'], split=row['split']))
    return active, images, provenance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT.parent / 'GAN_Project')
    parser.add_argument('--image-root', type=Path, default=ROOT.parent / 'image')
    parser.add_argument('--coco', type=Path, default=DEFAULT_COCO)
    parser.add_argument('--dest', type=Path, default=ROOT)
    args = parser.parse_args()
    rows = migrate(args.source, args.image_root, args.coco, args.dest)
    print(f'Imported {len(rows)} complete F971 TIFF frames')
