"""Verify full-frame F971 data without reading legacy cropped patches."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from backend.images import validate_tiff
from scratchforge.dataset import ROOT, save_json, sha, valid_polygons
from scripts.import_f971 import load_splits, validate_coco
from scripts.import_tiff_dataset import GROUPS, TIFF_SUFFIXES


def verify_segment(root, records, coco):
    catalog = json.loads((root / 'data/segment/index.json').read_text(encoding='utf-8'))
    by_id = {row['id']: row for row in catalog}
    if len(by_id) != len(catalog):
        raise ValueError('Duplicate Segment frame ID')
    annotations = {}
    for item in coco['annotations']:
        annotations.setdefault(item['image_id'], {})[item['id']] = (item['category_id'], item['segmentation'])
    for row in records:
        item = by_id.pop(row['id'], None)
        if not item or item['group'] != GROUPS[row['status']] or any(item.get(key) != value for key, value in row.items()):
            raise ValueError('Segment F971 frame differs from the verified manifest')
        actual = {a['id']: (a['category_id'], a['segmentation']) for a in item['annotations']}
        expected = annotations.get(row['coco_image_id'], {})
        if actual != expected or len(actual) != len(item['annotations']):
            raise ValueError('Segment seed geometry differs from verified COCO')
    for row in by_id.values():
        source = Path(row['path'])
        if (row['group'] != GROUPS['S942'] or row['status'] != 'OK' or row['split'] is not None
                or row['annotations'] or not source.is_absolute() or source.suffix.lower() not in TIFF_SUFFIXES
                or source.name != row['file_name'] or not source.is_file() or sha(source) != row['sha256']):
            raise ValueError('Invalid S942 Segment source')
    return dict(Counter(row['group'] for row in catalog))


def verify(root=ROOT):
    root = Path(root).resolve()
    records = json.loads((root / 'data/fullframe/index.json').read_text())
    provenance = json.loads((root / 'data/fullframe/provenance.json').read_text())
    for source in provenance:
        path = (root / source['new_path']).resolve()
        if not path.is_relative_to(root) or sha(path) != source['sha256']:
            raise ValueError('Imported source provenance checksum differs')
    coco = json.loads((root / 'data/fullframe/coco.json').read_text())
    original = json.loads((root / 'data/raw/F971/NG/annotations/coco.json').read_text())
    validate_coco(original)
    splits = load_splits(root / 'data/reference/source_image_splits.csv')
    if coco['annotations'] != original['annotations'] or coco['categories'] != original['categories']:
        raise ValueError('Original full-frame annotation data changed')
    images = {image['id']: image for image in coco['images']}
    if len(images) != len(records) or len(images) != len(coco['images']):
        raise ValueError('Full-frame manifest and COCO counts differ')
    keys = set()
    for row in records:
        path = (root / row['path']).resolve()
        if not path.is_relative_to(root) or sha(path) != row['sha256']:
            raise ValueError('Full-frame path or checksum verification failed')
        key = (row['status'], row['source_image'])
        image = images.get(row['coco_image_id'])
        if key in keys or row['split'] not in {'train', 'val', 'test'}:
            raise ValueError('Duplicate source or invalid split')
        if key not in splits or splits[key]['split'] != row['split']:
            raise ValueError('Source split differs from authoritative split CSV')
        keys.add(key)
        if not image or image['file_name'] != row['path'] or image['split'] != row['split']:
            raise ValueError('COCO image metadata differs from manifest')
        if image['status'] != row['status'] or (image['width'], image['height']) != (row['width'], row['height']):
            raise ValueError('COCO dimensions or status differ from manifest')
        with Image.open(path) as rgb:
            if rgb.mode != 'RGB' or rgb.size != (row['width'], row['height']):
                raise ValueError('RGB dimensions verification failed')
            if path.suffix.lower() in TIFF_SUFFIXES:
                validate_tiff(rgb)
    if keys != set(splits):
        raise ValueError('Split inventory differs from imported inventory')
    for annotation in coco['annotations']:
        image = images.get(annotation['image_id'])
        if not image or image['status'] != 'NG' or not valid_polygons(
                annotation['segmentation'], image['width'], image['height']):
            raise ValueError('Full-frame annotation verification failed')
    counts = Counter(f"{row['status']}/{row['split']}" for row in records)
    result = dict(status='PASS', images=len(records), ng=sum(r['status'] == 'NG' for r in records),
                  ok=sum(r['status'] == 'OK' for r in records), polygons=len(coco['annotations']),
                  split_counts=dict(counts), checks=['full-frame SHA256', 'RGB dimensions',
                  'source split integrity', 'full-frame COCO polygon bounds'])
    if any(Path(row['path']).suffix.lower() in TIFF_SUFFIXES for row in records):
        result['format'] = 'TIFF'
        result['segment_counts'] = verify_segment(root, records, coco)
    save_json(root / 'reports/verification.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.dest), indent=2))
