"""Verify full-frame F971 data without reading legacy cropped patches."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from scratchforge.dataset import ROOT, save_json, sha, valid_polygons
from scripts.import_f971 import load_splits, validate_coco


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
    save_json(root / 'reports/verification.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.dest), indent=2))
