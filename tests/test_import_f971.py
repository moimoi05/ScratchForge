import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.import_f971 import import_dataset
from scripts.build_library import build
from scripts.verify_dataset import verify
from scratchforge.dataset import sha


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'source'
    images = []
    rows = []
    specs = [('NG', 'train.jpg', 'train'), ('NG', 'val.jpg', 'validation'),
             ('OK', 'clean.jpg', 'train'), ('OK', 'test.jpg', 'test')]
    for n, (status, name, split) in enumerate(specs, 1):
        folder = root / 'F971' / status / 'images'
        folder.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (400, 300), (n * 20, 80, 100)).save(folder / name)
        if status == 'NG':
            images.append(dict(id=n, file_name=name, width=400, height=300))
        rows.append(dict(source_image=name, product_code='F971', source_status=status,
                         source_image_id=n if status == 'NG' else '', split=split))
    annotations = [dict(id=1, image_id=1, category_id=1,
                        segmentation=[[20, 20, 30, 20, 30, 280, 20, 280]]),
                   dict(id=2, image_id=1, category_id=1,
                        segmentation=[[0, 0, 10, 0, 10, 10, 0, 10]]),
                   dict(id=3, image_id=2, category_id=1,
                        segmentation=[[40, 40, 50, 40, 50, 50, 40, 50]])]
    annotation_dir = root / 'F971/NG/annotations'
    annotation_dir.mkdir(parents=True)
    (annotation_dir / 'coco.json').write_text(json.dumps(dict(
        images=images, annotations=annotations, categories=[dict(id=1, name='scratch')]
    )))
    (root / 'reports').mkdir()
    with (root / 'reports/source_image_splits.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root


def test_full_frame_import_is_independent_and_idempotent(source, tmp_path):
    dest = tmp_path / 'dest'
    first = import_dataset(source, dest)
    assert first == import_dataset(source, dest)
    assert len(first) == 4
    assert {r['split'] for r in first} == {'train', 'val', 'test'}
    assert all((r['width'], r['height']) == (400, 300) for r in first)
    coco = json.loads((dest / 'data/fullframe/coco.json').read_text())
    assert len(coco['images']) == 4 and len(coco['annotations']) == 3
    assert not any(a['image_id'] in [i['id'] for i in coco['images'] if i['status'] == 'OK']
                   for a in coco['annotations'])
    assert verify(dest)['status'] == 'PASS'


def test_library_full_instance_native_pixels_and_exclusions(source, tmp_path):
    dest = tmp_path / 'dest'
    import_dataset(source, dest)
    stamps = build(dest)
    assert len(stamps) == 1
    stamp = stamps[0]
    assert stamp['split'] == 'train' and stamp['source_annotation_id'] == 1
    assert stamp['rgb_sha256'] == sha(dest / 'data/library' / stamp['rgb'])
    assert stamp['mask_sha256'] == sha(dest / 'data/library' / stamp['mask'])
    assert stamp['height'] >= 292
    x, y, x2, y2 = stamp['roi']
    with Image.open(source / 'F971/NG/images/train.jpg') as original:
        with Image.open(dest / 'data/library' / stamp['rgb']) as rgb:
            assert np.array_equal(np.asarray(original.crop((x, y, x2, y2))), np.asarray(rgb))
    with Image.open(dest / 'data/library' / stamp['thumbnail']) as thumbnail:
        assert thumbnail.mode == 'RGBA' and thumbnail.getextrema()[3][0] == 0
    report = json.loads((dest / 'reports/library.json').read_text())
    assert len(report['hold']) == 1 and len(report['evaluation_excluded']) == 1


@pytest.mark.parametrize('mutation', ['unsafe_name', 'bad_dimensions', 'invalid_polygon', 'missing_split', 'duplicate_split', 'unknown_category'])
def test_rejects_inconsistent_source(source, tmp_path, mutation):
    path = source / 'F971/NG/annotations/coco.json'
    coco = json.loads(path.read_text())
    if mutation == 'unsafe_name':
        coco['images'][0]['file_name'] = '../train.jpg'
    elif mutation == 'bad_dimensions':
        coco['images'][0]['width'] = 399
    elif mutation == 'invalid_polygon':
        coco['annotations'][0]['segmentation'] = [[-1, 0, 1, 0, 1, 2]]
    elif mutation == 'unknown_category':
        coco['annotations'][0]['category_id'] = 999
    else:
        csv_path = source / 'reports/source_image_splits.csv'
        lines = csv_path.read_text().splitlines()
        lines = lines[:-1] if mutation == 'missing_split' else lines + [lines[1]]
        csv_path.write_text('\n'.join(lines) + '\n')
    path.write_text(json.dumps(coco))
    with pytest.raises(ValueError):
        import_dataset(source, tmp_path / 'dest')


def test_checksum_conflict_and_verify_detect_tampering(source, tmp_path):
    dest = tmp_path / 'dest'
    rows = import_dataset(source, dest)
    path = dest / rows[0]['path']
    path.write_bytes(b'changed')
    with pytest.raises(ValueError):
        import_dataset(source, dest)
    with pytest.raises(ValueError):
        verify(dest)


def test_rejects_overlapping_source_and_destination(source):
    with pytest.raises(ValueError):
        import_dataset(source, source / 'nested')


def test_verify_rejects_split_reassignment_and_annotation_drift(source, tmp_path):
    dest = tmp_path / 'dest'
    import_dataset(source, dest)
    index_path = dest / 'data/fullframe/index.json'
    coco_path = dest / 'data/fullframe/coco.json'
    rows = json.loads(index_path.read_text())
    coco = json.loads(coco_path.read_text())
    rows[0]['split'] = 'test'
    next(image for image in coco['images'] if image['id'] == rows[0]['coco_image_id'])['split'] = 'test'
    index_path.write_text(json.dumps(rows))
    coco_path.write_text(json.dumps(coco))
    with pytest.raises(ValueError):
        verify(dest)
    import_dataset(source, dest)
    coco = json.loads(coco_path.read_text())
    coco['annotations'] = []
    coco_path.write_text(json.dumps(coco))
    with pytest.raises(ValueError):
        verify(dest)


def test_boundary_hold_report_includes_evaluation_instances(source, tmp_path):
    path = source / 'F971/NG/annotations/coco.json'
    coco = json.loads(path.read_text())
    coco['annotations'][2]['segmentation'] = [[0, 40, 50, 40, 50, 50, 0, 50]]
    path.write_text(json.dumps(coco))
    dest = tmp_path / 'dest'
    import_dataset(source, dest)
    build(dest)
    report = json.loads((dest / 'reports/library.json').read_text())
    assert {row['annotation'] for row in report['hold']} == {2, 3}


def test_authoritative_metadata_is_checksum_protected(source, tmp_path):
    dest = tmp_path / 'dest'
    rows = import_dataset(source, dest)
    provenance = json.loads((dest / 'data/fullframe/provenance.json').read_text())
    assert len(provenance) == len(rows) + 2
    path = dest / 'data/reference/source_image_splits.csv'
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError):
        verify(dest)
