import json

import pytest
from PIL import Image

from scratchforge.dataset import sha
from scripts.build_library import build, make_stamp
from scripts.import_f971 import import_dataset
from scripts.import_tiff_dataset import match_coco, migrate
from scripts.verify_dataset import verify
from test_import_f971 import source


@pytest.fixture
def tiff_sources(source, tmp_path):
    image_root = tmp_path / 'images'
    folders = {
        'NG': image_root / 'F971_Scratch',
        'OK': image_root / '2026-09-14/ORG/MATTRUOC/CAM2/DOME-1',
        'S942': image_root / 'S942_OK',
    }
    for folder in folders.values():
        folder.mkdir(parents=True)
    for status in ('NG', 'OK'):
        for path in (source / 'F971' / status / 'images').glob('*.jpg'):
            Image.new('RGB', (400, 300), (15, 70, 80)).save(folders[status] / (path.stem + '.tif'))
    extra = '2026-09-09-14-21-22'
    Image.new('RGB', (400, 300)).save(folders['NG'] / (extra + '.tif'))
    Image.new('RGB', (400, 300)).save(folders['S942'] / 'fixture.TIFF')
    coco = json.loads((source / 'F971/NG/annotations/coco.json').read_text())
    coco['images'] = [
        {**row, 'file_name': row['file_name'].replace('.jpg', '_jpg.rf.' + 'a' * 32 + '.jpg'),
         **({'extra': {'name': row['file_name']}} if row['id'] == 1 else {})}
        for row in coco['images']
    ] + [{'id': 99, 'file_name': extra + '_jpg.rf.' + 'a' * 32 + '.jpg', 'width': 400, 'height': 300}]
    coco_path = tmp_path / 'roboflow.json'
    coco_path.write_text(json.dumps(coco))
    return source, image_root, coco_path, folders


def test_migration_preserves_ids_splits_geometry_and_legacy_assets(tiff_sources, tmp_path):
    source, image_root, coco, _ = tiff_sources
    dest = tmp_path / 'dest'
    legacy = import_dataset(source, dest)
    legacy_stamps = build(dest)
    active = migrate(source, image_root, coco, dest)
    assert active == migrate(source, image_root, coco, dest)
    assert json.loads((dest / 'data/fullframe/legacy-index.json').read_text()) == legacy
    assert {(row['id'], row['source_image'], row['coco_image_id'], row['split']) for row in active} == {
        (row['id'], row['source_image'], row['coco_image_id'], row['split']) for row in legacy
    }
    assert all(row['path'].endswith('.tif') and sha(dest / row['path']) == row['sha256'] for row in active)
    assert all(sha(dest / row['path']) == row['sha256'] for row in legacy)
    segment = json.loads((dest / 'data/segment/index.json').read_text())
    assert {row['group'] for row in segment} == {'f971-scratch', 'f971-ok', 's942-ok'}
    assert len(segment) == 5 and sum(len(row['annotations']) for row in segment) == 3
    assert next(row for row in segment if row['file_name'] == 'train.tif')['annotations'][0]['segmentation'] == [
        [20, 20, 30, 20, 30, 280, 20, 280]
    ]
    s942 = next(row for row in segment if row['group'] == 's942-ok')
    assert s942['split'] is None and s942['annotations'] == [] and s942['path'].endswith('fixture.TIFF')
    assert verify(dest)['status'] == 'PASS'
    stamps = build(dest)
    assert len(stamps) == 1 and stamps[0]['id'] != legacy_stamps[0]['id']
    assert sha(dest / 'data/library' / legacy_stamps[0]['rgb']) == legacy_stamps[0]['rgb_sha256']
    assert json.loads((dest / 'data/library/legacy-index.json').read_text())[0]['id'] == legacy_stamps[0]['id']


def test_build_upgrades_published_png_stamps_and_preserves_history(tiff_sources, tmp_path):
    from backend.segment import read_state, state_path
    from backend.store import Store, write_json
    source, image_root, coco, _ = tiff_sources
    dest = tmp_path/'dest'
    import_dataset(source, dest)
    records = migrate(source, image_root, coco, dest)
    store = Store(dest)
    frame = next(r for r in records if r['status'] == 'NG' and r['split'] == 'train')
    annotation = {'id': 1, 'segmentation': [[20, 20, 30, 20, 30, 280, 20, 280]]}
    old = {'id': 'old-review', 'rgb': 'old-review.png', 'annotation_revision': 'a'*32,
           'source_annotation_id': 1, 'source_frame_id': frame['id'], 'split': 'train',
           'review_status': 'APPROVED', 'segment_revision': 2}
    state = read_state(store)
    state['reviews'][frame['id']] = {'sha256': frame['sha256'], 'published_before': True,
                                    'annotations': [annotation], 'stamps': [old], 'history': [old]}
    write_json(state_path(store), state)
    build(dest)
    review = read_state(store)['reviews'][frame['id']]
    active, = store.library()
    assert active['review_status'] == 'APPROVED' and active['rgb'].endswith('/stamp_crop.tif')
    assert active['id'] != old['id'] and old in review['history']
    build(dest)
    assert read_state(store)['reviews'][frame['id']] == review


@pytest.mark.parametrize('mutation,error', [
    ('missing', 'Missing TIFF'), ('duplicate', 'Duplicate TIFF'), ('dimensions', 'dimensions'),
    ('invalid', 'Invalid.*TIFF'), ('multi', 'single-page'), ('oversize', 'megapixels'),
    ('roboflow_dimensions', 'dimensions'), ('roboflow_missing', 'Missing TIFF'),
    ('roboflow_duplicate', 'Duplicate'), ('roboflow_invalid', 'Invalid polygon'),
    ('roboflow_geometry', 'differs from'), ('extra_annotated', 'excluded'),
])
def test_migration_rejects_invalid_sources_before_publish(tiff_sources, tmp_path, monkeypatch, mutation, error):
    source, image_root, coco_path, folders = tiff_sources
    dest = tmp_path / 'dest'
    target = folders['NG'] / 'train.tif'
    if mutation == 'missing':
        target.unlink()
    elif mutation == 'duplicate':
        target.with_suffix('.tiff').write_bytes(target.read_bytes())
    elif mutation == 'dimensions':
        Image.new('RGB', (399, 300)).save(target)
    elif mutation == 'invalid':
        target.write_bytes(b'bad TIFF')
    elif mutation == 'multi':
        image = Image.new('RGB', (400, 300))
        image.save(target, save_all=True, append_images=[image])
    elif mutation == 'oversize':
        monkeypatch.setattr('backend.images.MAX_TIFF_PIXELS', 100)
    else:
        coco = json.loads(coco_path.read_text())
        if mutation == 'roboflow_dimensions':
            coco['images'][0]['width'] = 399
        elif mutation == 'roboflow_missing':
            coco['images'][0]['extra']['name'] = 'missing.jpg'
            coco['images'][0]['file_name'] = 'missing.jpg'
        elif mutation == 'roboflow_duplicate':
            coco['images'][1]['extra'] = {'name': 'train.jpg'}
            coco['images'][1]['file_name'] = 'train.jpg'
        elif mutation == 'roboflow_invalid':
            coco['annotations'][0]['segmentation'][0][0] = -1
        elif mutation == 'roboflow_geometry':
            coco['annotations'][0]['segmentation'][0][0] = 21
        elif mutation == 'extra_annotated':
            coco['annotations'].append({**coco['annotations'][0], 'id': 100, 'image_id': 99})
        coco_path.write_text(json.dumps(coco))
    with pytest.raises(ValueError, match=error):
        migrate(source, image_root, coco_path, dest)
    assert not (dest / 'data/fullframe/index.json').exists()


def test_review_stamp_revisions_preserve_assets_and_hold(tiff_sources, tmp_path):
    source, image_root, coco, _ = tiff_sources
    dest = tmp_path / 'dest'
    active = migrate(source, image_root, coco, dest)
    row = next(row for row in active if row['source_image'] == 'train.jpg')
    output = dest / 'data/library'
    output.mkdir()
    annotation = {'id': 1, 'segmentation': [[30, 30, 60, 30, 60, 60, 30, 60]]}
    first = make_stamp(dest, row, annotation, output, revision='a' * 32)
    second = make_stamp(dest, row, annotation, output, revision='b' * 32)
    assert first['id'] != second['id'] and first['annotation_revision'] == 'a' * 32
    assert sha(output / first['mask']) == first['mask_sha256']
    with pytest.raises(ValueError, match='HOLD'):
        make_stamp(dest, row, {**annotation, 'id': 53}, output, revision='c' * 32)


def test_segment_verification_detects_drift(tiff_sources, tmp_path):
    source, image_root, coco, _ = tiff_sources
    dest = tmp_path / 'dest'
    migrate(source, image_root, coco, dest)
    index = dest / 'data/segment/index.json'
    original = index.read_text()
    rows = json.loads(original)
    rows[0]['annotations'][0]['segmentation'][0][0] = 21
    index.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match='geometry'):
        verify(dest)
    index.write_text(original)
    (image_root / 'S942_OK/fixture.TIFF').unlink()
    with pytest.raises(ValueError, match='Invalid S942'):
        verify(dest)


@pytest.mark.parametrize('section', ['images', 'categories', 'annotations'])
@pytest.mark.parametrize('identifier', [True, -1, 2**53, '1'])
def test_coco_rejects_ids_that_cannot_round_trip_to_browser(section, identifier):
    coco = {'images': [], 'categories': [], 'annotations': []}
    coco[section] = [{'id': identifier}]
    with pytest.raises(ValueError, match=f'Invalid COCO {section} ID'):
        match_coco(coco, [])


@pytest.mark.parametrize('field', ['image_id', 'category_id'])
@pytest.mark.parametrize('identifier', [True, -1, 2**53, '1'])
def test_coco_rejects_references_that_cannot_round_trip_to_browser(field, identifier):
    coco = {'images': [{'id': 1}], 'categories': [{'id': 1, 'name': 'scratch'}],
            'annotations': [{'id': 1, **{'image_id': 1, 'category_id': 1, field: identifier}}]}
    with pytest.raises(ValueError, match=f'Invalid COCO {field}'):
        match_coco(coco, [])


@pytest.mark.parametrize('identifier, error', [
    (2**53, 'Invalid COCO images ID'), (2**53 - 1, 'safe COCO image IDs'),
])
def test_migration_checks_safe_ids_in_legacy_authority(tiff_sources, tmp_path, identifier, error):
    source, image_root, coco, _ = tiff_sources
    authority = source / 'F971/NG/annotations/coco.json'
    original = json.loads(authority.read_text())
    original['images'][0]['id'] = identifier
    for item in original['annotations']:
        if item['image_id'] == 1:
            item['image_id'] = identifier
    authority.write_text(json.dumps(original))
    splits = source / 'reports/source_image_splits.csv'
    splits.write_text(splits.read_text().replace('train.jpg,F971,NG,1,train', f'train.jpg,F971,NG,{identifier},train'))
    with pytest.raises(ValueError, match=error):
        migrate(source, image_root, coco, tmp_path / 'dest')
