import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from backend.segment import active_library, historical_library, historical_stamp_asset, segment_router
from backend.store import Store, write_json
from scratchforge.dataset import sha


TRAIN = 'f971-0123456789abcdef'
EVAL = 'f971-eval'
OK = 'f971-ok'
S942 = 's942-test'
POINTS = [8, 9, 25, 10, 23, 20]
FIXED = [9, 9, 25, 10, 23, 20]
BOUNDARY = [0, 9, 25, 10, 23, 20]


def ann(identifier, points=POINTS):
    return {'id': identifier, 'category_id': 1, 'segmentation': [points]}


def client_for(store):
    app = FastAPI()
    app.include_router(segment_router(store))
    return TestClient(app)


@pytest.fixture
def studio(tmp_path):
    store = Store(tmp_path)
    rows = []
    for identifier, group, split, status, annotations in [
        (TRAIN, 'f971-scratch', 'train', 'NG', [ann(10), ann(53)]),
        (EVAL, 'f971-scratch', 'val', 'NG', [ann(20)]),
        (OK, 'f971-ok', 'train', 'OK', []),
        (S942, 's942-ok', None, 'OK', []),
    ]:
        path = tmp_path / (identifier + '.tif')
        Image.new('RGB', (64, 48), 'blue').save(path)
        rows.append({'id': identifier, 'group': group, 'split': split, 'status': status,
                     'path': path.name, 'file_name': path.name, 'source_image': identifier + '.jpg',
                     'source_image_id': len(rows) + 1, 'sha256': sha(path),
                     'width': 64, 'height': 48, 'annotations': annotations})
    write_json(tmp_path / 'data/segment/index.json', rows)
    write_json(tmp_path / 'data/fullframe/index.json', rows[:3])
    return store, client_for(store), rows


def resource(identifier=TRAIN):
    return f'/api/segment/images/{identifier}'


def save(client, identifier=TRAIN, revision=0, annotations=None):
    return client.put(resource(identifier) + '/annotations', json={
        'revision': revision, 'annotations': [ann(10, FIXED), ann(53)] if annotations is None else annotations})


def approve(client, identifier=TRAIN, revision=0):
    return client.post(resource(identifier) + '/approve', json={'revision': revision})


def test_list_detail_and_preview(studio):
    store, client, rows = studio
    public = client.get('/api/segment/images').json()
    assert [row['id'] for row in public] == [TRAIN, EVAL, OK, S942]
    assert not {'path', 'sha256', 'annotations'} & public[0].keys()
    assert public[0]['annotation_count'] == 2
    detail = client.get(resource()).json()
    assert detail['annotations'] == [ann(10), ann(53)]
    assert detail['revision'] == 0
    assert detail['review_status'] == 'pending'
    for url in (detail['preview'], detail['thumbnail']):
        preview = client.get(url)
        assert preview.status_code == 200
        assert preview.headers['content-type'] == 'image/png'
    assert client.get(resource('missing')).status_code == 404
    assert client.get(resource() + '/unexpected').status_code == 404
    (store.root / rows[0]['path']).write_bytes(b'invalid TIFF')
    assert client.get(detail['preview']).status_code == 422


def test_draft_approval_history_and_withdrawal(studio):
    store, client, _ = studio
    base = [{'id': 'original-10', 'source_frame_id': TRAIN}]
    assert active_library(store, base) == base
    assert save(client).status_code == 200
    assert save(client).status_code == 409
    assert active_library(store, base) == base
    response = approve(client, revision=1)
    assert response.status_code == 200, response.text
    approved = response.json()
    assert approved['review_status'] == 'approved'
    assert approved['annotations'] == [ann(10, FIXED), ann(53)]
    stamps = active_library(store, base)
    assert len(stamps) == 1 and stamps[0]['source_annotation_id'] == 10
    stamp = stamps[0]
    assert stamp['id'] != 'original-10'
    assert stamp['review_status'] == 'APPROVED'
    assert historical_stamp_asset(store, stamp['rgb'])
    assert not historical_stamp_asset(store, '../index.json')
    assert (store.root / 'data/library' / stamp['rgb']).is_file()
    reopened = client_for(Store(store.root))
    assert reopened.get(resource()).json() == approved
    assert save(reopened, revision=approved['revision']).json() == approved
    assert approve(reopened, revision=approved['revision']).json() == approved
    edited = save(reopened, revision=approved['revision'], annotations=[ann(10), ann(53)]).json()
    assert edited['review_status'] == 'pending'
    assert active_library(store, base) == []
    assert historical_library(store) == stamps
    again = approve(reopened, revision=edited['revision']).json()
    assert again['review_status'] == 'approved'
    assert active_library(store, base)[0]['id'] != stamp['id']
    assert len(historical_library(store)) == 2


def test_validation_and_nontrain_approval(studio):
    store, client, _ = studio
    for annotations in ([ann(0)], [ann(999)], [ann(10), ann(10)], [ann(10, [1, 2, 3])],
                        [ann(10, [0, 0, 65, 10, 5, 12])],
                        [{'id': 10, 'category_id': 9, 'segmentation': [POINTS]}]):
        assert save(client, annotations=annotations).status_code == 422
    pending = save(client, annotations=[ann(None, BOUNDARY), ann(None, POINTS + POINTS[:2])]).json()
    assert pending['annotations'][0]['id'] != pending['annotations'][1]['id']
    assert pending['annotations'][1]['segmentation'] == [POINTS]
    assert approve(client, revision=pending['revision']).status_code == 200
    assert len(active_library(store, [])) == 1
    assert approve(client, EVAL).status_code == 200
    assert approve(client, OK).status_code == 200
    assert approve(client, S942).status_code == 200
    assert len(active_library(store, [])) == 1


def test_review_verifies_source_integrity(studio):
    store, client, rows = studio
    write_json(store.root / 'data/fullframe/index.json', rows[1:])
    assert approve(client).status_code == 422
    assert active_library(store, []) == []
    (store.root / rows[0]['path']).unlink()
    assert client.get(client.get(resource()).json()['preview']).status_code == 422


def coco_file(points=POINTS):
    return {
        'images': [
            {'id': 5, 'file_name': TRAIN + '_jpg.rf.abcdef.jpg', 'width': 64, 'height': 48,
             'extra': {'name': TRAIN + '.jpg'}},
            {'id': 6, 'file_name': EVAL + '.jpg', 'width': 64, 'height': 48},
            {'id': 7, 'file_name': '2026-09-09-14-21-22_jpg.rf.abcdef.jpg',
             'width': 64, 'height': 48},
        ],
        'categories': [{'id': 1, 'name': 'scratch'}],
        'annotations': [{**ann(10, points), 'image_id': 5}, {**ann(53), 'image_id': 5},
                        {**ann(20), 'image_id': 6}],
    }


def reimport(client, coco):
    return client.post('/api/segment/import', files={'file': ('coco.json', json.dumps(coco))})


def test_reimport_is_atomic_and_preserves_local_edits(studio):
    _, client, _ = studio
    assert reimport(client, coco_file()).json() == {'images': 2, 'annotations': 3}
    assert save(client).status_code == 200
    assert reimport(client, coco_file()).status_code == 200
    before = client.get(resource()).json()
    assert before['annotations'][0] == ann(10, FIXED)
    assert reimport(client, coco_file(FIXED)).status_code == 409
    assert client.get(resource()).json() == before
    bad = coco_file()
    bad['images'][0]['width'] -= 1
    assert reimport(client, bad).status_code == 422
    missing = coco_file()
    missing['images'].pop(1)
    missing['annotations'].pop(2)
    assert reimport(client, missing).status_code == 422


def test_reimport_updates_untouched_frames(studio):
    _, client, _ = studio
    changed = coco_file(FIXED)
    assert reimport(client, changed).status_code == 200
    detail = client.get(resource()).json()
    assert detail['revision'] == 1
    assert detail['annotations'][0] == ann(10, FIXED)
    assert save(client).status_code == 409
    assert client.post('/api/segment/import', files={'file': ('bad.json', b'not JSON')}).status_code == 422


def test_validates_absolute_catalog_sources_and_tiff_content(tmp_path, studio):
    store, client, rows = studio
    rows[0] = {**rows[0], 'path': str(tmp_path / rows[0]['path'])}
    rows[-1] = {**rows[-1], 'path': str(tmp_path / rows[-1]['path'])}
    write_json(store.root / 'data/segment/index.json', rows)
    assert client.get(resource() + '/preview').status_code == 422
    assert client.get(resource(S942) + '/preview').status_code == 200
    source = tmp_path / rows[-1]['path']
    first = Image.new('RGB', (64, 48), 'red')
    first.save(source, save_all=True, append_images=[first])
    write_json(store.root / 'data/segment/index.json', [*rows[:-1], {**rows[-1], 'sha256': sha(source)}])
    assert client.get(resource(S942) + '/preview').status_code == 422


def test_failed_publication_leaves_draft_and_base_stamps(studio, monkeypatch):
    from scripts import build_library

    store, client, _ = studio
    base = [{'id': 'original-10', 'source_frame_id': TRAIN}]
    response = save(client)
    assert response.status_code == 200

    def unavailable(*args, **kwargs):
        raise OSError('temporary disk error')

    monkeypatch.setattr(build_library, 'make_stamp', unavailable)
    assert approve(client, revision=1).status_code == 422
    assert active_library(store, base) == base
    assert historical_library(store) == []
    assert client.get(resource()).json() == response.json()


def test_caps_edit_and_coalesced_import_payloads(studio, monkeypatch):
    from backend import segment

    _, client, _ = studio
    monkeypatch.setattr(segment, 'MAX_COORDINATES', 8)
    assert save(client).status_code == 422
    assert reimport(client, coco_file()).status_code == 422
    monkeypatch.setattr(segment, 'MAX_IMPORT_BYTES', 16)
    assert reimport(client, coco_file()).status_code == 413
