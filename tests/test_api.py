import io
import json
import time
import zipfile

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import create_app
from scratchforge.dataset import sha


@pytest.fixture
def studio(tmp_path):
    (tmp_path / 'data/fullframe').mkdir(parents=True)
    (tmp_path / 'data/library').mkdir(parents=True)
    Image.new('RGB', (120, 100), (20, 30, 40)).save(tmp_path / 'background.png')
    index = [{'id': 'clean-1', 'status': 'OK', 'split': 'train', 'path': 'background.png', 'width': 120, 'height': 100, 'source_image': 'background.png', 'sha256': sha(tmp_path/'background.png')}]
    (tmp_path / 'data/fullframe/index.json').write_text(json.dumps(index))
    (tmp_path / 'data/library/index.json').write_text('[]')
    app = create_app(tmp_path, infer=lambda rgb, mask, seed, strength: Image.new('RGB', rgb.size, 'red'))
    with TestClient(app) as client:
        yield client


def finish(client, body):
    response = client.post('/api/jobs', json=body)
    assert response.status_code == 200, response.text
    for _ in range(100):
        job = client.get('/api/jobs/' + response.json()['id']).json()
        if job['status'] in ('succeeded', 'failed'):
            return job
        time.sleep(.01)
    raise AssertionError('job did not finish')


def test_review_gate_and_export_roundtrip(studio):
    image = studio.post('/api/backgrounds/clean-1/open').json()
    job = finish(studio, {'image_id': image['id'], 'engine': 'sdxl', 'brushes': [{'points': [20, 20, 60, 40], 'width': 4}]})
    assert job['status'] == 'succeeded', job
    result = job['result']
    assert result['polygons'] == []
    rid = result['id']
    assert studio.post(f'/api/results/{rid}/export').status_code == 409
    assert studio.post(f'/api/results/{rid}/approve', json={'confirmed': True}).status_code == 422
    polygons = [[19, 18, 65, 18, 65, 45, 19, 45]]
    assert studio.put(f'/api/results/{rid}/annotations', json={'polygons': polygons}).status_code == 200
    assert studio.post(f'/api/results/{rid}/approve', json={'confirmed': True}).status_code == 200
    exported = studio.post(f'/api/results/{rid}/export')
    assert exported.status_code == 200, exported.text
    archive = zipfile.ZipFile(io.BytesIO(studio.get(exported.json()['url']).content))
    coco = json.loads(archive.read('annotations.json'))
    assert coco['annotations'][0]['segmentation'] == polygons
    assert coco['images'][0]['width'] == 120
    assert 'provenance.json' in archive.namelist()
    download = exported.json()['url']
    studio.put(f'/api/results/{rid}/annotations', json={'polygons': polygons})
    assert studio.post(f'/api/results/{rid}/export').status_code == 409
    assert studio.get(download).status_code == 409


def test_upload_is_exploratory_and_invalid_requests(studio):
    assert studio.get('/api/health').status_code == 200
    assert studio.get('/api/backgrounds').json()[0]['id'] == 'clean-1'
    assert studio.get('/api/dataset').json()['ready'] is False
    assert studio.post('/api/upload', files={'file': ('bad.png', b'bad')}).status_code == 400
    stream = io.BytesIO(); Image.new('RGB', (100, 90)).save(stream, 'PNG')
    image = studio.post('/api/upload', files={'file': ('ok.png', stream.getvalue())}).json()
    job = finish(studio, {'image_id': image['id'], 'engine': 'baseline', 'brushes': [{'points': [10, 10, 30, 30], 'width': 3}]})
    rid = job['result']['id']
    studio.put(f'/api/results/{rid}/annotations', json={'polygons': [[10, 10, 30, 10, 30, 30]]})
    studio.post(f'/api/results/{rid}/approve', json={'confirmed': True})
    assert studio.post(f'/api/results/{rid}/export').status_code == 422
    assert studio.post('/api/jobs', json={'image_id': '../bad'}).status_code == 422
    assert studio.post('/api/jobs', json={'image_id': 'a' * 32}).status_code == 422
    assert studio.get('/api/results/' + 'a' * 32).status_code == 404


def test_boundary_assets_failed_job_and_validation(studio):
    assert studio.get('/api/backgrounds/clean-1/thumbnail').status_code == 200
    assert studio.get('/api/backgrounds/clean-1/image').status_code == 200
    assert studio.post('/api/backgrounds/missing/open').status_code == 404
    assert studio.get('/api/backgrounds/missing/image').status_code == 404
    assert studio.get('/api/backgrounds/missing/thumbnail').status_code == 404
    assert studio.get('/files/not-an-id.png').status_code == 404
    assert studio.get('/files/'+'a'*32+'.png').status_code == 404
    assert studio.get('/library-files/missing.png').status_code == 404
    assert studio.post('/api/jobs', headers={'Origin': 'https://evil.test'}, json={}).status_code == 403
    assert studio.post('/api/jobs', headers={'Content-Length': str(50*1024*1024)}, content=b'').status_code == 413
    image = studio.post('/api/backgrounds/clean-1/open').json()
    assert studio.post('/api/jobs', json={'image_id': image['id'], 'placements': [{'stamp_id': 'bad', 'x': 1, 'y': 1}]}).status_code == 422
    failed = finish(studio, {'image_id': image['id'], 'brushes': [{'points': [-100, -100, -90, -90]}]})
    assert failed['status'] == 'failed' and 'scratch' in failed['error']
    assert studio.post('/api/jobs', json={'image_id': image['id'], 'brushes': [{'points': [1, 2, 3, 4, 5]}]}).status_code == 422


def test_export_failure_does_not_destroy_existing_zip(studio, monkeypatch):
    image = studio.post('/api/backgrounds/clean-1/open').json()
    result = finish(studio, {'image_id': image['id'], 'brushes': [{'points': [10, 10, 30, 30], 'width': 3}]})['result']
    rid = result['id']
    studio.put(f'/api/results/{rid}/annotations', json={'polygons': [[10, 10, 30, 10, 30, 30]]})
    studio.post(f'/api/results/{rid}/approve', json={'confirmed': True})
    store = studio.app.state.store
    destination = store.out/f'{rid}.zip'
    destination.write_bytes(b'existing archive')
    def fail(*args, **kwargs):
        raise OSError('simulated full disk')
    monkeypatch.setattr(zipfile.ZipFile, 'write', fail)
    with pytest.raises(OSError):
        studio.app.state.jobs.export(rid)
    assert destination.read_bytes() == b'existing archive'


def test_changed_imported_background_cannot_claim_train_provenance(studio):
    root = studio.app.state.store.root
    Image.new('RGB', (120, 100), 'red').save(root/'background.png')
    response = studio.post('/api/backgrounds/clean-1/open')
    assert response.status_code == 422
    assert 'changed' in response.json()['detail']


def test_stamp_source_must_match_fullframe_train_manifest(studio):
    root = studio.app.state.store.root
    stamp = {'id': 'forged', 'split': 'train', 'source_frame_id': 'missing', 'source_sha256': 'bad',
             'source_image': 'missing.jpg', 'source_image_id': 1}
    (root/'data/library/index.json').write_text(json.dumps([stamp]))
    image = studio.post('/api/backgrounds/clean-1/open').json()
    response = studio.post('/api/jobs', json={'image_id': image['id'], 'placements': [{'stamp_id': 'forged', 'x': 40, 'y': 40}]})
    assert response.status_code == 422
    assert 'source' in response.json()['detail']


def test_reassigned_background_split_blocks_training_export(studio):
    image = studio.post('/api/backgrounds/clean-1/open').json()
    result = finish(studio, {'image_id': image['id'], 'brushes': [{'points': [10, 10, 30, 30], 'width': 3}]})['result']
    rid = result['id']
    studio.put(f'/api/results/{rid}/annotations', json={'polygons': [[10, 10, 30, 10, 30, 30]]})
    studio.post(f'/api/results/{rid}/approve', json={'confirmed': True})
    path = studio.app.state.store.root/'data/fullframe/index.json'
    rows = json.loads(path.read_text()); rows[0]['split'] = 'test'
    path.write_text(json.dumps(rows))
    assert studio.post(f'/api/results/{rid}/export').status_code == 422


def test_verified_stamp_source_and_assets_generate_successfully(studio):
    root = studio.app.state.store.root
    library = root/'data/library'
    Image.new('RGB', (20, 20), 'white').save(library/'stamp.png')
    Image.new('L', (20, 20), 255).save(library/'mask.png')
    source = {'id': 'ng-1', 'status': 'NG', 'split': 'train', 'sha256': sha(library/'stamp.png'),
              'source_image': 'ng.png', 'source_image_id': 7}
    manifest = root/'data/fullframe/index.json'
    manifest.write_text(json.dumps([*json.loads(manifest.read_text()), source]))
    stamp = {'id': 'stamp-1', 'split': 'train', 'source_frame_id': 'ng-1', 'source_image_id': 7,
             'source_image': 'ng.png', 'source_sha256': source['sha256'], 'rgb': 'stamp.png', 'mask': 'mask.png',
             'rgb_sha256': sha(library/'stamp.png'), 'mask_sha256': sha(library/'mask.png')}
    (library/'index.json').write_text(json.dumps([stamp]))
    assert studio.get('/library-files/stamp.png').status_code == 200
    image = studio.post('/api/backgrounds/clean-1/open').json()
    job = finish(studio, {'image_id': image['id'], 'placements': [{'stamp_id': 'stamp-1', 'x': 40, 'y': 40}]})
    assert job['status'] == 'succeeded', job
    assert job['result']['provenance']['stamps'][0]['source_frame_id'] == 'ng-1'
