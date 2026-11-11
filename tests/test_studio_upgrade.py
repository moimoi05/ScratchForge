"""Integration contracts for native artifacts, operation jobs and restored scenes."""
import io
import json
import threading
import time
import zipfile

import numpy as np
import pytest
from PIL import Image

from backend.jobs import Jobs
from backend.models import Request
from backend.store import Store, write_json
from test_api import decode_coco, finish, studio


def await_job(jobs, identifier):
    for _ in range(200):
        record = jobs.store.load('job', identifier)
        if record['status'] in {'succeeded', 'failed', 'cancelled'}:
            return record
        time.sleep(.01)
    raise AssertionError('Job did not terminate')


def test_stamp_brush_scene_roundtrips_explicit_sources():
    body = {'image_id': 'a'*32, 'brushes': [{'points': [1, 2, 30, 40], 'stamp_id': 'sample'}],
            'brush_method': 'stamp_brush', 'brush_stamp_mode': 'selected', 'brush_stamp_id': 'sample'}
    assert Request(**body).model_dump()['brushes'][0]['stamp_id'] == 'sample'
    with pytest.raises(ValueError):
        Request(**{**body, 'lora_id': '../weights'})


def test_generated_labels_are_ready_for_confirmation_and_export(studio):
    opened = studio.post('/api/backgrounds/clean-1/open').json()
    result = finish(studio, {'image_id': opened['id'], 'brush_method': 'sd15',
                            'brushes': [{'points': [20, 20, 70, 55], 'width': 7}]})['result']
    assert result['polygons'] and result['review_status'] == 'pending'
    assert studio.post(f"/api/results/{result['id']}/export").status_code == 409
    assert studio.get(result['label_mask']).status_code == 200
    coco = studio.get(result['polygon_file']).json()
    assert coco['images'][0]['width'] == 120
    assert studio.post(f"/api/results/{result['id']}/approve", json={'confirmed': True}).status_code == 200
    exported = studio.post(f"/api/results/{result['id']}/export").json()
    with zipfile.ZipFile(io.BytesIO(studio.get(exported['url']).content)) as archive:
        assert {'image.tif', 'mask_full.png', 'polygon.json', 'mask.png', 'annotations.json',
                'scene.json', 'provenance.json'} <= set(archive.namelist())
        assert archive.read('polygon.json') == archive.read('annotations.json')
        assert archive.read('mask_full.png') == archive.read('mask.png')
        with Image.open(io.BytesIO(archive.read('image.tif'))) as image:
            assert image.mode == 'RGB' and image.size == (120, 100)
            assert image.tag_v2[259] == 5  # TIFF LZW
            assert tuple(round(float(x)) for x in image.info['dpi']) == (300, 300)


def test_annotation_edit_updates_artifacts_and_revokes_download(studio):
    opened = studio.post('/api/backgrounds/clean-1/open').json()
    result = finish(studio, {'image_id': opened['id'], 'brushes': [{'points': [20, 20, 70, 55]}]})['result']
    rid = result['id']
    studio.post(f'/api/results/{rid}/approve', json={'confirmed': True})
    download = studio.post(f'/api/results/{rid}/export').json()['url']
    polygons = [[1, 1, 9, 1, 9, 8, 1, 8]]
    updated = studio.put(f'/api/results/{rid}/annotations', json={'polygons': polygons}).json()
    with Image.open(io.BytesIO(studio.get(updated['label_mask']).content)) as mask:
        assert mask.getpixel((2, 2)) == 255 and mask.getpixel((30, 30)) == 0
        segmentation = studio.get(updated['polygon_file']).json()['annotations'][0]['segmentation']
        assert np.array_equal(np.asarray(mask), decode_coco(segmentation, mask.size))
    assert studio.get(download).status_code == 409


def test_approved_labels_detect_disk_tampering(studio):
    opened = studio.post('/api/backgrounds/clean-1/open').json()
    result = finish(studio, {'image_id': opened['id'], 'brushes': [{'points': [20, 20, 70, 55]}]})['result']
    rid = result['id']
    studio.post(f'/api/results/{rid}/approve', json={'confirmed': True})
    (studio.app.state.store.out/f'{rid}-labels.png').write_bytes(b'tampered')
    assert studio.post(f'/api/results/{rid}/export').status_code == 422


def test_operations_are_serial_and_persist_real_progress(tmp_path):
    jobs = Jobs(Store(tmp_path))
    entered = threading.Event(); release = threading.Event(); sequence = []
    class Training:
        def train(self, payload, progress, cancelled):
            sequence.append(payload['seed'])
            if payload['seed'] == 1:
                entered.set(); release.wait(5)
            progress(50, 'training')
            return {'lora_id': 'b'*32}
    jobs.training = Training()
    try:
        one = jobs.submit_operation('train', {'seed': 1})
        assert entered.wait(2)
        two = jobs.submit_operation('train', {'seed': 2})
        assert jobs.store.load('job', two['id'])['status'] == 'queued'
        release.set()
        assert await_job(jobs, one['id'])['operation_result']['lora_id'] == 'b'*32
        assert await_job(jobs, two['id'])['progress'] == 100
        assert sequence == [1, 2]
    finally:
        release.set(); jobs.close()


def test_cancellation_prevents_queued_work(tmp_path):
    jobs = Jobs(Store(tmp_path)); entered = threading.Event(); release = threading.Event(); calls = []
    class Training:
        def train(self, payload, progress, cancelled):
            calls.append(payload['seed']); entered.set(); release.wait(5)
            return {}
    jobs.training = Training()
    try:
        first = jobs.submit_operation('train', {'seed': 1}); assert entered.wait(2)
        second = jobs.submit_operation('train', {'seed': 2})
        assert jobs.cancel(second['id'])['status'] == 'cancelled'
        release.set(); await_job(jobs, first['id'])
        assert await_job(jobs, second['id'])['status'] == 'cancelled'
        assert calls == [1]
    finally:
        release.set(); jobs.close()


def test_cancel_running_operation_persists_terminal_state(tmp_path):
    jobs = Jobs(Store(tmp_path)); entered = threading.Event()
    class Training:
        def train(self, payload, progress, cancelled):
            entered.set()
            for _ in range(200):
                if cancelled():
                    raise RuntimeError('Cancelled')
                time.sleep(.005)
            raise AssertionError('Cancellation never delivered')
    jobs.training = Training()
    try:
        operation = jobs.submit_operation('train', {})
        assert entered.wait(2)
        jobs.cancel(operation['id'])
        assert await_job(jobs, operation['id'])['status'] == 'cancelled'
    finally:
        jobs.close()


def test_restart_reports_interrupted_job(tmp_path):
    store = Store(tmp_path); identifier = 'c'*32
    store.save('job', identifier, {'id': identifier, 'status': 'running', 'kind': 'train', 'progress': 45})
    jobs = Jobs(store)
    try:
        assert store.load('job', identifier)['status'] == 'failed'
        assert 'stopped' in store.load('job', identifier)['error']
    finally:
        jobs.close()


def test_operation_api_validates_unknown_fields_and_models(studio):
    for route in ['/api/training/lora', '/api/library/generate']:
        response = studio.post(route, json={'base_model': 'arbitrary/path', 'stamp_ids': ['one']})
        assert response.status_code == 422
    assert studio.post('/api/jobs/'+'f'*32+'/cancel').status_code == 404
    assert studio.get('/api/training').json()['status'] == 'local LoRA training'


def test_random_brush_sources_are_immutable_and_saved_per_stroke(tmp_path):
    jobs = Jobs(Store(tmp_path))
    stamps = {name: {'id': name, 'split': 'train', 'rgb': name+'/stamp_crop.tif'}
              for name in ['a', 'b', 'c']}
    stamps['held'] = {'id': 'held', 'split': 'test', 'rgb': 'held/stamp_crop.tif'}
    scene = {'brush_method': 'stamp_brush', 'brush_stamp_mode': 'random', 'seed': 7,
             'brushes': [{'points': [1, 1, 20, 20]} for _ in range(5)]}
    try:
        resolved = jobs.resolve_brush_stamps(scene, stamps)
        assert len({brush['stamp_id'] for brush in resolved['brushes'][:3]}) == 3
        assert resolved == jobs.resolve_brush_stamps(scene, stamps)
        assert all(brush['stamp_id'] in {'a', 'b', 'c'} for brush in resolved['brushes'])
        assert all('stamp_id' not in brush for brush in scene['brushes'])
        with pytest.raises(ValueError, match='No train TIFF'):
            jobs.resolve_brush_stamps(scene, {})
    finally:
        jobs.close()


def test_cancel_after_atomic_publication_does_not_claim_cancelled(tmp_path):
    jobs = Jobs(Store(tmp_path)); committed = threading.Event(); release = threading.Event()
    class Training:
        def train(self, payload, progress, cancelled):
            with jobs.store.lock:
                progress(100, phase='completed')
                committed.set()
            release.wait(5)
            return {'lora_id': 'a'*32}
    jobs.training = Training()
    try:
        operation = jobs.submit_operation('train', {})
        assert committed.wait(2)
        assert jobs.cancel(operation['id'])['committed'] is True
        release.set()
        assert await_job(jobs, operation['id'])['status'] == 'succeeded'
    finally:
        release.set(); jobs.close()


def test_ai_lora_training_sources_remain_part_of_export_gate(studio):
    from backend.library import Library
    image = io.BytesIO(); Image.new('RGB', (24, 24), 'gray').save(image, 'TIFF')
    mask = Image.new('L', (24, 24)); mask.paste(255, (8, 3, 12, 20))
    encoded_mask = io.BytesIO(); mask.save(encoded_mask, 'PNG')
    store = studio.app.state.store
    stamp = Library(store).import_pair(image.getvalue(), mode='crop', source_id='new-train-source',
                                       confirmed_train=True, mask_bytes=encoded_mask.getvalue())[0]
    opened = studio.post('/api/backgrounds/clean-1/open').json()
    result = finish(studio, {'image_id': opened['id'], 'brushes': [{'points': [20, 20, 70, 55]}]})['result']
    rid = result['id']
    record = store.load('result', rid)
    store.save('result', rid, {**record, 'provenance': {**record['provenance'],
                                                      'lora': {'id': 'f'*32, 'sources': [stamp]}}})
    assert studio.post(f'/api/results/{rid}/approve', json={'confirmed': True}).status_code == 200
    assert studio.post(f'/api/results/{rid}/export').status_code == 200
    path = store.root/'data/fullframe/index.json'
    write_json(path, [*json.loads(path.read_text()), {'id': 'new-train-source', 'split': 'val'}])
    assert studio.post(f'/api/results/{rid}/export').status_code == 422
