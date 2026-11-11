import io
import json

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from PIL import Image

from backend.store import Store


def client_for(tmp_path):
    from backend.library_api import library_router
    class Jobs:
        def submit_operation(self, kind, payload):
            return {'id': 'a' * 32, 'kind': kind, 'payload': payload, 'status': 'queued'}
    app = FastAPI()
    app.add_exception_handler(ValueError, lambda request, error: JSONResponse({'detail': str(error)}, status_code=422))
    app.include_router(library_router(Store(tmp_path), Jobs()))
    return TestClient(app)


def encoded(image, fmt):
    output = io.BytesIO()
    image.save(output, format=fmt)
    return output.getvalue()


def test_pair_import_review_and_queue(tmp_path):
    client = client_for(tmp_path)
    mask = Image.new('L', (40, 30)); mask.paste(255, (5, 10, 35, 15))
    files = {'image': ('source.tif', encoded(Image.new('RGB', (40, 30), 'blue'), 'TIFF'), 'image/tiff'),
             'mask': ('mask.png', encoded(mask, 'PNG'), 'image/png')}
    response = client.post('/api/library/import', files=files, data={
        'mode': 'crop', 'source_id': 'new-frame', 'confirmed_train': 'true'})
    assert response.status_code == 200, response.text
    record = response.json()[0]
    assert client.get('/api/library/' + record['id']).json()['id'] == record['id']
    assert client.post('/api/library/' + record['id'] + '/approve', json={'confirmed': True}).json()['review_status'] == 'APPROVED'
    augmented = client.post('/api/library/augment', json={'stamp_ids': [record['id']], 'copies': 3, 'seed': 8})
    assert augmented.json()['kind'] == 'augment'
    revised = client.put('/api/library/' + record['id'] + '/annotations', json={
        'polygons': [[6, 10, 30, 10, 30, 13, 6, 13]]})
    assert revised.status_code == 200 and revised.json()['id'] != record['id']
    assert client.get('/api/library/missing').status_code == 404


def test_pair_import_validates_required_files_and_json(tmp_path):
    client = client_for(tmp_path)
    image = ('image.tif', encoded(Image.new('RGB', (40, 30)), 'TIFF'), 'image/tiff')
    for files in ({'image': image}, {'image': image, 'annotations': ('bad.json', b'{', 'application/json')}):
        response = client.post('/api/library/import', files=files, data={
            'mode': 'fullframe', 'source_id': 'new-frame', 'confirmed_train': 'true'})
        assert response.status_code == 422
    for payload in ({'stamp_ids': [], 'copies': 1}, {'stamp_ids': ['one'], 'copies': 0},
                    {'stamp_ids': ['one'], 'seed': -1}, {'stamp_ids': ['one'], 'unknown': True}):
        assert client.post('/api/library/augment', json=payload).status_code == 422
