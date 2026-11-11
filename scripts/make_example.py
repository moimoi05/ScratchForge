"""Real full-frame baseline smoke; leaves results pending human review."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app
from scratchforge.dataset import ROOT, save_json


def main():
    with TestClient(app) as client:
        backgrounds = client.get('/api/backgrounds').json()
        stamps = client.get('/api/library').json()
        if not backgrounds or not stamps:
            raise RuntimeError('Run import_f971.py and build_library.py first.')
        source = backgrounds[0]
        opened = client.post(f"/api/backgrounds/{source['id']}/open")
        opened.raise_for_status()
        image = opened.json()
        scene = {'image_id': image['id'], 'engine': 'baseline', 'seed': 0,
                 'placements': [{'stamp_id': stamps[0]['id'], 'x': image['width']/2,
                                 'y': image['height']/2, 'scale': 1, 'rotation': 0, 'flip': False}],
                 'brushes': []}
        submitted = client.post('/api/jobs', json=scene)
        submitted.raise_for_status()
        deadline = time.monotonic() + 120
        while True:
            job = client.get('/api/jobs/' + submitted.json()['id']).json()
            if job['status'] == 'failed':
                raise RuntimeError(job['error'])
            if job['status'] == 'succeeded':
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Baseline smoke did not finish within 120 seconds.')
            time.sleep(.25)
        result = job['result']
        store = app.state.store
        source_record = store.load('input', image['id'])
        with Image.open(store.input_image(source_record)) as before, \
             Image.open(store.result_image(result)) as after, \
             Image.open(ROOT / result['edit_mask'].lstrip('/').replace('files/', 'outputs/', 1)) as mask:
            outside = np.asarray(mask) == 0
            assert before.format == after.format == 'TIFF'
            assert before.size == after.size == (image['width'], image['height'])
            assert np.array_equal(np.asarray(before)[outside], np.asarray(after)[outside])
        blocked = client.post(f"/api/results/{result['id']}/export")
        assert blocked.status_code == 409
        report = {'engine': 'baseline', 'source': source['source_image'],
                  'width': image['width'], 'height': image['height'], 'result': result,
                  'outside_edit_region_exact': True, 'unreviewed_export_blocked': True,
                  'note': 'Software smoke only; image and polygons await human review.'}
        save_json(ROOT/'reports/example-runtime.json', report)
        print(json.dumps({key: value for key, value in report.items() if key != 'result'}, indent=2))
        print('Preview: http://127.0.0.1:8000' + result['preview'])


if __name__ == '__main__':
    main()
