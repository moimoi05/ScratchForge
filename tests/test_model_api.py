from fastapi.testclient import TestClient
from backend.main import create_app

class Downloads:
    def __init__(self): self.closed = False
    def status(self): return [{'model_id': 'pctnet', 'status': 'not_downloaded'}]
    def start(self, model_id): return {'model_id': model_id, 'status': 'downloading'}
    def cancel(self, model_id): return {'model_id': model_id, 'status': 'cancelled'}
    def close(self): self.closed = True

def test_download_api_validation_and_shutdown(tmp_path):
    manager = Downloads()
    with TestClient(create_app(tmp_path, downloads=manager)) as client:
        assert client.get('/api/models/status').json()['models'][0]['model_id'] == 'pctnet'
        assert client.post('/api/models/download', json={'model_id':'pctnet'}).json()['status'] == 'downloading'
        assert client.post('/api/models/cancel', json={'model_id':'pctnet'}).json()['status'] == 'cancelled'
        for body in [{'model_id':'../x'}, {'model_id':[]}, {'model_id':'sdxl','url':'https://evil.test'}]:
            assert client.post('/api/models/download', json=body).status_code == 422
        assert client.post('/api/models/cancel', json={'model_id':'pctnet'}, headers={'Origin':'https://evil.test'}).status_code == 403
    assert manager.closed


def test_serial_job_releases_only_inactive_model_caches(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from backend.jobs import Jobs
    from backend.store import Store
    from scratchforge.harmonization import StampHarmonizer
    cleared = []
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True, empty_cache=lambda: cleared.append(True))))
    jobs = Jobs(Store(tmp_path))
    jobs.ai.pipeline = object()
    active = object()
    jobs.sd15.pipeline = active
    jobs.harmonize = StampHarmonizer(tmp_path)
    kept = object()
    jobs.harmonize._loaded = {'pctnet': object(), 'rainnet': kept}
    jobs.release_inactive_models({'brushes':[{}], 'placements':[{}],
                                  'brush_method':'sd15', 'stamp_method':'rainnet'})
    assert jobs.ai.pipeline is None and jobs.sd15.pipeline is active
    assert jobs.harmonize._loaded == {'rainnet': kept}
    assert cleared
    jobs.release_inactive_models({'brushes':[], 'placements':[]})
    assert jobs.sd15.pipeline is None and jobs.harmonize._loaded == {}
    jobs.executor.shutdown()
