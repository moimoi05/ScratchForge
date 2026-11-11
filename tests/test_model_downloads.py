from pathlib import Path
import pytest
from backend.model_downloads import ModelDownloads


def test_missing_and_invalid(tmp_path):
    manager = ModelDownloads(tmp_path)
    assert len(manager.status()) == 5
    assert all(x['status'] == 'not_downloaded' for x in manager.status())
    with pytest.raises(ValueError): manager.start('../bad')
    manager.close()


def test_cancel_kills_worker_and_cleans_partial(tmp_path):
    class Process:
        killed = False
        def poll(self): return None if not self.killed else -15
        def terminate(self): self.killed = True
        def wait(self, timeout=None): return -15
    process = Process()
    manager = ModelDownloads(tmp_path, launcher=lambda *a, **k: process)
    assert manager.start('pctnet')['status'] == 'downloading'
    stage = tmp_path/'data/models/.downloads/pctnet'
    (stage/'checkpoint.part').write_bytes(b'partial')
    assert manager.cancel('pctnet')['status'] == 'cancelled'
    assert process.killed and not stage.exists()
    manager.close()


def test_worker_success_is_published_once_and_deleted_cache_is_not_ready(tmp_path, monkeypatch):
    from backend import model_downloads as module
    class Process:
        def poll(self): return 0
    manager = ModelDownloads(tmp_path, launcher=lambda *a, **k: Process())
    def verify(folder, model_id, **kwargs):
        if not (folder/'checkpoint.pth').is_file(): raise ValueError('checkpoint missing')
        return True
    monkeypatch.setattr(module, 'verify_model', verify)
    manager.start('pctnet')
    stage = tmp_path/'data/models/.downloads/pctnet'
    payload = stage/'payload'
    payload.mkdir(parents=True)
    (payload/'checkpoint.pth').write_bytes(b'weights')
    (payload/'download-manifest.json').write_text('{}')
    (stage/'progress.json').write_text('{"status":"verifying","progress":99}')
    ready = next(row for row in manager.status() if row['model_id']=='pctnet')
    assert ready['status']=='ready' and ready['progress']==100
    assert not manager.tasks
    assert next(row for row in manager.status() if row['model_id']=='pctnet')['status']=='ready'
    (tmp_path/'data/models/harmonization/pctnet/checkpoint.pth').unlink()
    assert next(row for row in manager.status() if row['model_id']=='pctnet')['status']=='not_downloaded'
    manager.close()


def test_exited_worker_without_payload_is_error_and_retry_cleans_stage(tmp_path):
    class Process:
        def poll(self): return 0
    manager=ModelDownloads(tmp_path, launcher=lambda *a, **k: Process())
    manager.start('sd15')
    status=next(row for row in manager.status() if row['model_id']=='sd15')
    assert status['status']=='error'
    assert next(row for row in manager.status() if row['model_id']=='sd15')['status']=='error'
    manager.close()
