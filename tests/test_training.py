import json

import pytest

from backend.store import Store
from backend.training import GenerateStampsRequest, TrainLoraRequest, Training


def test_request_defaults_and_limits():
    request = TrainLoraRequest(stamp_ids=['one'])
    assert request.base_model == 'sd15' and request.steps == 1000
    for field, value in [('steps', 0), ('base_model', 'arbitrary'), ('seed', -1)]:
        with pytest.raises(ValueError):
            TrainLoraRequest(stamp_ids=['one'], **{field: value})
    with pytest.raises(ValueError):
        GenerateStampsRequest(stamp_ids=['one'], lora_id='../escape')


def test_training_without_cuda_fails_explicitly(tmp_path, monkeypatch):
    training = Training(Store(tmp_path))
    monkeypatch.setattr(training, '_snapshot', lambda ids: [])
    with pytest.raises(RuntimeError, match='CUDA'):
        training.train({'stamp_ids': ['one']}, lambda *a, **k: None, lambda: False)
    assert training.checkpoints() == []


def test_cancelled_before_training_never_publishes(tmp_path):
    training = Training(Store(tmp_path))
    with pytest.raises(RuntimeError, match='cancel'):
        training.train({'stamp_ids': ['one']}, lambda *a, **k: None, lambda: True)
    assert not (tmp_path / 'data/models/lora/index.json').exists()


def test_only_approved_real_or_augmentation_can_train(tmp_path, monkeypatch):
    from backend.library import Library
    store = Store(tmp_path)
    monkeypatch.setattr(store, 'library', lambda: [
        {'id': 'one', 'split': 'train', 'review_status': 'pending', 'origin': 'import'}])
    monkeypatch.setattr(Library, 'verify_stamp', lambda self, row, **kwargs: None)
    training = Training(store)
    with pytest.raises(ValueError, match='approved'):
        training._snapshot(['one'])
    monkeypatch.setattr(store, 'library', lambda: [
        {'id': 'one', 'split': 'train', 'review_status': 'approved', 'origin': 'synthetic'}])
    with pytest.raises(ValueError, match='synthetic'):
        training._snapshot(['one'])


def source_and_model(tmp_path):
    import io
    from PIL import Image
    from backend.library import Library

    store = Store(tmp_path)
    library = Library(store)
    rgb = Image.new('RGB', (80, 50), (110, 112, 115))
    mask = Image.new('L', rgb.size); mask.paste(255, (10, 20, 70, 26))
    streams = [io.BytesIO(), io.BytesIO()]
    rgb.save(streams[0], format='TIFF'); mask.save(streams[1], format='PNG')
    stamp = library.import_pair(streams[0].getvalue(), mode='crop', source_id='new-frame',
                               confirmed_train=True, mask_bytes=streams[1].getvalue())[0]
    stamp = library.approve(stamp['id'])
    folder = tmp_path/'data/models/sd15-inpaint'
    folder.mkdir(parents=True)
    (folder/'model_index.json').write_text('{}')
    (folder/'model-provenance.json').write_text(json.dumps({'resolved_revision': 'fixed'}))
    return store, stamp


def fake_weights(stage):
    import numpy as np
    from safetensors.numpy import save_file

    save_file({'unet.test.lora.down.weight': np.ones((2, 2), dtype='float32')},
              str(stage/'pytorch_lora_weights.safetensors'))


def test_successful_training_registry_and_generation_bundle(tmp_path, monkeypatch):
    import torch
    import numpy as np
    from PIL import Image
    from scratchforge.ai import SD15Generator

    store, stamp = source_and_model(tmp_path)
    training = Training(store)
    def worker(stage, progress, cancelled):
        request = json.loads((stage/'request.json').read_text())
        assert request['resolution'] == 512 and request['sources'][0]['id'] == stamp['id']
        fake_weights(stage)
        return {'final_loss': .1}
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(training, '_run_worker', worker)
    events = []
    result = training.train({'stamp_ids': [stamp['id']], 'steps': 2},
                            lambda p, **kw: events.append((p, kw)), lambda: False)
    assert training.checkpoints()[0]['id'] == result['lora_id']
    assert events[-1][1]['phase'] == 'completed'
    class Adapter:
        def set_lora(self, identifier): self.selected = identifier
        def __call__(self, image, mask, seed, strength):
            return Image.composite(Image.new('RGB', image.size, 'white'), image, mask)
    adapter = Adapter()
    generated = training.generate({'stamp_ids': [stamp['id']], 'lora_id': result['lora_id'], 'count': 2},
                                  adapter, lambda *a, **k: None, lambda: False)
    assert len(generated['stamps']) == 2
    for row in generated['stamps']:
        assert row['origin'] == 'synthetic' and row['review_status'] == 'PENDING_VISUAL_REVIEW'
        assert row['provenance']['training_stamps'][0]['id'] == stamp['id']
        assert (store.root/'data/library'/row['rgb']).is_file()
    with pytest.raises(ValueError, match='synthetic'):
        from backend.library import Library
        approved = Library(store).approve(generated['stamps'][0]['id'])
        training._snapshot([approved['id']])


def test_training_cancel_at_publication_never_registers(tmp_path, monkeypatch):
    import torch
    store, stamp = source_and_model(tmp_path)
    training = Training(store)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    state = {'cancel': False}
    def worker(stage, progress, cancelled):
        fake_weights(stage); state['cancel'] = True
        return {'final_loss': .1}
    monkeypatch.setattr(training, '_run_worker', worker)
    with pytest.raises(RuntimeError, match='cancelled'):
        training.train({'stamp_ids': [stamp['id']]}, lambda *a, **k: None, lambda: state['cancel'])
    assert training.checkpoints() == []
    assert not list((tmp_path/'data/models/lora').glob('*.staging'))


def test_invalid_weights_and_synthetic_input_determinism(tmp_path):
    import numpy as np
    from backend.training import _synthetic_input
    from safetensors.numpy import save_file

    store, stamp = source_and_model(tmp_path)
    directory = store.root/'data/library'
    first = _synthetic_input(directory, stamp, 7)
    second = _synthetic_input(directory, stamp, 7)
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(first, second))
    bad = tmp_path/'bad.safetensors'
    save_file({'unet.bad.lora.weight': np.full((1,), np.nan)}, str(bad))
    with pytest.raises(ValueError, match='non-finite'):
        Training._validate_weights(bad)
    with pytest.raises(RuntimeError, match='without'):
        Training._validate_weights(tmp_path/'missing.safetensors')


def test_training_subprocess_progress_success_failure_and_cancel(tmp_path, monkeypatch):
    import subprocess
    from backend.store import write_json

    flags = []
    class Process:
        def __init__(self, *args, **kwargs):
            flags.append(kwargs); self.returncode = None; self.calls = 0
        def poll(self):
            self.calls += 1
            if self.calls > 1: self.returncode = 0
            return self.returncode
        def terminate(self): self.returncode = -1
        def kill(self): self.returncode = -2
        def wait(self, **kwargs): return self.returncode
    monkeypatch.setattr(subprocess, 'Popen', Process)
    monkeypatch.setattr('backend.training.time.sleep', lambda _: None)
    write_json(tmp_path/'progress.json', {'progress': 33, 'phase': 'training'})
    write_json(tmp_path/'completed.json', {'status': 'succeeded', 'metrics': {'loss': .3}})
    events = []
    assert Training._run_worker(tmp_path, lambda p, **kw: events.append(p), lambda: False)['loss'] == .3
    assert events == [33] and flags[0]['stdout'] is flags[0]['stderr']
    write_json(tmp_path/'completed.json', {'status': 'failed', 'error': 'out of VRAM'})
    with pytest.raises(RuntimeError, match='VRAM'):
        Training._run_worker(tmp_path, lambda *a, **k: None, lambda: False)
    with pytest.raises(RuntimeError, match='cancel'):
        Training._run_worker(tmp_path, lambda *a, **k: None, lambda: True)
