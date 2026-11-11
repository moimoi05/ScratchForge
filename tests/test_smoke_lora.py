"""GPU acceptance gates; fixture tests are never hardware evidence."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image


def test_cpu_never_calls_training_or_reports_cuda_pass(tmp_path, monkeypatch):
    from scripts import smoke_lora
    monkeypatch.setattr(smoke_lora, 'inspect_gpu', lambda: {'status': 'NOT_AVAILABLE', 'available': False,
                                                         'kernel_pass': False})
    monkeypatch.setattr(smoke_lora, 'smoke_model', lambda *args: pytest.fail('CPU reached CUDA smoke'))
    report = smoke_lora.run_acceptance(tmp_path, ['sd15', 'sdxl'], [], 2)
    assert report['status'] == 'FAIL'
    assert report['models'] == []
    assert report['halcon']['status'] == 'NOT_RUN'


def test_partial_model_results_do_not_pass_acceptance(tmp_path, monkeypatch):
    from scripts import smoke_lora
    monkeypatch.setattr(smoke_lora, 'inspect_gpu', lambda: {'status': 'PASS', 'available': True, 'kernel_pass': True})
    monkeypatch.setattr(smoke_lora, 'smoke_model', lambda *args: {'base_model': args[1], 'status': 'PASS',
                        'training': True, 'weights_reloaded': True, 'cancelled_without_publication': False,
                        'fullframe_tiff': True, 'outside_mask_exact': True, 'scratch_detected': True})
    report = smoke_lora.run_acceptance(tmp_path, ['sd15'], ['stamp'], 2)
    assert report['status'] == 'FAIL'


def test_model_failure_is_recorded_and_other_model_still_checked(tmp_path, monkeypatch):
    from scripts import smoke_lora
    monkeypatch.setattr(smoke_lora, 'inspect_gpu', lambda: {'status': 'PASS', 'available': True, 'kernel_pass': True})
    called = []
    def fail(store, base, *args):
        called.append(base)
        raise RuntimeError('CUDA out of VRAM')
    monkeypatch.setattr(smoke_lora, 'smoke_model', fail)
    report = smoke_lora.run_acceptance(tmp_path, ['sd15', 'sdxl'], ['stamp'], 2)
    assert report['status'] == 'FAIL' and called == ['sd15', 'sdxl']
    assert 'VRAM' in report['models'][0]['error']


def test_native_verification_checks_pixels_metadata_and_labels(tmp_path):
    from scripts.smoke_lora import verify_native
    from scratchforge.dataset import mask_for
    original = Image.new('RGB', (40, 30), (10, 20, 30))
    polygons = [[10, 10, 20, 10, 20, 15, 10, 15]]
    mask = mask_for(polygons, original.size)
    changed = original.copy(); changed.paste((200, 180, 160), (0, 0, 40, 30), mask)
    record = verify_native(original, changed, mask, mask, polygons, tmp_path/'result.tif')
    assert record['fullframe_tiff'] and record['outside_mask_exact'] and record['scratch_detected']
    with Image.open(tmp_path/'result.tif') as image:
        assert image.mode == 'RGB' and image.tag_v2[259] == 5
        assert tuple(round(x) for x in image.info['dpi']) == (300, 300)
    changed.putpixel((0, 0), (1, 2, 3))
    with pytest.raises(ValueError, match='outside'):
        verify_native(original, changed, mask, mask, polygons, tmp_path/'bad.tif')


def test_acceptance_cli_writes_failure_report_and_nonzero_exit(tmp_path, monkeypatch):
    from scripts import smoke_lora
    monkeypatch.setattr(smoke_lora, 'run_acceptance', lambda *args: {'status': 'FAIL', 'models': []})
    assert smoke_lora.main(['--root', str(tmp_path)]) == 1
    assert (tmp_path/'reports/lora-smoke-runtime.json').is_file()


def test_cancel_smoke_requires_a_real_training_step_and_no_publication(tmp_path):
    from scripts.smoke_lora import _cancel_smoke
    class Training:
        registry = SimpleNamespace(directory=tmp_path)
        def checkpoints(self): return []
        def train(self, payload, progress, cancelled):
            progress(10, phase='training', step=1)
            assert cancelled()
            raise RuntimeError('LoRA training cancelled')
    assert _cancel_smoke(Training(), {})
    (tmp_path/'.sample.staging').mkdir()
    with pytest.raises(RuntimeError, match='staging'):
        _cancel_smoke(Training(), {})
    class Premature(Training):
        def train(self, *args): raise RuntimeError('CUDA unavailable')
    with pytest.raises(RuntimeError, match='CUDA'):
        _cancel_smoke(Premature(), {})
    class IgnoresCancel(Training):
        def train(self, payload, progress, cancelled): return {}
    with pytest.raises(RuntimeError, match='instead of cancelling'):
        _cancel_smoke(IgnoresCancel(), {})


def test_background_requires_verified_full_frame_and_matching_checksum(tmp_path):
    from scripts.smoke_lora import _background
    from scratchforge.dataset import sha
    class Store:
        def index(self): return rows
        def asset(self, path): return tmp_path/path
    rows = []
    with pytest.raises(ValueError, match='train OK TIFF'):
        _background(Store())
    # Header claims native dimensions but content does not: reject after checksum validation.
    source = tmp_path/'background.tif'
    Image.new('RGB', (60, 40)).save(source)
    rows = [{'id': 'background', 'status': 'OK', 'split': 'train', 'width': 5861, 'height': 3421,
             'path': source.name, 'sha256': 'wrong'}]
    with pytest.raises(ValueError, match='checksum'):
        _background(Store())
    rows[0] = {**rows[0], 'sha256': sha(source)}
    with pytest.raises(ValueError, match='full-resolution'):
        _background(Store())
    rows[0] = {**rows[0], 'width': 60, 'height': 40}
    assert _background(Store())[0].size == (60, 40)


def test_smoke_model_orchestration_with_fixture_only_not_hardware(tmp_path, monkeypatch):
    import sys
    import backend.training
    import scratchforge.ai
    import scratchforge.generation
    from scripts import smoke_lora
    from backend.store import Store
    from scratchforge.dataset import mask_for
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(reset_peak_memory_stats=lambda: None,
        synchronize=lambda: None, max_memory_allocated=lambda: 1024**3, empty_cache=lambda: None))
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    class Training:
        def __init__(self, store): pass
        def train(self, payload, progress, cancelled):
            progress(100, phase='training', step=payload['steps'])
            return {'lora_id': 'f'*32, 'checkpoint': {'metrics': {'steps': payload['steps']}}}
    class Adapter:
        def __init__(self, root): self.pipeline = object()
        def set_lora(self, identifier): self.loaded_lora_id = identifier
        def metadata(self): return {'lora_loaded': bool(self.loaded_lora_id)}
    monkeypatch.setattr(backend.training, 'Training', Training)
    monkeypatch.setattr(scratchforge.ai, 'SD15Generator', Adapter)
    monkeypatch.setattr(scratchforge.ai, 'SDXLGenerator', Adapter)
    monkeypatch.setattr(smoke_lora, '_cancel_smoke', lambda *args: True)
    original = Image.new('RGB', (60, 40), (10, 20, 30))
    polygons = [[5, 5, 20, 5, 20, 12, 5, 12]]
    mask = mask_for(polygons, original.size)
    output = original.copy(); output.paste((100, 110, 120), (0, 0, *original.size), mask)
    monkeypatch.setattr(smoke_lora, '_background', lambda store: (original, {'id': 'background', 'sha256': 'digest'}))
    monkeypatch.setattr(scratchforge.generation, 'generate_scene', lambda *args, **kwargs:
                        SimpleNamespace(image=output, edit=mask, label_mask=mask, polygons=polygons))
    store = Store(tmp_path)
    for base in ['sd15', 'sdxl']:
        result = smoke_lora.smoke_model(store, base, ['approved-stamp'], 2)
        assert result['weights_reloaded'] and result['inference_peak_allocated_gib'] == 1
        assert result['background_id'] == 'background'
    with pytest.raises(ValueError, match='Approve'):
        smoke_lora.smoke_model(store, 'sd15', [], 2)


def test_complete_model_results_pass_only_gpu_scoped_acceptance(tmp_path, monkeypatch):
    from scripts import smoke_lora
    monkeypatch.setattr(smoke_lora, 'inspect_gpu', lambda: {'status': 'PASS', 'available': True, 'kernel_pass': True})
    monkeypatch.setattr(smoke_lora, 'smoke_model', lambda *args: {'base_model': args[1], 'status': 'PASS',
                        **{key: True for key in smoke_lora.CHECKS}})
    report = smoke_lora.run_acceptance(tmp_path, ['sd15', 'sdxl'], ['stamp'], 2)
    assert report['status'] == 'PASS' and report['halcon']['status'] == 'NOT_RUN'
    monkeypatch.setattr(smoke_lora, 'run_acceptance', lambda *args: report)
    assert smoke_lora.main(['--root', str(tmp_path), '--base-model', 'sd15', '--steps', '2']) == 0
    with pytest.raises(SystemExit):
        smoke_lora.main(['--root', str(tmp_path), '--steps', '0'])


def test_native_verification_rejects_empty_or_wrong_size_results(tmp_path):
    from scripts.smoke_lora import verify_native
    original = Image.new('RGB', (30, 20))
    mask = Image.new('L', original.size)
    with pytest.raises(ValueError, match='dimensions'):
        verify_native(original, Image.new('RGB', (1, 1)), mask, mask, [], tmp_path/'bad.tif')
    with pytest.raises(ValueError, match='scratch'):
        verify_native(original, original, mask, mask, [], tmp_path/'bad.tif')
