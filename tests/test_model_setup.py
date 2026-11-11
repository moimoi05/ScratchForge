"""Offline tests for reproducible local model acquisition."""
import json

import pytest

from scripts.download_model import download_model


def test_download_pins_commit_and_records_provenance(tmp_path):
    calls = []

    def info(model_id, revision):
        assert revision == 'main'
        return type('Info', (), {'sha': 'a' * 40})()

    def snapshot(**kwargs):
        calls.append(kwargs)
        (tmp_path / 'model_index.json').write_text('{}')
        return str(tmp_path)

    metadata = download_model(tmp_path, model_info=info, snapshot_download=snapshot)
    assert calls[0]['revision'] == 'a' * 40
    assert calls[0]['local_dir'] == str(tmp_path)
    assert '*.fp16.safetensors' in calls[0]['allow_patterns']
    assert '*.safetensors' not in calls[0]['allow_patterns']
    assert not any('.bin' in pattern for pattern in calls[0]['allow_patterns'])
    assert json.loads((tmp_path / 'model-provenance.json').read_text()) == metadata
    assert metadata['resolved_revision'] == 'a' * 40


def test_failed_download_has_no_success_record(tmp_path):
    def fail(**kwargs):
        raise OSError('No space left')

    info = lambda *args, **kwargs: type('Info', (), {'sha': 'b' * 40})()
    with pytest.raises(OSError, match='No space'):
        download_model(tmp_path, model_info=info, snapshot_download=fail)
    assert not (tmp_path / 'model-provenance.json').exists()


def test_existing_different_model_commit_is_refused(tmp_path):
    (tmp_path / 'model-provenance.json').write_text(json.dumps({
        'model_id': 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1',
        'resolved_revision': 'a' * 40,
    }))
    info = lambda *args, **kwargs: type('Info', (), {'sha': 'b' * 40})()
    with pytest.raises(ValueError, match='revision'):
        download_model(tmp_path, model_info=info, snapshot_download=lambda **kwargs: None)


def test_sd15_download_uses_public_successor_and_shared_fp16_weights(tmp_path):
    calls = []

    def info(model_id, revision):
        assert model_id == 'stable-diffusion-v1-5/stable-diffusion-inpainting'
        return type('Info', (), {'sha': 'c' * 40})()

    def snapshot(**kwargs):
        calls.append(kwargs)
        (tmp_path/'model_index.json').write_text('{}')

    metadata = download_model(tmp_path, model='sd15', model_info=info, snapshot_download=snapshot)
    assert metadata['model_id'] == calls[0]['repo_id']
    assert '*.fp16.safetensors' in calls[0]['allow_patterns']


def test_sd15_cli_default_destination_and_incomplete_download(tmp_path, monkeypatch):
    import runpy
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    import scripts.download_model as module
    calls = []
    def snapshot(**kwargs):
        calls.append(kwargs)
        raise OSError('download interrupted')
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        HfApi=lambda: SimpleNamespace(model_info=lambda *a, **k: SimpleNamespace(sha='d'*40)),
        snapshot_download=snapshot))
    monkeypatch.setattr(sys, 'argv', ['download_model.py', '--model', 'sd15'])
    script = tmp_path/'scripts/download_model.py'
    script.parent.mkdir()
    script.write_text(Path(module.__file__).read_text())
    with pytest.raises(OSError, match='interrupted'):
        runpy.run_path(str(script), run_name='__main__')
    assert Path(calls[0]['local_dir']).name == 'sd15-inpaint'
    assert calls[0]['repo_id'] == module.MODELS['sd15']
    with pytest.raises(OSError, match='incomplete'):
        download_model(tmp_path, model='sd15', snapshot_download=lambda **kwargs: None)
    assert not (tmp_path/'model-provenance.json').exists()
