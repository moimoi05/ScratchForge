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
