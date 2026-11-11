import json
from types import SimpleNamespace

import pytest

from scratchforge.ai import SD15Generator
from scratchforge.dataset import sha
from scratchforge.lora import LoraRegistry, model_info


def checkpoint(root, identifier='a' * 32, base='sd15', revision='fixed'):
    folder = root / 'data/models/lora' / identifier
    folder.mkdir(parents=True)
    weights = folder / 'pytorch_lora_weights.safetensors'
    weights.write_bytes(b'weights')
    row = {'id': identifier, 'base_model': base, 'base_revision': revision,
           'weights': f'{identifier}/{weights.name}', 'sha256': sha(weights),
           'status': 'succeeded', 'steps': 2, 'sources': [{'id': 'real', 'split': 'train'}]}
    (folder / 'manifest.json').write_text(json.dumps(row))
    (folder.parent / 'index.json').write_text(json.dumps([row]))
    return row, weights


def test_registry_compatibility_checksum_and_safe_paths(tmp_path):
    row, weights = checkpoint(tmp_path)
    registry = LoraRegistry(tmp_path)
    assert registry.list()[0]['id'] == row['id']
    assert registry.resolve(row['id'], 'sd15', 'fixed')[1] == weights
    with pytest.raises(ValueError, match='base'):
        registry.resolve(row['id'], 'sdxl', 'fixed')
    with pytest.raises(ValueError, match='revision'):
        registry.resolve(row['id'], 'sd15', 'other')
    with pytest.raises(ValueError, match='identifier'):
        registry.resolve('../escape', 'sd15', 'fixed')
    weights.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        registry.resolve(row['id'], 'sd15', 'fixed')


def test_registry_rejects_escaping_weight_path_and_unfinished(tmp_path):
    row, _ = checkpoint(tmp_path)
    index = tmp_path / 'data/models/lora/index.json'
    index.write_text(json.dumps([{**row, 'weights': '../outside.safetensors'}]))
    with pytest.raises(ValueError, match='unsafe|Unsafe'):
        LoraRegistry(tmp_path).resolve(row['id'], 'sd15', 'fixed')
    index.write_text(json.dumps([{**row, 'status': 'cancelled'}]))
    assert LoraRegistry(tmp_path).list() == []


def test_adapter_reports_only_actually_loaded_lora_and_unloads(tmp_path, monkeypatch):
    row, weights = checkpoint(tmp_path)
    monkeypatch.setattr(LoraRegistry, 'verify_sources', lambda self, row: None)
    adapter = SD15Generator(tmp_path)
    adapter.directory.mkdir(parents=True)
    (adapter.directory / 'model_index.json').write_text('{}')
    (adapter.directory / 'model-provenance.json').write_text(json.dumps({'resolved_revision': 'fixed'}))
    adapter.set_lora(row['id'])
    assert adapter.metadata()['lora_loaded'] is False
    events = []
    adapter.pipeline = SimpleNamespace(
        load_lora_weights=lambda directory, **kw: events.append(('load', directory, kw)),
        unload_lora_weights=lambda: events.append(('unload',)))
    adapter.set_lora(row['id'])
    assert adapter.metadata()['loaded_lora_id'] == row['id']
    assert events[0][1] == str(weights.parent)
    adapter.set_lora(None)
    assert events[-1] == ('unload',)
    assert adapter.metadata()['lora_loaded'] is False


def test_model_info_rejects_unknown_and_requires_local_base(tmp_path):
    with pytest.raises(ValueError, match='base'):
        model_info(tmp_path, 'anything')
    with pytest.raises(RuntimeError, match='missing'):
        model_info(tmp_path, 'sdxl')


def test_registry_requires_reviewed_training_sources(tmp_path):
    registry = LoraRegistry(tmp_path)
    with pytest.raises(ValueError, match='provenance'):
        registry.verify_sources({'sources': []})
    with pytest.raises(ValueError, match='not approved'):
        registry.verify_sources({'sources': [{'id': 'pending', 'review_status': 'pending'}]})


def test_failed_switch_does_not_leave_previous_lora_loaded(tmp_path, monkeypatch):
    row, _ = checkpoint(tmp_path)
    monkeypatch.setattr(LoraRegistry, 'verify_sources', lambda self, row: None)
    adapter = SD15Generator(tmp_path)
    adapter.directory.mkdir(parents=True)
    (adapter.directory / 'model_index.json').write_text('{}')
    (adapter.directory / 'model-provenance.json').write_text(json.dumps({'resolved_revision': 'fixed'}))
    events = []
    adapter.pipeline = SimpleNamespace(load_lora_weights=lambda *a, **k: None,
                                       unload_lora_weights=lambda: events.append('unload'))
    adapter.set_lora(row['id'])
    with pytest.raises(ValueError):
        adapter.set_lora('b' * 32)
    assert events == ['unload']
    assert adapter.metadata()['lora_loaded'] is False
