"""Offline registration tests; never download or execute checkpoint contents."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts.setup_harmonization import SOURCES, register_model


def fake_git(model, calls, *, revision=None, dirty=False, remote=None):
    repository, expected_revision = SOURCES[model]

    def run(arguments, **kwargs):
        calls.append(arguments)
        assert kwargs == {'check': True, 'capture_output': True, 'text': True}
        if arguments[1] == 'clone':
            from pathlib import Path
            (Path(arguments[-1])/'.git').mkdir(parents=True)
        if arguments[-2:] == ['rev-parse', 'HEAD']:
            output = revision or expected_revision
        elif arguments[-2:] == ['status', '--porcelain']:
            output = ' M source.py' if dirty else ''
        elif arguments[-3:] == ['remote', 'get-url', 'origin']:
            output = remote or repository
        else:
            output = ''
        return SimpleNamespace(stdout=output)

    return run


@pytest.mark.parametrize('model', tuple(SOURCES))
def test_register_pinned_source_and_checked_local_checkpoint(tmp_path, model):
    checkpoint = tmp_path/'weights.pth'
    checkpoint.write_bytes(b'local state dictionary placeholder')
    calls = []
    metadata = register_model(model, checkpoint, root=tmp_path, runner=fake_git(model, calls))
    folder = tmp_path/'data/models/harmonization'/model
    assert (folder/'checkpoint.pth').read_bytes() == checkpoint.read_bytes()
    assert metadata['source_revision'] == SOURCES[model][1]
    assert metadata['checkpoint_sha256'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert metadata == json.loads((folder/'model-provenance.json').read_text())
    assert any(command[1:3] == ['clone', '--no-checkout'] for command in calls)
    assert any(command[-3:] == ['checkout', '--detach', SOURCES[model][1]] for command in calls)


def test_idempotent_registration_reuses_checkout_without_clone(tmp_path):
    checkpoint = tmp_path/'weights.pt'; checkpoint.write_bytes(b'weights')
    register_model('rainnet', checkpoint, root=tmp_path, runner=fake_git('rainnet', []))
    calls = []
    register_model('rainnet', checkpoint, root=tmp_path, runner=fake_git('rainnet', calls))
    assert not any(command[1] == 'clone' for command in calls)


@pytest.mark.parametrize('filename,content', [('missing.pth', None), ('unsafe.exe', b'data'), ('empty.pth', b'')])
def test_reject_checkpoint_before_any_network(tmp_path, filename, content):
    checkpoint = tmp_path/filename
    if content is not None:
        checkpoint.write_bytes(content)
    calls = []
    with pytest.raises(ValueError, match='checkpoint'):
        register_model('pctnet', checkpoint, root=tmp_path, runner=fake_git('pctnet', calls))
    assert calls == []


def test_refuse_replacing_different_weights(tmp_path):
    checkpoint = tmp_path/'weights.pth'; checkpoint.write_bytes(b'original')
    register_model('rainnet', checkpoint, root=tmp_path, runner=fake_git('rainnet', []))
    checkpoint.write_bytes(b'changed')
    calls = []
    with pytest.raises(ValueError, match='different checkpoint'):
        register_model('rainnet', checkpoint, root=tmp_path, runner=fake_git('rainnet', calls))
    assert calls == []
    assert (tmp_path/'data/models/harmonization/rainnet/checkpoint.pth').read_bytes() == b'original'


@pytest.mark.parametrize('options,message', [({'dirty': True}, 'uncommitted'),
                                           ({'revision': 'bad'}, 'revision'),
                                           ({'remote': 'https://example.com/untrusted.git'}, 'repository')])
def test_reject_modified_checkout_without_reset(tmp_path, options, message):
    checkpoint = tmp_path/'weights.pth'; checkpoint.write_bytes(b'weights')
    folder = tmp_path/'data/models/harmonization/harmonizer/source'
    (folder/'.git').mkdir(parents=True)
    calls = []
    with pytest.raises(ValueError, match=message):
        register_model('harmonizer', checkpoint, root=tmp_path,
                       runner=fake_git('harmonizer', calls, **options))
    assert not any('checkout' in command or 'reset' in command for command in calls)
    assert not (folder.parent/'checkpoint.pth').exists()


def test_unknown_model_rejected(tmp_path):
    with pytest.raises(ValueError, match='Unknown'):
        register_model('../other', tmp_path/'weights.pth', root=tmp_path)
