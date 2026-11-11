import hashlib
import json
from pathlib import Path
import pytest

from scratchforge.model_files import digest, model_directory, verify_model


def test_verified_harmonization_manifest_and_tamper_detection(tmp_path):
    folder = model_directory(tmp_path, 'pctnet')
    (folder/'source/iharm').mkdir(parents=True)
    (folder/'checkpoint.pth').write_bytes(b'weights')
    files = [{'path': 'checkpoint.pth', 'size': 7,
              'sha256': digest(folder/'checkpoint.pth')}]
    (folder/'download-manifest.json').write_text(json.dumps({'model_id': 'pctnet', 'files': files}))
    assert verify_model(folder, 'pctnet')
    assert verify_model(folder, 'pctnet', hashes=False)
    (folder/'checkpoint.pth').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        verify_model(folder, 'pctnet')


def test_manifest_blocks_traversal_and_bad_shape(tmp_path):
    folder = model_directory(tmp_path, 'rainnet')
    folder.mkdir(parents=True)
    (folder/'download-manifest.json').write_text(json.dumps({'model_id': 'rainnet', 'files': [
        {'path': '../secret', 'size': 1, 'sha256': '0'*64}]}))
    with pytest.raises(ValueError, match='unsafe'):
        verify_model(folder, 'rainnet')
    (folder/'download-manifest.json').write_text(json.dumps({'model_id': 'rainnet', 'files': [None]}))
    with pytest.raises(ValueError, match='entry'):
        verify_model(folder, 'rainnet')


def test_diffusion_components_must_have_config_and_weights(tmp_path):
    folder = model_directory(tmp_path, 'sd15')
    (folder/'unet').mkdir(parents=True)
    (folder/'vae').mkdir()
    (folder/'text_encoder').mkdir()
    (folder/'unet/model.fp16.safetensors').write_bytes(b'x')
    (folder/'vae/model.fp16.safetensors').write_bytes(b'x')
    (folder/'text_encoder/model.fp16.safetensors').write_bytes(b'x')
    (folder/'model_index.json').write_text(json.dumps({'unet':['x','Y'], 'vae':['x','Y'], 'text_encoder':['x','Y']}))
    files = []
    for path in folder.rglob('*'):
        if path.is_file(): files.append({'path':path.relative_to(folder).as_posix(), 'size':path.stat().st_size, 'sha256':digest(path)})
    (folder/'download-manifest.json').write_text(json.dumps({'model_id':'sd15','files':files}))
    assert verify_model(folder, 'sd15')
    (folder/'vae/model.fp16.safetensors').unlink()
    with pytest.raises(ValueError, match='missing'):
        verify_model(folder, 'sd15', hashes=False)


def test_previous_cli_cache_layout_is_accepted_when_complete(tmp_path):
    import hashlib
    from scratchforge.model_files import verify_legacy_model
    from scripts.setup_harmonization import SOURCES
    pct=tmp_path/'pctnet'; (pct/'source/iharm').mkdir(parents=True)
    weight=pct/'checkpoint.pth'; weight.write_bytes(b'weights')
    (pct/'model-provenance.json').write_text(json.dumps({
        'model':'pctnet','source_repository':SOURCES['pctnet'][0],
        'source_revision':SOURCES['pctnet'][1],
        'checkpoint_sha256':hashlib.sha256(b'weights').hexdigest()}))
    assert verify_legacy_model(pct,'pctnet')
    weight.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):
        verify_legacy_model(pct,'pctnet')


def test_previous_diffusers_cache_checks_provenance_and_every_component(tmp_path):
    from scratchforge.model_catalog import MODEL_CATALOG
    from scratchforge.model_files import verify_legacy_model
    folder=tmp_path/'sd15'
    for component in ('unet','vae','text_encoder','text_encoder_2','tokenizer'):
        (folder/component).mkdir(parents=True)
        (folder/component/('weights.fp16.safetensors' if component in ('unet','vae','text_encoder','text_encoder_2') else 'config.json')).write_bytes(b'complete')
    (folder/'model_index.json').write_text(json.dumps({key:['module','Class'] for key in ('unet','vae','text_encoder','text_encoder_2','tokenizer')}))
    (folder/'model-provenance.json').write_text(json.dumps({'model_id':MODEL_CATALOG['sd15']['repo_id'],'resolved_revision':'a'*40}))
    assert verify_legacy_model(folder,'sd15')
    (folder/'vae/weights.fp16.safetensors').unlink()
    with pytest.raises(ValueError,match='empty'):
        verify_legacy_model(folder,'sd15')

