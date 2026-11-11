"""Model cache verification shared by download manager and worker."""
import hashlib
import json
from pathlib import Path
import re


def model_directory(root, model_id):
    base = Path(root)/'data/models'
    return base/f'{model_id}-inpaint' if model_id in {'sdxl', 'sd15'} else base/'harmonization'/model_id


def digest(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_model(folder, model_id, *, hashes=True):
    folder = Path(folder)
    manifest_path = folder/'download-manifest.json'
    if not manifest_path.exists():
        return verify_legacy_model(folder, model_id)
    manifest = json.loads(manifest_path.read_text())
    if (not isinstance(manifest, dict) or manifest.get('model_id') != model_id
            or not isinstance(manifest.get('files'), list) or not manifest['files']):
        raise ValueError('Model manifest missing or incorrect')
    for item in manifest['files']:
        if (not isinstance(item, dict) or not isinstance(item.get('path'), str)
                or not isinstance(item.get('size'), int) or item['size'] < 0
                or not isinstance(item.get('sha256'), str)
                or not re.fullmatch(r'[0-9a-f]{64}', item['sha256'])):
            raise ValueError('Model manifest entry is invalid')
        path = (folder/item['path']).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file():
            raise ValueError('Model file missing or unsafe')
        if path.stat().st_size != item['size'] or (hashes and digest(path) != item['sha256']):
            raise ValueError('Model file checksum mismatch')
    if model_id in {'sdxl', 'sd15'}:
        config = json.loads((folder/'model_index.json').read_text())
        if not isinstance(config, dict):
            raise ValueError('Model configuration is invalid')
        for component, value in config.items():
            if component.startswith('_') or not isinstance(value, list) or not value or value[0] is None: continue
            directory = folder/component
            if not directory.is_dir() or not any(directory.iterdir()):
                raise ValueError(f'Model component missing: {component}')
        for component in ('unet', 'vae', 'text_encoder'):
            if not list((folder/component).glob('*.safetensors')): raise ValueError(f'Weights missing: {component}')
    else:
        if not (folder/'checkpoint.pth').is_file() or not (folder/'source').is_dir():
            raise ValueError('Harmonization source or checkpoint missing')
    return True


def verify_legacy_model(folder, model_id):
    """Accept complete caches created by the earlier CLI setup tools."""
    import json
    folder = Path(folder)
    provenance_path = folder/'model-provenance.json'
    if not provenance_path.is_file():
        raise ValueError('Model provenance missing')
    provenance = json.loads(provenance_path.read_text())
    if model_id in {'sdxl', 'sd15'}:
        from scratchforge.model_catalog import MODEL_CATALOG
        if provenance.get('model_id') != MODEL_CATALOG[model_id]['repo_id']:
            raise ValueError('Model provenance does not match model ID')
        revision = provenance.get('resolved_revision')
        if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise ValueError('Model revision is not pinned')
        config = json.loads((folder/'model_index.json').read_text())
        if not isinstance(config, dict): raise ValueError('Model configuration is invalid')
        for component, value in config.items():
            if component.startswith('_') or not isinstance(value, list) or not value or value[0] is None: continue
            directory = (folder/component).resolve()
            if not directory.is_relative_to(folder.resolve()) or not directory.is_dir():
                raise ValueError(f'Model component missing: {component}')
            if not any(path.is_file() and path.stat().st_size for path in directory.rglob('*')):
                raise ValueError(f'Model component empty: {component}')
        for component in ('unet', 'vae', 'text_encoder'):
            if not any(path.is_file() and path.stat().st_size for path in (folder/component).glob('*.fp16.safetensors')):
                raise ValueError(f'Model weights missing: {component}')
    else:
        from scripts.setup_harmonization import SOURCES
        repository, revision = SOURCES[model_id]
        checkpoint = folder/'checkpoint.pth'
        if (provenance.get('model') != model_id or provenance.get('source_repository') != repository
                or provenance.get('source_revision') != revision or not checkpoint.is_file()
                or not (folder/'source').is_dir()):
            raise ValueError('Harmonization model source or checkpoint is incomplete')
        if digest(checkpoint) != provenance.get('checkpoint_sha256'):
            raise ValueError('Harmonization checkpoint checksum mismatch')
        prefix = {'pctnet': 'iharm', 'harmonizer': 'src', 'rainnet': 'models'}[model_id]
        if not (folder/'source'/prefix).is_dir():
            raise ValueError('Harmonization source code missing')
    return True
