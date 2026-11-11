"""Local, checksum-verified F971 LoRA registry. Never resolves remote weights."""
import json
import os
import re
from pathlib import Path

from scratchforge.dataset import sha


def model_info(root, base_model):
    if base_model not in {'sd15', 'sdxl'}:
        raise ValueError('Unknown LoRA base model')
    variable = 'SCRATCHFORGE_SD15_MODEL_DIR' if base_model == 'sd15' else 'SCRATCHFORGE_MODEL_DIR'
    directory = Path(os.environ.get(variable, Path(root) / 'data/models' / f'{base_model}-inpaint')).resolve()
    if not (directory / 'model_index.json').is_file():
        raise RuntimeError(f'{base_model} inpainting model missing; download the base model first')
    path = directory / 'model-provenance.json'
    provenance = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    revision = provenance.get('resolved_revision') or provenance.get('revision') or 'local-unversioned'
    return directory, revision


class LoraRegistry:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.directory = self.root / 'data/models/lora'

    def list(self):
        index = self.directory / 'index.json'
        rows = json.loads(index.read_text(encoding='utf-8')) if index.exists() else []
        if not isinstance(rows, list):
            raise ValueError('Invalid LoRA registry')
        return [row for row in rows if row.get('status') == 'succeeded']

    def resolve(self, identifier, base_model, revision):
        if not isinstance(identifier, str) or not re.fullmatch('[a-f0-9]{32}', identifier):
            raise ValueError('Invalid LoRA identifier')
        row = next((item for item in self.list() if item.get('id') == identifier), None)
        if row is None:
            raise ValueError('LoRA checkpoint was not found or is incomplete')
        if row.get('base_model') != base_model:
            raise ValueError('LoRA base model does not match the selected inpainting model')
        if row.get('base_revision') != revision:
            raise ValueError('LoRA base revision does not match the installed inpainting model')
        path = (self.directory / row.get('weights', '')).resolve()
        expected = (self.directory / identifier).resolve()
        if not path.is_relative_to(expected) or path.suffix != '.safetensors' or not path.is_file():
            raise ValueError('Unsafe or missing LoRA weights')
        if not row.get('sha256') or sha(path) != row['sha256']:
            raise ValueError('LoRA weights checksum mismatch')
        return dict(row), path

    def verify_sources(self, row):
        from backend.library import Library
        from backend.store import Store

        sources = row.get('sources')
        if not isinstance(sources, list) or not sources:
            raise ValueError('LoRA training provenance is missing')
        library = Library(Store(self.root))
        for stamp in sources:
            if str(stamp.get('review_status', '')).upper() != 'APPROVED':
                raise ValueError('LoRA source was not approved')
            library.verify_stamp(stamp, historical=True)

    def register(self, row):
        from backend.store import write_json

        if row.get('status') != 'succeeded':
            raise ValueError('Only a completed LoRA checkpoint can be registered')
        rows = self.list()
        if any(item['id'] == row['id'] for item in rows):
            raise ValueError('LoRA checkpoint already registered')
        directory = self.directory / row['id']
        path = (self.directory / row['weights']).resolve()
        if (not re.fullmatch('[a-f0-9]{32}', row['id']) or not path.is_relative_to(directory.resolve())
                or not path.is_file() or path.suffix != '.safetensors' or sha(path) != row['sha256']):
            raise ValueError('Invalid completed LoRA checkpoint')
        write_json(directory / 'manifest.json', row)
        write_json(self.directory / 'index.json', [*rows, dict(row)])
        return dict(row)
