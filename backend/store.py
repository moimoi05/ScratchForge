"""Filesystem repository with atomic metadata updates and opaque public IDs."""
import json
import re
import threading
import uuid
from pathlib import Path

from backend.images import save_tiff
from scratchforge.dataset import sha


def read_json(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.out = self.root / 'outputs'
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    def asset(self, path):
        resolved = (self.root / path).resolve()
        if not resolved.is_relative_to(self.root) or not resolved.is_file():
            raise ValueError('Invalid dataset asset')
        return resolved

    def index(self):
        return read_json(self.root/'data/fullframe/index.json', [])

    def library(self):
        from backend.library import merge_library
        from backend.segment import active_library

        with self.lock:
            return merge_library(self, active_library(self, read_json(self.root/'data/library/index.json', [])))

    def verify_train_source(self, source, status, *, historical=False):
        rows = self.index()
        if historical:
            rows = [*rows, *read_json(self.root/'data/fullframe/legacy-index.json', [])]
        keys = ('sha256', 'source_image', 'source_image_id')
        for row in rows:
            if (row['id'] == source.get('id') and row.get('split') == 'train' and row.get('status') == status
                    and source.get('sha256') and all(row.get(key) == source.get(key) for key in keys)):
                return row
        raise ValueError('Training source no longer matches the verified train manifest')

    def load(self, kind, identifier):
        if not re.fullmatch('[a-f0-9]{32}', identifier):
            raise ValueError('Invalid identifier')
        with self.lock:
            value = read_json(self.out/f'{identifier}-{kind}.json')
        if value is None:
            raise FileNotFoundError('Record not found')
        return value

    def save(self, kind, identifier, value):
        with self.lock:
            write_json(self.out/f'{identifier}-{kind}.json', value)

    def register(self, image, source=None):
        identifier = uuid.uuid4().hex
        native = save_tiff(image, self.out/f'{identifier}-input.tif')
        record = {'id': identifier, 'width': image.width, 'height': image.height,
                  'url': f'/files/{identifier}-input.png', 'source': source,
                  'input_format': 'tiff', 'input_sha256': sha(native)}
        try:
            self.save('input', identifier, record)
        except Exception:
            native.unlink(missing_ok=True)
            raise
        return {key: value for key, value in record.items() if key not in ('source', 'input_sha256')}

    def input_image(self, record):
        suffix = '.tif' if record.get('input_format') == 'tiff' else '.png'
        return self.out/f"{record['id']}-input{suffix}"

    def result_image(self, record):
        suffix = '.tif' if record.get('image_format') == 'tiff' else '.png'
        return self.out/f"{record['id']}{suffix}"
