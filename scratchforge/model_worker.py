"""Killable model download worker. Every network byte stays inside staging."""
import fnmatch
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.parse import quote
from urllib.request import Request, urlopen
import zipfile

from scratchforge.model_catalog import MODEL_CATALOG
from scratchforge.model_files import digest, verify_model


class Download:
    def __init__(self, stage):
        self.stage = Path(stage)
        self.payload = self.stage/'payload'
        self.payload.mkdir(parents=True, exist_ok=True)
        self.done, self.total, self.started = 0, 0, time.monotonic()

    def progress(self, status='downloading', error=None):
        state = {'status': status, 'downloaded_bytes': self.done, 'total_bytes': self.total or None,
                 'progress': min(99, self.done/self.total*100) if self.total else 0,
                 'speed_bps': self.done/max(.01, time.monotonic()-self.started), 'error': error}
        temporary = self.stage/'progress.tmp'
        temporary.write_text(json.dumps(state))
        destination = self.stage/'progress.json'
        for attempt in range(5):
            try:
                temporary.replace(destination)
                return
            except PermissionError:
                # The API may briefly have the previous JSON open while polling.
                # Windows refuses replacement during that read; don't kill a large
                # model transfer over a transient progress-file lock.
                if attempt < 4:
                    time.sleep(.025 * (attempt + 1))

    def fetch(self, url, path, expected_size=None, checksum=None, git_blob=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name+'.part')
        sha256 = hashlib.sha256()
        sha1 = hashlib.sha1(f'blob {expected_size}\0'.encode()) if git_blob else None
        count = 0
        with urlopen(Request(url, headers={'User-Agent': 'ScratchForge/1.0'}), timeout=30) as response:
            with partial.open('wb') as stream:
                while chunk := response.read(256*1024):
                    stream.write(chunk)
                    sha256.update(chunk)
                    if sha1: sha1.update(chunk)
                    count += len(chunk); self.done += len(chunk)
                    self.progress()
        if expected_size is not None and count != expected_size:
            raise ValueError(f'Incomplete model download: {path.name}')
        if checksum and sha256.hexdigest() != checksum:
            raise ValueError(f'Model checksum mismatch: {path.name}')
        if git_blob and sha1.hexdigest() != git_blob:
            raise ValueError(f'Model Git checksum mismatch: {path.name}')
        partial.replace(path)

    def diffusion(self, model_id, info):
        repo = info['repo_id']
        with urlopen(f'https://huggingface.co/api/models/{repo}?blobs=true', timeout=30) as response:
            metadata = json.load(response)
        revision = metadata['sha']
        patterns = ('*.json', '*.txt', '*.model', '*.fp16.safetensors')
        files = [item for item in metadata['siblings'] if any(fnmatch.fnmatch(item['rfilename'], pattern) for pattern in patterns)]
        self.total = sum(item['size'] for item in files)
        for item in files:
            name = item['rfilename']
            target = (self.payload/name).resolve()
            if not target.is_relative_to(self.payload.resolve()): raise ValueError('Unsafe Hub filename')
            lfs = item.get('lfs') or {}
            self.fetch(f'https://huggingface.co/{repo}/resolve/{revision}/{quote(name)}', target,
                       item['size'], lfs.get('sha256'), None if lfs else item['blobId'])
        (self.payload/'model-provenance.json').write_text(json.dumps({'model_id': repo, 'resolved_revision': revision}))

    def harmonization(self, model_id, info):
        self.total = info['size_bytes']
        self.fetch(info['checkpoint_url'], self.payload/'checkpoint.pth', info['size_bytes'], info['checkpoint_sha256'])
        repo = info['source_repository'].removesuffix('.git').replace('https://github.com/', '')
        archive = self.stage/'source.zip'
        self.total = 0  # Source archive size is not known until received.
        self.fetch(f'https://codeload.github.com/{repo}/zip/{info["source_revision"]}', archive)
        with zipfile.ZipFile(archive) as package:
            for member in package.infolist():
                parts = Path(member.filename).parts[1:]
                if not parts or member.is_dir(): continue
                destination = (self.payload/'source'/Path(*parts)).resolve()
                if not destination.is_relative_to((self.payload/'source').resolve()): raise ValueError('Unsafe source archive path')
                destination.parent.mkdir(parents=True, exist_ok=True)
                with package.open(member) as source, destination.open('wb') as output:
                    import shutil
                    shutil.copyfileobj(source, output)
        metadata = {key: info[key] for key in ('source_repository', 'source_revision', 'checkpoint_sha256')}
        (self.payload/'model-provenance.json').write_text(json.dumps(metadata))

    def run(self, model_id):
        info = MODEL_CATALOG[model_id]
        if 'repo_id' in info: self.diffusion(model_id, info)
        else: self.harmonization(model_id, info)
        self.progress('verifying')
        files = [{'path': str(path.relative_to(self.payload)).replace('\\', '/'),
                  'size': path.stat().st_size, 'sha256': digest(path)}
                 for path in self.payload.rglob('*') if path.is_file()]
        (self.payload/'download-manifest.json').write_text(json.dumps({'model_id': model_id, 'files': files}))
        verify_model(self.payload, model_id)
        self.progress('verifying')


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    worker = Download(arguments[1])
    try: worker.run(arguments[0])
    except Exception as error:
        worker.progress('error', str(error))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
