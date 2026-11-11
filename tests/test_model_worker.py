from io import BytesIO
import hashlib
import json
from pathlib import Path
import zipfile

import pytest
import scratchforge.model_worker as worker
from scratchforge.model_worker import Download


class Response:
    def __init__(self, data): self.data=BytesIO(data)
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self, size=-1): return self.data.read(size)


def test_fetch_stream_hashes_and_checks_git_blob(tmp_path, monkeypatch):
    data=b'official bytes'
    monkeypatch.setattr(worker,'urlopen',lambda *a,**k:Response(data))
    path=tmp_path/'payload/w.pth'
    blob=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
    Download(tmp_path).fetch('https://example.test/w',path,len(data),git_blob=blob)
    assert path.read_bytes()==data and not path.with_suffix('.pth.part').exists()


def test_fetch_rejects_size_sha_and_git_blob(tmp_path, monkeypatch):
    monkeypatch.setattr(worker,'urlopen',lambda *a,**k:Response(b'bad'))
    downloader=Download(tmp_path)
    with pytest.raises(ValueError,match='Incomplete'):
        downloader.fetch('https://example.test/w',tmp_path/'a',4)
    with pytest.raises(ValueError,match='checksum'):
        downloader.fetch('https://example.test/w',tmp_path/'b',3,'0'*64)
    with pytest.raises(ValueError,match='Git checksum'):
        downloader.fetch('https://example.test/w',tmp_path/'c',3,git_blob='0'*40)


def test_progress_retries_windows_sharing_violation(tmp_path, monkeypatch):
    original = Path.replace
    attempts = []
    def replace(path, target):
        if Path(target).name == 'progress.json':
            attempts.append(target)
            if len(attempts) < 3:
                raise PermissionError(5, 'Access is denied')
        return original(path, target)
    monkeypatch.setattr(Path, 'replace', replace)
    download = Download(tmp_path)
    download.progress()
    assert len(attempts) == 3
    assert json.loads((tmp_path/'progress.json').read_text())['status'] == 'downloading'


def test_progress_lock_never_aborts_model_worker(tmp_path, monkeypatch):
    def locked(path, target):
        if Path(target).name == 'progress.json':
            raise PermissionError(5, 'Access is denied')
        return path
    monkeypatch.setattr(Path, 'replace', locked)
    download = Download(tmp_path)
    download.progress('verifying')
    assert (tmp_path/'progress.tmp').is_file()


def test_diffusion_pins_revision_and_checks_safe_paths(tmp_path, monkeypatch):
    data=b'model config'
    api={'sha':'a'*40,'siblings':[{'rfilename':'unet/config.json','size':len(data), 'blobId':'b'*40},
                                  {'rfilename':'../../outside.txt','size':1,'blobId':'c'*40},
                                  {'rfilename':'weights.bin','size':1,'blobId':'d'*40}]}
    monkeypatch.setattr(worker,'urlopen',lambda *a,**k:Response(json.dumps(api).encode()))
    downloader=Download(tmp_path)
    fetched=[]
    def fetch(url,path,size,checksum=None,blob=None):
        fetched.append((url,path,size,blob)); path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
    downloader.fetch=fetch
    info={'repo_id':'example/model'}
    with pytest.raises(ValueError,match='Unsafe'):
        downloader.diffusion('sd15',info)
    api['siblings']=api['siblings'][:1]
    downloader.diffusion('sd15',info)
    assert '/resolve/' + 'a'*40 + '/unet/config.json' in fetched[0][0]
    assert (tmp_path/'payload/model-provenance.json').is_file()


def test_harmonization_run_extracts_pinned_source_and_writes_manifest(tmp_path,monkeypatch):
    archive=BytesIO()
    with zipfile.ZipFile(archive,'w') as zf:
        zf.writestr('repo-commit/iharm/model.py','pinned source')
    info={'checkpoint_url':'https://example.test/cp','size_bytes':4,'checkpoint_sha256':'c'*64,
          'source_repository':'https://github.com/example/repo.git','source_revision':'d'*40}
    monkeypatch.setitem(worker.MODEL_CATALOG,'pctnet',info)
    monkeypatch.setattr(worker,'verify_model',lambda *a,**k:True)
    download=Download(tmp_path)
    def fetch(url,path,*args,**kwargs):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(b'cp!!' if path.name=='checkpoint.pth' else archive.getvalue())
    download.fetch=fetch
    download.run('pctnet')
    assert (tmp_path/'payload/source/iharm/model.py').read_text()=='pinned source'
    manifest=json.loads((tmp_path/'payload/download-manifest.json').read_text())
    assert manifest['model_id']=='pctnet' and len(manifest['files'])==3


def test_worker_main_records_failure_status(tmp_path,monkeypatch):
    monkeypatch.setattr(Download,'run',lambda *a:(_ for _ in ()).throw(OSError('network broke')))
    assert worker.main(['sdxl',str(tmp_path)])==1
    status=json.loads((tmp_path/'progress.json').read_text())
    assert status['status']=='error' and 'network broke' in status['error']
