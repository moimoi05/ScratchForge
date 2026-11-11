"""Background downloads isolated in killable processes; publish verified models only."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import RLock

from scratchforge.model_files import model_directory, verify_model

IDS = ('pctnet', 'harmonizer', 'rainnet', 'sdxl', 'sd15')
SIZES = {'pctnet': '~30 MB', 'harmonizer': '~25 MB', 'rainnet': '~210 MB', 'sdxl': '~7 GB', 'sd15': '~2.5 GB'}


class ModelDownloads:
    def __init__(self, root, launcher=subprocess.Popen):
        self.root = Path(root).resolve()
        self.launcher = launcher
        self.tasks = {}
        self.lock = RLock()
        self.cached = {}

    def _check(self, model_id):
        if model_id not in IDS:
            raise ValueError('Unknown model_id')

    def _stage(self, model_id):
        return self.root/'data/models/.downloads'/model_id

    def _row(self, model_id):
        row = {'model_id': model_id, 'status': 'not_downloaded', 'downloaded_bytes': 0,
               'total_bytes': None, 'progress': 0, 'speed_bps': 0, 'error': None, 'size_label': SIZES[model_id]}
        task = self.tasks.get(model_id)
        if task:
            row = {**row, **task['state']}
            progress = self._stage(model_id)/'progress.json'
            if progress.exists():
                try: row = {**row, **json.loads(progress.read_text())}
                except (OSError, ValueError): pass
            code = task['process'].poll()
            if code is None:
                return row
            if task['state']['status'] not in {'cancelled', 'error', 'ready'}:
                if code == 0:
                    try:
                        payload = self._stage(model_id)/'payload'
                        # The worker has already hashed the staged files. Check the
                        # manifest and structure here without re-reading gigabytes;
                        # this keeps status and cancellation responsive.
                        verify_model(payload, model_id, hashes=False)
                        destination = model_directory(self.root, model_id)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if destination.exists():
                            backup = destination.with_name(destination.name+'.incomplete-backup')
                            if backup.exists(): raise ValueError('Previous incomplete backup exists; inspect model directory')
                            destination.rename(backup)
                            try: payload.rename(destination)
                            except Exception:
                                backup.rename(destination)
                                raise
                        else: payload.rename(destination)
                        self.cached[model_id] = self._signature(destination)
                        row = {**row, 'status': 'ready', 'progress': 100, 'error': None}
                    except Exception as error:
                        row = {**row, 'status': 'error', 'error': str(error)}
                else:
                    row = {**row, 'status': 'error', 'error': row.get('error') or 'Download worker stopped unexpectedly'}
                task['state'] = row
                shutil.rmtree(self._stage(model_id), ignore_errors=True)
                if task['state']['status'] == 'ready':
                    self.tasks.pop(model_id, None)
                    return row
            return {**row, **task['state']}
        folder = model_directory(self.root, model_id)
        try:
            signature = self._signature(folder)
            if self.cached.get(model_id) != signature:
                verify_model(folder, model_id)
                self.cached[model_id] = signature
            return {**row, 'status': 'ready', 'progress': 100}
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return row

    @staticmethod
    def _signature(folder):
        return tuple((str(p.relative_to(folder)), p.stat().st_size, p.stat().st_mtime_ns)
                     for p in folder.rglob('*') if p.is_file() and '.git' not in p.parts)

    def status(self):
        with self.lock: return [self._row(key) for key in IDS]

    def start(self, model_id):
        self._check(model_id)
        with self.lock:
            row = self._row(model_id)
            if row['status'] in {'ready', 'downloading', 'verifying'}: return row
            stage = self._stage(model_id)
            shutil.rmtree(stage, ignore_errors=True)
            stage.mkdir(parents=True)
            process = self.launcher([sys.executable, '-m', 'scratchforge.model_worker', model_id, str(stage)],
                                    cwd=str(Path(__file__).resolve().parents[1]),
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            state = {**row, 'status': 'downloading', 'error': None, 'progress': 0}
            self.tasks[model_id] = {'process': process, 'state': state}
            return state

    download = start

    def cancel(self, model_id):
        self._check(model_id)
        with self.lock:
            task = self.tasks.get(model_id)
            if not task or task['process'].poll() is not None: return self._row(model_id)
            process = task['process']
            if os.name == 'nt' and hasattr(process, 'pid'):
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
            else: process.terminate()
            process.wait(timeout=10)
            shutil.rmtree(self._stage(model_id), ignore_errors=True)
            task['state'] = {**task['state'], 'status': 'cancelled', 'progress': 0, 'downloaded_bytes': 0}
            return task['state']

    def close(self):
        for key in tuple(self.tasks): self.cancel(key)
