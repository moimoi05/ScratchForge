"""Local LoRA lifecycle and synthetic stamps, called inside the shared GPU queue."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Literal
import uuid

import numpy as np
from PIL import Image
from pydantic import Field

from backend.models import StrictModel
from backend.store import read_json, write_json
from scratchforge.dataset import sha
from scratchforge.lora import LoraRegistry, model_info


class TrainLoraRequest(StrictModel):
    stamp_ids: list[str] = Field(min_length=1, max_length=1000)
    base_model: Literal['sd15', 'sdxl'] = 'sd15'
    steps: int = Field(default=1000, ge=1, le=100000)
    seed: int = Field(default=42, ge=0, le=2**32-1)


class GenerateStampsRequest(StrictModel):
    stamp_ids: list[str] = Field(min_length=1, max_length=100)
    base_model: Literal['sd15', 'sdxl'] = 'sd15'
    lora_id: str = Field(pattern='^[a-f0-9]{32}$')
    count: int = Field(default=8, ge=1, le=100)
    seed: int = Field(default=42, ge=0, le=2**32-1)
    strength: float = Field(default=.65, ge=.1, le=.99)
    threshold: int = Field(default=8, ge=1, le=255)
    min_area: int = Field(default=3, ge=1, le=10000)


def _synthetic_ancestor(stamp, library, seen=None):
    seen = frozenset() if seen is None else seen
    if stamp.get('id') in seen or len(seen) > 64:
        raise ValueError('Cyclic or excessively deep stamp lineage')
    if stamp.get('origin') in {'synthetic', 'generated', 'ai'} or stamp.get('provenance', {}).get('kind') == 'synthetic':
        return True
    parent = stamp.get('parent_stamp')
    if parent is None and stamp.get('parent_stamp_id'):
        parent = library.lookup(stamp['parent_stamp_id'])
    return bool(parent and _synthetic_ancestor(parent, library, seen | {stamp.get('id')}))


class Training:
    def __init__(self, store):
        self.store = store
        self.registry = LoraRegistry(store.root)

    def checkpoints(self):
        return self.registry.list()

    def _snapshot(self, identifiers):
        from backend.library import Library

        library = Library(self.store)
        rows = {row['id']: row for row in self.store.library()}
        selected = []
        for identifier in dict.fromkeys(identifiers):
            stamp = rows.get(identifier)
            if stamp is None or stamp.get('split') != 'train':
                raise ValueError('LoRA requires registered train stamps')
            if str(stamp.get('review_status', '')).upper() != 'APPROVED':
                raise ValueError('LoRA requires approved stamp images and polygons')
            if _synthetic_ancestor(stamp, library):
                raise ValueError('LoRA training excludes synthetic stamps and their descendants')
            library.verify_stamp(stamp)
            if Path(stamp.get('rgb', '')).suffix.lower() not in {'.tif', '.tiff'}:
                raise ValueError('LoRA requires original TIFF stamp bundles; rebuild legacy stamps first')
            selected.append(dict(stamp))
        return selected

    def train(self, payload, progress, cancelled):
        if cancelled():
            raise RuntimeError('LoRA training cancelled')
        request = TrainLoraRequest(**payload).model_dump()
        try:
            import torch
        except ImportError as error:
            raise RuntimeError('CUDA PyTorch missing; install requirements-gpu.txt on the target machine') from error
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU unavailable; LoRA training requires the NVIDIA target machine')
        model, revision = model_info(self.store.root, request['base_model'])
        sources = self._snapshot(request['stamp_ids'])
        identifier = uuid.uuid4().hex
        stage = self.registry.directory / f'.{identifier}.staging'
        stage.mkdir(parents=True)
        try:
            write_json(stage/'request.json', {**request, 'sources': sources,
                       'model_directory': str(model), 'library_directory': str(self.store.root/'data/library'),
                       'resolution': 512 if request['base_model'] == 'sd15' else 1024})
            metrics = self._run_worker(stage, progress, cancelled)
            if cancelled():
                raise RuntimeError('LoRA training cancelled')
            # Re-check original snapshots, not current replacements, before publishing weights.
            from backend.library import Library
            for source in sources:
                Library(self.store).verify_stamp(source, historical=True)
            weights = stage/'pytorch_lora_weights.safetensors'
            self._validate_weights(weights)
            row = {'id': identifier, 'base_model': request['base_model'], 'base_revision': revision,
                   'weights': f'{identifier}/{weights.name}', 'sha256': sha(weights),
                   'status': 'succeeded', 'steps': request['steps'], 'seed': request['seed'],
                   'sources': sources, 'metrics': metrics, 'created_at': time.time()}
            with self.store.lock:
                if cancelled():
                    raise RuntimeError('LoRA training cancelled')
                stage.rename(self.registry.directory/identifier)
                try:
                    self.registry.register(row)
                except Exception:
                    shutil.rmtree(self.registry.directory/identifier)
                    raise
                progress(100, phase='completed', step=request['steps'], total=request['steps'])
            return {'lora_id': identifier, 'checkpoint': row}
        finally:
            if (stage/'worker.log').exists():
                shutil.copyfile(stage/'worker.log', self.store.out/f'{identifier}-training.log')
            if stage.exists():
                shutil.rmtree(stage)

    @staticmethod
    def _validate_weights(path):
        from safetensors import safe_open

        if not path.is_file():
            raise RuntimeError('Training ended without LoRA weights')
        with safe_open(str(path), framework='np') as weights:
            keys = list(weights.keys())
            if not keys or any(not key.startswith('unet.') or 'lora' not in key.lower() for key in keys):
                raise ValueError('Training output is not UNet LoRA weights')
            if any(not np.isfinite(weights.get_tensor(key)).all() for key in keys):
                raise ValueError('Training output contains non-finite LoRA weights')

    @staticmethod
    def _run_worker(stage, progress, cancelled):
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        with (stage/'worker.log').open('wb') as log:
            process = subprocess.Popen([sys.executable, '-m', 'scratchforge.lora_worker', str(stage/'request.json')],
                                       cwd=Path(__file__).resolve().parents[1], stdout=log, stderr=log,
                                       creationflags=flags)
            previous = None
            try:
                while process.poll() is None:
                    if cancelled():
                        write_json(stage/'cancel', {'cancelled': True})
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill(); process.wait(timeout=5)
                        raise RuntimeError('LoRA training cancelled')
                    status = read_json(stage/'progress.json', {})
                    if status and status != previous:
                        progress(status['progress'], **{key: value for key, value in status.items() if key != 'progress'})
                        previous = status
                    time.sleep(.1)
            finally:
                if process.poll() is None:
                    process.kill(); process.wait(timeout=5)
        result = read_json(stage/'completed.json', {})
        if process.returncode != 0 or result.get('status') != 'succeeded':
            raise RuntimeError(result.get('error', 'LoRA worker stopped unexpectedly; see the local worker log'))
        return result['metrics']

    def generate(self, payload, adapter, progress, cancelled):
        from backend.library import Library
        from scratchforge.labels import difference_mask
        from scratchforge.stamps import write_bundle

        request = GenerateStampsRequest(**payload).model_dump()
        if cancelled():
            raise RuntimeError('Stamp generation cancelled')
        _, revision = model_info(self.store.root, request['base_model'])
        checkpoint, _ = self.registry.resolve(request['lora_id'], request['base_model'], revision)
        library = Library(self.store)
        templates = self._snapshot(request['stamp_ids'])
        for source in checkpoint['sources']:
            library.verify_stamp(source, historical=True)
        adapter.set_lora(request['lora_id'])
        random = np.random.default_rng(request['seed'])
        records, rejected = [], []
        published = False
        directory = self.store.root/'data/library'
        try:
            for index in range(request['count']):
                if cancelled():
                    raise RuntimeError('Stamp generation cancelled')
                template = templates[int(random.integers(len(templates)))]
                seed = (request['seed']+index) % 2**32
                background, guidance = _synthetic_input(directory, template, seed)
                generated = adapter(background, guidance, seed, request['strength'])
                if generated.size != background.size:
                    raise ValueError('Stamp AI returned incorrect image dimensions')
                mask = difference_mask(background, generated, guidance, request['threshold'], request['min_area'])
                if not mask.getbbox():
                    rejected.append({'index': index, 'seed': seed, 'reason': 'No scratch detected; adjust threshold/strength'})
                else:
                    identifier = uuid.uuid4().hex
                    rgb = Image.composite(generated.convert('RGB'), background, guidance)
                    metadata = {'origin': 'synthetic', 'split': 'train', 'review_status': 'PENDING_VISUAL_REVIEW',
                                'parent_stamp_id': template['id'], 'parent_stamp': template,
                                'provenance': {'kind': 'synthetic', 'base_model': request['base_model'],
                                               'base_revision': revision, 'lora_id': checkpoint['id'],
                                               'lora_sha256': checkpoint['sha256'], 'training_stamps': checkpoint['sources'],
                                               'seed': seed, 'strength': request['strength'],
                                               'threshold': request['threshold'], 'min_area': request['min_area']}}
                    records.append(write_bundle(directory, identifier, rgb, mask, metadata=metadata))
                progress(99*(index+1)/request['count'], phase='generating', completed=index+1, total=request['count'])
            if cancelled():
                raise RuntimeError('Stamp generation cancelled')
            if not records:
                raise ValueError('No valid synthetic stamps; lower the segmentation threshold or change guidance')
            with self.store.lock:
                if cancelled():
                    raise RuntimeError('Stamp generation cancelled')
                library.publish(records)
                published = True
                progress(100, phase='completed', completed=request['count'], total=request['count'])
            return {'stamps': records, 'rejected': rejected}
        except Exception:
            if not published:
                for record in records:
                    shutil.rmtree(directory/record['id'])
            raise


def _synthetic_input(directory, template, seed):
    random = np.random.default_rng(seed)
    with Image.open(directory/template['mask']) as source:
        guidance = source.convert('L').rotate(float(random.uniform(-20, 20)), resample=Image.Resampling.NEAREST, expand=True)
    # Synthetic clean material: no held-out image is sampled or copied.
    width, height = guidance.size
    noise = random.normal(0, 1.4, (height, width, 1))
    lines = np.sin(np.arange(width)[None, :, None]/7)*.7
    pixels = np.clip(np.array([110, 112, 115])+noise+lines, 0, 255).astype('uint8')
    return Image.fromarray(pixels), guidance
