"""Serialized jobs and reviewed training export lifecycle."""
import logging
import random
import threading
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
from PIL import Image

from backend.images import save_image, save_tiff
from backend.store import read_json, write_json
from scratchforge.ai import NEGATIVE, PROMPT, SD15Generator, SDXLGenerator
from scratchforge.dataset import mask_for, sha, valid_polygons
from scratchforge.generation import brush_method, generate_scene

LOG = logging.getLogger(__name__)


class Jobs:
    def __init__(self, store, infer=None, harmonize=None):
        self.store = store
        self.ai = SDXLGenerator(store.root)
        self.sd15 = SD15Generator(store.root)
        self.infer = infer
        self.harmonize = harmonize
        self._training = None
        self._cancel = {}
        self._closed = False
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='scratchforge-gpu')
        for path in store.out.glob('*-job.json'):
            job = read_json(path)
            if job.get('status') in ('queued', 'running', 'cancelling'):
                write_json(path, {**job, 'status': 'failed', 'error': 'Application stopped before this job completed. Please retry.'})

    def submit(self, scene):
        self.store.load('input', scene['image_id'])
        stamps = {stamp['id']: stamp for stamp in self.store.library()}
        referenced = {item['stamp_id'] for item in [*scene['placements'], *scene['brushes']] if item.get('stamp_id')}
        missing = referenced - stamps.keys()
        if missing:
            from backend.library import registered_records
            stamps.update({stamp['id']: stamp for stamp in registered_records(self.store) if stamp['id'] in missing})
        scene = self.resolve_brush_stamps(scene, stamps)
        selected = {}
        for placement in [*scene.get('placements', []), *(
                scene.get('brushes', []) if brush_method(scene) in {'stamp_brush', 'stamp_brush_v2'} else [])]:
            if placement['stamp_id'] not in stamps or stamps[placement['stamp_id']].get('split') != 'train':
                raise ValueError('Unknown or non-training stamp')
            stamp = stamps[placement['stamp_id']]
            self.verify_stamp_source(stamp)
            selected[stamp['id']] = stamp
        return self._enqueue({'kind': 'scene', 'scene': scene, 'stamps': list(selected.values())})

    @property
    def training(self):
        if self._training is None:
            from backend.training import Training
            self._training = Training(self.store)
        return self._training

    @training.setter
    def training(self, value):
        self._training = value

    def resolve_brush_stamps(self, scene, stamps):
        if brush_method(scene) not in {'stamp_brush', 'stamp_brush_v2'} or not scene.get('brushes'):
            return scene
        eligible = sorted(identifier for identifier, stamp in stamps.items()
                          if stamp.get('split') == 'train' and Path(stamp.get('rgb', '')).suffix.lower() in {'.tif', '.tiff'})
        rng = random.Random(scene.get('seed', 0)); brushes = []
        assigned = {brush['stamp_id'] for brush in scene['brushes'] if brush.get('stamp_id')}
        available = [identifier for identifier in eligible if identifier not in assigned]
        rng.shuffle(available)
        for brush in scene['brushes']:
            identifier = brush.get('stamp_id')
            if not identifier and scene.get('brush_stamp_mode', 'selected') == 'random':
                if not eligible:
                    raise ValueError('No train TIFF stamps available for Stamp Brush')
                if not available:
                    available = eligible.copy()
                    rng.shuffle(available)
                identifier = available.pop()
            identifier = identifier or scene.get('brush_stamp_id')
            if identifier not in eligible:
                raise ValueError('Select a train TIFF stamp for Stamp Brush')
            brushes.append({**brush, 'stamp_id': identifier})
        return {**scene, 'brushes': brushes}

    def submit_operation(self, kind, payload):
        if kind not in {'train', 'augment', 'generate_stamps'}:
            raise ValueError('Unknown operation')
        return self._enqueue({'kind': kind, 'payload': payload})

    def _enqueue(self, fields):
        with self.store.lock:
            if self._closed:
                raise ValueError('Job service is shutting down')
            queued = sum(read_json(path).get('status') in ('queued', 'running', 'cancelling')
                         for path in self.store.out.glob('*-job.json'))
            if queued >= 8:
                raise ValueError('Job queue is full. Wait for a running job to finish.')
            identifier = uuid.uuid4().hex
            self._cancel[identifier] = threading.Event()
            self.store.save('job', identifier, {**fields, 'id': identifier, 'status': 'queued', 'progress': 0})
            self.executor.submit(self.run, identifier)
            return {'id': identifier, 'status': 'queued', 'progress': 0}

    def cancel(self, identifier):
        with self.store.lock:
            job = self.store.load('job', identifier)
            if job['status'] in {'succeeded', 'failed', 'cancelled'} or job.get('committed'):
                return job
            self._cancel.setdefault(identifier, threading.Event()).set()
            status = 'cancelled' if job['status'] == 'queued' else 'cancelling'
            updated = {**job, 'status': status, 'message': 'Cancellation requested'}
            self.store.save('job', identifier, updated)
            return updated

    def close(self):
        with self.store.lock:
            self._closed = True
            for identifier in list(self._cancel):
                self.cancel(identifier)
        self.executor.shutdown(wait=True, cancel_futures=False)

    def _progress(self, identifier, value, message=None, **details):
        with self.store.lock:
            job = self.store.load('job', identifier)
            if job['status'] != 'running':
                return
            fields = value if isinstance(value, dict) else {'progress': value}
            amount = max(job.get('progress', 0), min(99, float(fields.get('progress', 0))))
            safe_details = {key: details[key] for key in ('phase', 'step', 'total', 'loss', 'completed') if key in details}
            committed = job.get('committed', False) or details.get('phase') == 'completed'
            self.store.save('job', identifier, {**job, **safe_details, 'committed': committed, 'progress': amount,
                            'message': message or fields.get('message') or fields.get('status') or 'Processing'})

    def run(self, identifier):
        with self.store.lock:
            job = self.store.load('job', identifier)
            if job['status'] == 'cancelled':
                self._cancel.pop(identifier, None)
                return
            self.store.save('job', identifier, {**job, 'status': 'running'})
        event = self._cancel.setdefault(identifier, threading.Event())
        try:
            if job.get('kind', 'scene') == 'scene':
                result = self.generate(job['scene'], job.get('stamps'))
                fields = {'result_id': result['id']}
            else:
                progress = lambda value, message=None, **details: self._progress(identifier, value, message, **details)
                result = self._operation(job['kind'], job['payload'], progress, event.is_set)
                fields = {'operation_result': result}
            with self.store.lock:
                current = self.store.load('job', identifier)
                self.store.save('job', identifier, {**current, **fields,
                                'status': 'cancelled' if event.is_set() else 'succeeded', 'progress': 100})
        except Exception as error:
            if not event.is_set():
                LOG.exception('Job failed: %s', identifier)
            message = str(error) if isinstance(error, (ValueError, RuntimeError)) else 'Job failed. See the server log for details.'
            with self.store.lock:
                current = self.store.load('job', identifier)
                self.store.save('job', identifier, {**current, 'status': 'cancelled' if event.is_set() else 'failed', 'error': message})
        finally:
            with self.store.lock:
                self._cancel.pop(identifier, None)

    def _operation(self, kind, payload, progress, cancelled):
        if kind == 'augment':
            from backend.library import Library
            return {'stamps': Library(self.store).augment(payload, progress, cancelled)}
        if kind == 'train':
            self.release_inactive_models({})
            return self.training.train(payload, progress, cancelled)
        method = payload['base_model']
        self.release_inactive_models({'brushes': [{}], 'brush_method': method})
        return self.training.generate(payload, self.brush_adapter(method), progress, cancelled)

    def release_inactive_models(self, scene):
        import gc
        import sys
        brush = brush_method(scene) if scene.get('brushes') else None
        released = False
        for name, adapter in [('sdxl', self.ai), ('sd15', self.sd15)]:
            if name != brush and adapter.pipeline is not None:
                adapter.pipeline = None
                released = True
        if hasattr(self.harmonize, 'release_except'):
            uses_stamp = scene.get('placements') or (scene.get('brushes') and brush == 'stamp_brush')
            stamp = scene.get('stamp_method', 'classical_mixed') if uses_stamp else None
            released = self.harmonize.release_except(stamp) or released
        if released:
            gc.collect()
            torch = sys.modules.get('torch')
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate(self, scene, stamps=None):
        self.release_inactive_models(scene)
        source = self.store.load('input', scene['image_id'])
        native = self.store.input_image(source)
        if sha(native) != source['input_sha256']:
            raise ValueError('Input image changed after it was opened')
        with Image.open(native) as image:
            original = image.convert('RGB')
        if stamps is None:
            stamps = self.store.library()
        stamp_lookup = {stamp['id']: stamp for stamp in stamps}
        scene = self.resolve_brush_stamps(scene, stamp_lookup)
        used = [*scene.get('placements', []), *(
                scene.get('brushes', []) if brush_method(scene) in {'stamp_brush', 'stamp_brush_v2'} else [])]
        for placement in used:
            self.verify_stamp_source(stamp_lookup[placement['stamp_id']])
        method = brush_method(scene)
        infer = self.brush_adapter(method)
        lora = self.lora_provenance(scene, method)
        if scene.get('brushes') and method in {'sd15', 'sdxl'}:
            adapter = self.sd15 if method == 'sd15' else self.ai
            adapter.set_lora(scene.get('lora_id'))
        if self.harmonize is None and (scene.get('placements') or method == 'stamp_brush'):
            from scratchforge.harmonization import StampHarmonizer
            self.harmonize = StampHarmonizer(self.store.root)
        generated = generate_scene(original, scene, stamps, self.store.root/'data/library', infer, self.harmonize)
        rgb, guidance, edit, polygons = generated
        identifier = uuid.uuid4().hex
        native_result = save_tiff(rgb, self.store.out/f'{identifier}.tif')
        guidance.save(self.store.out/f'{identifier}-guidance.png')
        edit.save(self.store.out/f'{identifier}-edit.png')
        provenance = {'background': source.get('source'), 'input_sha256': source['input_sha256'],
                      'stamps': list({p['stamp_id']: stamp_lookup[p['stamp_id']] for p in used}.values()),
                      'engine': scene.get('engine') or 'split', 'stamp_method': scene.get('stamp_method', 'classical_mixed'),
                      'brush_method': method, 'seed': scene['seed'], 'strength': scene['strength'],
                      'lora': lora,
                      'edit_margin': scene['edit_margin'], 'ai': self.brush_metadata(method) if scene['brushes'] else None,
                      'layers': generated.layers,
                      'regions': [self.region_mapping(layer['roi'], layer['effective_method']) for layer in generated.layers],
                      'inference': {'steps': 30, 'guidance_scale': 6, 'prompt': PROMPT, 'negative_prompt': NEGATIVE}
                      if scene['brushes'] and method in {'sdxl', 'sd15'} else None}
        labels = self.write_labels(identifier, polygons, rgb.size)
        record = {'id': identifier, 'preview': f'/files/{identifier}.png', 'edit_mask': f'/files/{identifier}-edit.png',
                  'native_image': f'/files/{identifier}.tif', **labels,
                  'guidance_mask': f'/files/{identifier}-guidance.png', 'width': rgb.width, 'height': rgb.height,
                  'polygons': polygons, 'review_status': 'pending', 'scene': scene, 'provenance': provenance,
                  'image_format': 'tiff', 'image_sha256': sha(native_result), 'warnings': list(generated.warnings)}
        self.store.save('result', identifier, record)
        return record

    def lora_provenance(self, scene, method):
        if method not in {'sd15', 'sdxl'} or not scene.get('brushes') or not scene.get('lora_id'):
            return None
        from backend.library import Library
        from scratchforge.lora import LoraRegistry
        adapter = self.sd15 if method == 'sd15' else self.ai
        checkpoint, _ = LoraRegistry(self.store.root).resolve(scene['lora_id'], method, adapter.metadata()['revision'])
        if not checkpoint.get('sources'):
            raise ValueError('LoRA has no verified training source provenance')
        for stamp in checkpoint['sources']:
            Library(self.store).verify_stamp(stamp, historical=True)
        return checkpoint

    def brush_adapter(self, method):
        if method in {'procedural', 'stamp_brush', 'stamp_brush_v2'}:
            return None
        if isinstance(self.infer, dict):
            return self.infer.get(method)
        return self.infer or (self.sd15 if method == 'sd15' else self.ai)

    def brush_metadata(self, method):
        if method in {'procedural', 'stamp_brush', 'stamp_brush_v2'}:
            return None
        return (self.sd15 if method == 'sd15' else self.ai).metadata()

    @staticmethod
    def region_mapping(box, method='sdxl'):
        width, height = box[2]-box[0], box[3]-box[1]
        mapping = {'fullframe_box': list(box), 'native_size': [width, height], 'method': method}
        if method not in {'sdxl', 'sd15'}:
            return {**mapping, 'model_canvas': None, 'resized': False}
        canvas = 512 if method == 'sd15' else 1024
        ratio = canvas/max(width, height)
        return {**mapping, 'model_canvas': [canvas, canvas],
                'scaled_size': [max(1, round(width*ratio)), max(1, round(height*ratio))],
                'padding': 'right-bottom edge padding; mask padding is zero'}

    def annotations(self, identifier, polygons):
        with self.store.lock:
            record = self.store.load('result', identifier)
            if polygons and not valid_polygons(polygons, record['width'], record['height']):
                raise ValueError('Polygons must be finite, nondegenerate and inside the full image')
            labels = self.write_labels(identifier, polygons, (record['width'], record['height']))
            updated = {**record, **labels, 'polygons': polygons, 'review_status': 'pending'}
            updated.pop('export_url', None)
            self.store.save('result', identifier, updated)
            return updated

    def approve(self, identifier):
        with self.store.lock:
            record = self.store.load('result', identifier)
            if not record['polygons']:
                raise ValueError('Verify the final scratch polygons before approving')
            self.verify_image(record)
            self.verify_labels(record)
            updated = {**record, 'review_status': 'approved'}
            self.store.save('result', identifier, updated)
            return updated

    def verify_image(self, record):
        if sha(self.store.result_image(record)) != record['image_sha256']:
            raise ValueError('Result image changed; generate and review it again')

    def write_labels(self, identifier, polygons, size):
        from scratchforge.labels import coco_document
        union = mask_for(polygons, size) if polygons else Image.new('L', size)
        mask_path = self.store.out/f'{identifier}-labels.png'
        polygon_path = self.store.out/f'{identifier}-polygon.json'
        save_image(union, mask_path, 'PNG')
        write_json(polygon_path, coco_document(polygons, size))
        return {'label_mask': f'/files/{identifier}-labels.png',
                'polygon_file': f'/files/{identifier}-polygon.json',
                'label_sha256': sha(mask_path), 'polygon_sha256': sha(polygon_path)}

    def verify_labels(self, record):
        for suffix, key in [('-labels.png', 'label_sha256'), ('-polygon.json', 'polygon_sha256')]:
            if record.get(key):
                path = self.store.out/f"{record['id']}{suffix}"
                if not path.is_file() or sha(path) != record[key]:
                    raise ValueError('Result labels changed; review and approve them again')

    def verify_stamp_source(self, stamp, *, historical=False):
        if stamp.get('origin') or stamp.get('bundle_version'):
            from backend.library import Library
            Library(self.store).verify_stamp(stamp, historical=historical)
            return
        source = {'id': stamp.get('source_frame_id'), 'sha256': stamp.get('source_sha256'),
                  'source_image': stamp.get('source_image'), 'source_image_id': stamp.get('source_image_id')}
        self.store.verify_train_source(source, 'NG', historical=historical)

    def verify_export_sources(self, record):
        background = record['provenance'].get('background')
        if not background or background.get('split') != 'train' or background.get('status') != 'OK':
            raise ValueError('Training exports require an imported train OK background; uploads are exploratory')
        historical = record.get('image_format') != 'tiff'
        self.store.verify_train_source(background, 'OK', historical=historical)
        for stamp in record['provenance']['stamps']:
            if stamp.get('split') != 'train':
                raise ValueError('Training exports require train stamps')
            self.verify_stamp_source(stamp, historical=historical)
        lora = record['provenance'].get('lora')
        if lora:
            if not lora.get('sources'):
                raise ValueError('LoRA training provenance is missing')
            from backend.library import Library
            for stamp in lora['sources']:
                Library(self.store).verify_stamp(stamp, historical=True)

    def export(self, identifier):
        with self.store.lock:
            record = self.store.load('result', identifier)
            if record['review_status'] != 'approved':
                raise PermissionError('Review and approve image and final polygons before training export')
            self.verify_image(record)
            self.verify_labels(record)
            self.verify_export_sources(record)
            url = self._write_export(record)
            self.store.save('result', identifier, {**record, 'export_url': url})
            return {'url': url}

    def _write_export(self, record):
        identifier = record['id']; size = (record['width'], record['height'])
        from scratchforge.labels import coco_document
        coco = coco_document(record['polygons'], size)
        union = mask_for(record['polygons'], size)
        import json
        temporary = self.store.out/f'{identifier}-{uuid.uuid4().hex}.partial'
        try:
            with TemporaryDirectory(dir=self.store.out) as work, zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
                source_image = self.store.result_image(record)
                with Image.open(source_image) as rgb:
                    source_image = save_tiff(rgb.convert('RGB'), Path(work)/'image.tif')
                archive.write(source_image, 'image.tif', compress_type=zipfile.ZIP_STORED)
                mask_path = save_image(union, Path(work)/'mask_full.png', 'PNG')
                archive.write(mask_path, 'mask.png')
                archive.write(mask_path, 'mask_full.png')
                for name, value in [('annotations.json', coco), ('polygon.json', coco),
                                    ('scene.json', record['scene']), ('provenance.json', record['provenance'])]:
                    archive.writestr(name, json.dumps(value, indent=2))
            temporary.replace(self.store.out/f'{identifier}.zip')
        finally:
            temporary.unlink(missing_ok=True)
        return f'/files/{identifier}.zip'
