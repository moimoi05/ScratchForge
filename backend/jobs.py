"""Serialized jobs and reviewed training export lifecycle."""
import logging
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

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
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='scratchforge-gpu')
        for path in store.out.glob('*-job.json'):
            job = read_json(path)
            if job.get('status') in ('queued', 'running'):
                write_json(path, {**job, 'status': 'failed', 'error': 'Application stopped before this job completed. Please retry.'})

    def submit(self, scene):
        self.store.load('input', scene['image_id'])
        stamps = {stamp['id']: stamp for stamp in self.store.library()}
        for placement in scene['placements']:
            if placement['stamp_id'] not in stamps or stamps[placement['stamp_id']].get('split') != 'train':
                raise ValueError('Unknown or non-training stamp')
            self.verify_stamp_source(stamps[placement['stamp_id']])
        queued = sum(read_json(path).get('status') in ('queued', 'running') for path in self.store.out.glob('*-job.json'))
        if queued >= 8:
            raise ValueError('Job queue is full. Wait for a running job to finish.')
        identifier = uuid.uuid4().hex
        job = {'id': identifier, 'status': 'queued', 'scene': scene}
        self.store.save('job', identifier, job)
        self.executor.submit(self.run, identifier)
        return {'id': identifier, 'status': 'queued'}

    def run(self, identifier):
        job = self.store.load('job', identifier)
        self.store.save('job', identifier, {**job, 'status': 'running'})
        try:
            result = self.generate(job['scene'])
            self.store.save('job', identifier, {**job, 'status': 'succeeded', 'result_id': result['id']})
        except Exception as error:
            LOG.exception('Generation job failed: %s', identifier)
            message = str(error) if isinstance(error, (ValueError, RuntimeError)) else 'Generation failed. See the server log for details.'
            self.store.save('job', identifier, {**job, 'status': 'failed', 'error': message})

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
            stamp = scene.get('stamp_method', 'classical_mixed') if scene.get('placements') else None
            released = self.harmonize.release_except(stamp) or released
        if released:
            gc.collect()
            torch = sys.modules.get('torch')
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate(self, scene):
        self.release_inactive_models(scene)
        source = self.store.load('input', scene['image_id'])
        with Image.open(self.store.out/f"{scene['image_id']}-input.png") as image:
            original = image.convert('RGB')
        if sha(self.store.out/f"{scene['image_id']}-input.png") != source['input_sha256']:
            raise ValueError('Input image changed after it was opened')
        stamps = self.store.library()
        stamp_lookup = {stamp['id']: stamp for stamp in stamps}
        for placement in scene['placements']:
            self.verify_stamp_source(stamp_lookup[placement['stamp_id']])
        method = brush_method(scene)
        infer = self.brush_adapter(method)
        if self.harmonize is None and scene['placements']:
            from scratchforge.harmonization import StampHarmonizer
            self.harmonize = StampHarmonizer(self.store.root)
        generated = generate_scene(original, scene, stamps, self.store.root/'data/library', infer, self.harmonize)
        rgb, guidance, edit, polygons = generated
        identifier = uuid.uuid4().hex
        rgb.save(self.store.out/f'{identifier}.png')
        guidance.save(self.store.out/f'{identifier}-guidance.png')
        edit.save(self.store.out/f'{identifier}-edit.png')
        provenance = {'background': source.get('source'), 'input_sha256': source['input_sha256'],
                      'stamps': [stamp_lookup[p['stamp_id']] for p in scene['placements']],
                      'engine': scene.get('engine') or 'split', 'stamp_method': scene.get('stamp_method', 'classical_mixed'),
                      'brush_method': method, 'seed': scene['seed'], 'strength': scene['strength'],
                      'edit_margin': scene['edit_margin'], 'ai': self.brush_metadata(method) if scene['brushes'] else None,
                      'layers': generated.layers,
                      'regions': [self.region_mapping(layer['roi'], layer['effective_method']) for layer in generated.layers],
                      'inference': {'steps': 30, 'guidance_scale': 6, 'prompt': PROMPT, 'negative_prompt': NEGATIVE}
                      if scene['brushes'] and method in {'sdxl', 'sd15'} else None}
        record = {'id': identifier, 'preview': f'/files/{identifier}.png', 'edit_mask': f'/files/{identifier}-edit.png',
                  'guidance_mask': f'/files/{identifier}-guidance.png', 'width': rgb.width, 'height': rgb.height,
                  'polygons': polygons, 'review_status': 'pending', 'scene': scene, 'provenance': provenance,
                  'image_sha256': sha(self.store.out/f'{identifier}.png'), 'warnings': list(generated.warnings)}
        self.store.save('result', identifier, record)
        return record

    def brush_adapter(self, method):
        if method == 'procedural':
            return None
        if isinstance(self.infer, dict):
            return self.infer.get(method)
        return self.infer or (self.sd15 if method == 'sd15' else self.ai)

    def brush_metadata(self, method):
        if method == 'procedural':
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
            updated = {**record, 'polygons': polygons, 'review_status': 'pending'}
            updated.pop('export_url', None)
            self.store.save('result', identifier, updated)
            return updated

    def approve(self, identifier):
        with self.store.lock:
            record = self.store.load('result', identifier)
            if not record['polygons']:
                raise ValueError('Draw and verify the final scratch polygons before approving')
            self.verify_image(record)
            updated = {**record, 'review_status': 'approved'}
            self.store.save('result', identifier, updated)
            return updated

    def verify_image(self, record):
        if sha(self.store.out/f"{record['id']}.png") != record['image_sha256']:
            raise ValueError('Result image changed; generate and review it again')

    def verify_stamp_source(self, stamp):
        source = {'id': stamp.get('source_frame_id'), 'sha256': stamp.get('source_sha256'),
                  'source_image': stamp.get('source_image'), 'source_image_id': stamp.get('source_image_id')}
        self.store.verify_train_source(source, 'NG')

    def verify_export_sources(self, record):
        background = record['provenance'].get('background')
        if not background or background.get('split') != 'train' or background.get('status') != 'OK':
            raise ValueError('Training exports require an imported train OK background; uploads are exploratory')
        self.store.verify_train_source(background, 'OK')
        for stamp in record['provenance']['stamps']:
            if stamp.get('split') != 'train':
                raise ValueError('Training exports require train stamps')
            self.verify_stamp_source(stamp)

    def export(self, identifier):
        with self.store.lock:
            record = self.store.load('result', identifier)
            if record['review_status'] != 'approved':
                raise PermissionError('Review and approve image and final polygons before training export')
            self.verify_image(record)
            self.verify_export_sources(record)
            url = self._write_export(record)
            self.store.save('result', identifier, {**record, 'export_url': url})
            return {'url': url}

    def _write_export(self, record):
        identifier = record['id']; size = (record['width'], record['height'])
        union = Image.new('L', size); annotations = []
        for polygon in record['polygons']:
            mask = mask_for([polygon], size); union.paste(255, (0, 0), mask)
            box = mask.getbbox()
            annotations.append({'id': len(annotations)+1, 'image_id': 1, 'category_id': 1,
                                'segmentation': [polygon], 'iscrowd': 0, 'area': int(np.count_nonzero(np.asarray(mask))),
                                'bbox': [box[0], box[1], box[2]-box[0], box[3]-box[1]]})
        union.save(self.store.out/f'{identifier}-labels.png')
        coco = {'images': [{'id': 1, 'file_name': 'image.png', 'width': size[0], 'height': size[1]}],
                'categories': [{'id': 1, 'name': 'scratch'}], 'annotations': annotations}
        import json
        temporary = self.store.out/f'{identifier}-{uuid.uuid4().hex}.partial'
        try:
            with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
                archive.write(self.store.out/f'{identifier}.png', 'image.png')
                archive.write(self.store.out/f'{identifier}-labels.png', 'mask.png')
                for name, value in [('annotations.json', coco), ('scene.json', record['scene']), ('provenance.json', record['provenance'])]:
                    archive.writestr(name, json.dumps(value, indent=2))
            temporary.replace(self.store.out/f'{identifier}.zip')
        finally:
            temporary.unlink(missing_ok=True)
        return f'/files/{identifier}.zip'
