"""Versioned stamp publication with explicit train ancestry and review state."""
import hashlib
import io
import json
import re
import shutil
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from backend.images import validate_tiff
from backend.store import read_json, write_json
from scratchforge.dataset import crop_box, mask_for, sha, translate, valid_polygons
from scratchforge.stamps import ASSET_KEYS, augment_pair, write_bundle

MAX_IMAGE_BYTES = 200 * 1024 * 1024
MAX_MASK_BYTES = 32 * 1024 * 1024
MAX_JSON_BYTES = 16 * 1024 * 1024
HOLD_IDS = frozenset({53, 194, 203, 233})


def registered_records(store):
    from backend.segment import historical_library

    directory = store.root / 'data/library'
    return [*read_json(directory / 'index.json', []), *read_json(directory / 'legacy-index.json', []),
            *historical_library(store), *read_json(directory / 'extra-index.json', [])]


def merge_library(store, base_records):
    from backend.segment import read_state

    extras = read_json(store.root / 'data/library/extra-index.json', [])
    records = {row['id']: row for row in [*base_records, *extras]}
    history = {row['id']: row for row in registered_records(store)}
    sources = {row['id']: row for row in store.index()}
    overrides = {key: {item['id'] for item in review['stamps']}
                 for key, review in read_state(store)['reviews'].items()
                 if review.get('published_before') and key in sources
                 and review['sha256'] == sources[key].get('sha256')}

    def available(record, visited=frozenset()):
        if record['id'] in visited:
            return False
        if parent := record.get('parent_stamp_id'):
            return parent in history and available(history[parent], visited | {record['id']})
        frame = record.get('source_frame_id')
        return frame not in overrides or record['id'] in overrides[frame]

    return [row for row in records.values() if not row.get('superseded_by') and available(row)]


def _safe_source_id(identifier):
    if not isinstance(identifier, str) or not re.fullmatch(r'[\w .-]{1,150}', identifier, re.UNICODE):
        raise ValueError('Declare a source ID using letters, numbers, spaces, dots or hyphens')
    return identifier


class Library:
    def __init__(self, store):
        self.store = store
        self.directory = store.root / 'data/library'
        self.index_path = self.directory / 'extra-index.json'

    def lookup(self, identifier):
        records = {row['id']: row for row in registered_records(self.store)}
        if identifier not in records:
            raise FileNotFoundError('Stamp not found')
        return records[identifier]

    def _check_source(self, source_id, digest, filename=None):
        _safe_source_id(source_id)
        for row in self.store.index():
            names = {str(row.get('id', '')), str(row.get('source_image', '')),
                     Path(row.get('path', '')).name, str(row.get('file_name', ''))}
            aliases = {name.casefold() for name in names}
            matches = (source_id.casefold() in aliases or (filename and filename.casefold() in aliases)
                       or digest == row.get('sha256'))
            if matches and row.get('split') != 'train':
                raise ValueError('Validation/test/HOLD sources cannot enter the train stamp library')
        return True

    def verify_stamp(self, stamp, *, historical=False, _visited=None):
        visited = frozenset() if _visited is None else _visited
        if stamp.get('id') in visited or len(visited) >= 64:
            raise ValueError('Invalid cyclic or excessive stamp ancestry')
        visited = visited | {stamp.get('id')}
        if stamp.get('split') != 'train':
            raise ValueError('Only train stamps are available')
        for key in ('rgb', 'mask'):
            path = self.store.asset('data/library/' + stamp.get(key, ''))
            if not stamp.get(key + '_sha256') or sha(path) != stamp[key + '_sha256']:
                raise ValueError('Stamp asset checksum changed')
        for key in ('preview_alpha', 'polygon', 'metadata'):
            if stamp.get(key):
                path = self.store.asset('data/library/' + stamp[key])
                if sha(path) != stamp.get(key + '_sha256'):
                    raise ValueError('Stamp asset checksum changed')
        if parent_id := stamp.get('parent_stamp_id'):
            parent = self.lookup(parent_id)
            self.verify_stamp(parent, historical=historical, _visited=visited)
            if stamp.get('origin') == 'synthetic':
                provenance = stamp.get('provenance', {})
                training = provenance.get('training_stamps', [])
                for item in training:
                    self.verify_stamp(self.lookup(item) if isinstance(item, str) else item,
                                      historical=historical, _visited=visited)
            return
        imported = stamp.get('import_provenance')
        if imported:
            if imported.get('confirmed_train') is not True:
                raise ValueError('Import has no explicit training declaration')
            self._check_source(imported['source_id'], imported['sha256'], imported.get('filename'))
            path = self.store.asset(imported['path'])
            if sha(path) != imported['sha256']:
                raise ValueError('Imported source checksum changed')
            if imported.get('annotation_id') in HOLD_IDS:
                raise ValueError('HOLD source annotation is not eligible')
            return
        source = {'id': stamp.get('source_frame_id'), 'sha256': stamp.get('source_sha256'),
                  'source_image': stamp.get('source_image'), 'source_image_id': stamp.get('source_image_id')}
        row = self.store.verify_train_source(source, 'NG', historical=historical)
        if stamp.get('source_annotation_id') in HOLD_IDS:
            raise ValueError('HOLD source annotation is not eligible')
        if sha(self.store.asset(row['path'])) != row['sha256']:
            raise ValueError('Stamp source checksum changed')

    def publish(self, records):
        with self.store.lock:
            for record in records:
                self.verify_stamp(record)
            previous = read_json(self.index_path, [])
            known = {row['id'] for row in registered_records(self.store)}
            if any(row['id'] in known for row in records) or len({row['id'] for row in records}) != len(records):
                raise ValueError('Stamp ID already registered')
            write_json(self.index_path, [*previous, *records])
        return records

    def import_pair(self, image_bytes, *, mode, source_id, confirmed_train, filename='source.tif',
                    annotations=None, mask_bytes=None):
        if confirmed_train is not True:
            raise ValueError('Confirm that this source is eligible for training')
        if mode not in ('fullframe', 'crop') or not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
            raise ValueError('Invalid import mode or TIFF exceeds 200 MiB')
        if Path(filename).suffix.lower() not in {'.tif', '.tiff'}:
            raise ValueError('Use a TIFF image')
        digest = hashlib.sha256(image_bytes).hexdigest()
        self._check_source(source_id, digest, Path(filename).name)
        try:
            with Image.open(io.BytesIO(image_bytes)) as source:
                validate_tiff(source)
                rgb = source.copy()
            items = self._import_items(rgb, mode, annotations, mask_bytes)
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
            raise ValueError('Invalid TIFF or mask image') from error
        source_key = uuid.uuid4().hex
        source_dir = self.directory / 'imports' / source_key
        source_dir.mkdir(parents=True)
        source_path = source_dir / 'source.tif'
        created = []
        try:
            source_path.write_bytes(image_bytes)
            declaration = {'source_id': source_id, 'confirmed_train': True, 'sha256': digest,
                           'path': source_path.relative_to(self.store.root).as_posix(),
                           'filename': Path(filename).name, 'mode': mode}
            write_json(source_dir / 'declaration.json', declaration)
            for annotation_id, local_rgb, local_mask, polygons, offset in items:
                record = write_bundle(self.directory, uuid.uuid4().hex, local_rgb, local_mask,
                                      polygons=polygons, metadata={
                                          'origin': 'import', 'source_image': Path(filename).name,
                                          'source_offset': offset, 'source_id': source_id,
                                          'import_provenance': {**declaration, 'annotation_id': annotation_id}})
                created.append(record)
            return self.publish(created)
        except Exception:
            self._discard(created)
            shutil.rmtree(source_dir)
            raise

    def _import_items(self, rgb, mode, annotations, mask_bytes):
        from scratchforge.labels import coco_mask, mask_to_polygons

        if mode == 'crop':
            if not mask_bytes or len(mask_bytes) > MAX_MASK_BYTES or annotations is not None:
                raise ValueError('Crop import requires a PNG mask only, at most 32 MiB')
            with Image.open(io.BytesIO(mask_bytes)) as source:
                if source.format != 'PNG' or source.mode not in {'1', 'L'} or source.size != rgb.size:
                    raise ValueError('Use a binary PNG mask matching the crop dimensions')
                mask = source.convert('L')
            if not set(np.unique(np.asarray(mask)).tolist()) <= {0, 255} or not mask.getbbox():
                raise ValueError('Mask must be nonempty binary black/white')
            return [(None, rgb, mask, mask_to_polygons(mask), [0, 0])]
        try:
            if mask_bytes is not None or not isinstance(annotations, dict):
                raise ValueError('Full-frame import requires COCO Polygon JSON')
            frames = annotations['images']
            if len(frames) != 1 or (frames[0]['width'], frames[0]['height']) != rgb.size:
                raise ValueError('COCO must describe this single full frame at its original size')
            categories = {item['id'] for item in annotations['categories'] if item['name'].lower() == 'scratch'}
            instances = annotations['annotations']
            if not 1 <= len(instances) <= 500:
                raise ValueError('COCO must contain 1 to 500 scratch instances')
            items, known, coordinate_count = [], set(), 0
            for annotation in instances:
                identifier, polygons = annotation['id'], annotation['segmentation']
                if (type(identifier) is not int or identifier < 0 or identifier in known
                        or annotation['image_id'] != frames[0]['id'] or annotation['category_id'] not in categories
                        or annotation.get('iscrowd', 0) != 0 or identifier in HOLD_IDS):
                    raise ValueError('COCO contains invalid, duplicate, non-scratch or HOLD instances')
                known.add(identifier)
                if not valid_polygons(polygons, *rgb.size):
                    raise ValueError('Invalid COCO polygons')
                coordinate_count += sum(map(len, polygons))
                if coordinate_count > 200_000:
                    raise ValueError('Too many COCO polygon coordinates')
                points = np.concatenate([np.asarray(poly).reshape(-1, 2) for poly in polygons])
                if (points.min(axis=0) <= 0).any() or (points.max(axis=0) >= np.asarray(rgb.size)-1).any():
                    raise ValueError('HOLD: full-frame scratch touches the image boundary')
                box = crop_box(polygons, rgb.size)
                local = translate(polygons, *box[:2])
                crop = rgb.crop(box)
                local_mask = coco_mask(local, crop.size)
                items.append((identifier, crop, local_mask, mask_to_polygons(local_mask), list(box[:2])))
            return items
        except (KeyError, TypeError, IndexError, AttributeError) as error:
            raise ValueError('Invalid COCO Polygon JSON') from error

    def augment(self, payload, progress=None, cancelled=None):
        identifiers, copies, seed = payload['stamp_ids'], payload.get('copies', 1), payload.get('seed', 0)
        if (not identifiers or len(identifiers) > 100 or len(set(identifiers)) != len(identifiers)
                or type(copies) is not int or not 1 <= copies <= 100 or len(identifiers)*copies > 500
                or type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError('Invalid augmentation selection, count or seed')
        stamps = [self.lookup(identifier) for identifier in identifiers]
        for stamp in stamps:
            self.verify_stamp(stamp)
            if Path(stamp['rgb']).suffix.lower() not in {'.tif', '.tiff'}:
                raise ValueError('Original TIFF stamp is required for augmentation; legacy PNG is reference only')
        records, published = [], False
        try:
            for index, stamp in enumerate(stamps):
                with Image.open(self.store.asset('data/library/' + stamp['rgb'])) as source:
                    rgb = source.convert('RGB')
                with Image.open(self.store.asset('data/library/' + stamp['mask'])) as source:
                    mask = source.convert('L')
                for variant in range(copies):
                    self._cancel(cancelled)
                    sample_seed = int(np.random.SeedSequence([seed, index, variant]).generate_state(1)[0])
                    warped, warped_mask, parameters = augment_pair(rgb, mask, sample_seed)
                    record = write_bundle(self.directory, uuid.uuid4().hex, warped, warped_mask, metadata={
                        'origin': 'augment', 'parent_stamp_id': stamp['id'],
                        'provenance': {'kind': 'augment', 'parameters': parameters}})
                    records.append(record)
                    if progress:
                        progress(99*len(records)/(len(stamps)*copies), phase='augmenting',
                                 completed=len(records), total=len(stamps)*copies)
            with self.store.lock:
                self._cancel(cancelled)
                self.publish(records)
                published = True
                if progress:
                    progress(100, phase='completed')
            return records
        except Exception:
            if not published:
                self._discard(records)
            raise

    @staticmethod
    def _cancel(cancelled):
        if cancelled and cancelled():
            raise RuntimeError('Stamp augmentation cancelled')

    def _discard(self, records):
        for record in records:
            path = (self.directory / record['id']).resolve()
            if path.parent == self.directory.resolve() and path.is_dir():
                shutil.rmtree(path)

    def annotations(self, identifier, polygons):
        with self.store.lock:
            previous = self.lookup(identifier)
            self.verify_stamp(previous)
            if Path(previous['rgb']).suffix.lower() not in {'.tif', '.tiff'}:
                raise ValueError('Original TIFF is required to revise a legacy stamp')
            if (sum(map(len, polygons)) > 200_000 or len(polygons) > 10_000
                    or not valid_polygons(polygons, previous['width'], previous['height'])):
                raise ValueError('Invalid stamp polygons')
            with Image.open(self.store.asset('data/library/' + previous['rgb'])) as source:
                rgb = source.convert('RGB')
            metadata = {key: value for key, value in previous.items()
                        if key not in ASSET_KEYS and not key.endswith('_sha256')
                        and key not in {'id', 'segmentation', 'superseded_by'}}
            metadata.update(parent_stamp_id=identifier, review_status='PENDING_VISUAL_REVIEW')
            revised = write_bundle(self.directory, uuid.uuid4().hex, rgb, mask_for(polygons, rgb.size),
                                   polygons=polygons, metadata=metadata)
            try:
                extras = read_json(self.index_path, [])
                updated = [row for row in extras if row['id'] != identifier]
                write_json(self.index_path, [*updated, {**previous, 'superseded_by': revised['id']}, revised])
                return revised
            except Exception:
                self._discard([revised])
                raise

    def approve(self, identifier):
        with self.store.lock:
            record = self.lookup(identifier)
            self.verify_stamp(record)
            if record.get('superseded_by'):
                raise ValueError('This stamp has a newer annotation revision')
            approved = {**record, 'review_status': 'APPROVED'}
            extras = [row for row in read_json(self.index_path, []) if row['id'] != identifier]
            write_json(self.index_path, [*extras, approved])
            return approved
