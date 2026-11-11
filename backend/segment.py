"""Local TIFF annotation, review, and publication of versioned train stamps."""

import json
import logging
import uuid
from operator import itemgetter
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.store import read_json, write_json
from scratchforge.dataset import sha, valid_polygons


LOG = logging.getLogger(__name__)
MAX_COORDINATES = 200_000
MAX_ANNOTATIONS = 500
MAX_RINGS = 500
MAX_IMPORT_BYTES = 16 * 1024 * 1024
FIRST_MANUAL_ID = 1_000_000_000
REVIEW_FIELDS = ('id', 'group', 'file_name', 'width', 'height', 'split')

AnnotationID = Annotated[int, Field(strict=True, ge=0, lt=2**53)]
Coordinate = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Ring = Annotated[list[Coordinate], Field(min_length=6, max_length=MAX_COORDINATES)]


class ReviewRevision(BaseModel):
    revision: int = Field(strict=True, ge=0)


class Annotation(BaseModel):
    id: AnnotationID | None = None
    category_id: AnnotationID
    segmentation: list[Ring] = Field(min_length=1, max_length=MAX_RINGS)


class Draft(ReviewRevision):
    annotations: list[Annotation] = Field(max_length=MAX_ANNOTATIONS)


def state_path(store):
    return store.root / 'data/segment/review.json'


def read_state(store):
    return read_json(state_path(store), {'imports': {}, 'reviews': {}})


def catalog(store):
    return read_json(store.root / 'data/segment/index.json', [])


def imported_state(row, state):
    imported = state['imports'].get(row['id'])
    if imported and imported['sha256'] == row['sha256']:
        return imported
    return {'sha256': row['sha256'], 'annotations': row['annotations'], 'revision': 0}


def current_review(row, state):
    review = state['reviews'].get(row['id'])
    if review and review['sha256'] == row['sha256']:
        return review
    return {**imported_state(row, state), 'review_status': 'pending', 'next_id': FIRST_MANUAL_ID,
            'published_before': False, 'stamps': [], 'history': []}


def public_review(row, review, detail=False):
    image_url = f"/api/segment/images/{row['id']}"
    public = {key: row[key] for key in REVIEW_FIELDS}
    public.update(review_status=review['review_status'], revision=review['revision'],
                  annotation_count=len(review['annotations']), preview=f'{image_url}/preview',
                  thumbnail=f'{image_url}/thumbnail')
    return {**public, 'annotations': review['annotations']} if detail else public


def active_library(store, base_stamps):
    with store.lock:
        reviews = read_state(store)['reviews']
        if not reviews:
            return base_stamps
        sources = {row['id']: row for row in store.index()
                   if row['status'] == 'NG' and row['split'] == 'train'}
        overrides = {identifier: review for identifier, review in reviews.items()
                     if review['published_before'] and identifier in sources
                     and sources[identifier]['sha256'] == review['sha256']}
        return [stamp for stamp in base_stamps if stamp.get('source_frame_id') not in overrides] + [
            stamp for review in overrides.values() for stamp in review['stamps']]


def historical_library(store):
    with store.lock:
        return [stamp for review in read_state(store)['reviews'].values() for stamp in review['history']]


def historical_stamp_asset(store, name):
    return any(name in (stamp.get(key) for key in ('rgb', 'mask', 'thumbnail', 'preview_alpha', 'polygon', 'metadata'))
               for stamp in historical_library(store))


def frame_for(store, identifier):
    row = next((row for row in catalog(store) if row['id'] == identifier), None)
    if row is None:
        raise HTTPException(404, 'Segment image not found')
    return row


def verify_source(store, row):
    source = Path(row['path'])
    try:
        if not source.is_absolute():
            source = store.asset(source)
        elif row['group'] != 's942-ok':
            raise ValueError('Unexpected absolute source path')
        if source.suffix.lower() not in {'.tif', '.tiff'} or sha(source) != row['sha256']:
            raise ValueError('Source format or checksum changed')
    except (ValueError, OSError) as error:
        raise HTTPException(422, 'TIFF source unavailable or changed; reload the TIFF catalogue') from error
    return source


def checked_review(row, state, revision):
    previous = state['reviews'].get(row['id'])
    if previous and previous['sha256'] != row['sha256']:
        raise HTTPException(409, 'TIFF source changed after this image was edited')
    review = current_review(row, state)
    if review['revision'] != revision:
        raise HTTPException(409, 'This review changed; reopen the image before saving')
    return review


def edited_annotations(draft, review, size):
    if sum(len(ring) for item in draft.annotations for ring in item.segmentation) > MAX_COORDINATES:
        raise HTTPException(422, 'Too many polygon coordinates')
    known_categories = {item['id']: item['category_id'] for item in review['annotations']}
    next_id = review['next_id']
    annotations, assigned = [], set()
    for item in draft.annotations:
        identifier = item.id
        if identifier is None:
            if item.category_id != 1:
                raise HTTPException(422, 'New annotations must use the scratch category')
            while next_id in known_categories:
                next_id += 1
            identifier = next_id
            next_id += 1
        elif known_categories.get(identifier) != item.category_id or identifier in assigned:
            raise HTTPException(422, 'Unknown or duplicate polygon annotation ID or category')
        assigned.add(identifier)
        rings = [canonical_ring(ring) for ring in item.segmentation]
        if not valid_polygons(rings, *size):
            raise HTTPException(422, 'Polygons must be nondegenerate and inside the full TIFF image')
        annotations.append({'id': identifier, 'category_id': item.category_id, 'segmentation': rings})
    return annotations, next_id


def canonical_ring(ring):
    end = len(ring)
    while end >= 6 and ring[end-2:end] == ring[:2]:
        end -= 2
    return ring[:end]


def boundary_touch(segmentation, width, height):
    return any(x <= 0 or y <= 0 or x >= width - 1 or y >= height - 1
               for ring in segmentation for x, y in zip(ring[::2], ring[1::2]))


def publish_stamps(store, row, review):
    from scripts.build_library import HOLD_ANNOTATION_IDS, make_stamp

    source = next((source for source in store.index() if source['id'] == row['id']), None)
    source_fields = ('sha256', 'source_image', 'source_image_id', 'width', 'height', 'status', 'split')
    if not source or any(source.get(key) != row.get(key) for key in source_fields):
        raise HTTPException(422, 'Stamp publication requires a matching verified train source')
    output = store.root / 'data/library'
    output.mkdir(parents=True, exist_ok=True)
    revision = uuid.uuid4().hex
    eligible = [item for item in review['annotations']
                if item['id'] not in HOLD_ANNOTATION_IDS
                and not boundary_touch(item['segmentation'], row['width'], row['height'])]
    return [{**make_stamp(store.root, source, item, output, revision=revision),
             'review_status': 'APPROVED', 'segment_revision': review['revision'] + 1}
            for item in eligible]


def save_review(store, state, row, review):
    reviews = {**state['reviews'], row['id']: review}
    write_json(state_path(store), {**state, 'reviews': reviews})
    return public_review(row, review, detail=True)


def import_annotations(store, coco):
    from scripts.import_tiff_dataset import EXTRA_SCRATCH_STEM, match_coco

    with store.lock:
        rows = {row['id']: row for row in catalog(store) if row['group'] == 'f971-scratch'}
        if not rows:
            raise HTTPException(422, 'Load the F971 Scratch TIFF catalogue before importing COCO')
        try:
            matched = match_coco(coco, list(rows.values()), ignored_names={EXTRA_SCRATCH_STEM})
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as error:
            raise HTTPException(422, f'Invalid COCO: {error}') from error
        if missing := rows.keys() - matched.keys():
            missing_names = ', '.join(sorted(rows[identifier]['file_name'] for identifier in missing))
            raise HTTPException(422, f'COCO is missing F971 Scratch images: {missing_names}')
        for identifier, annotations in matched.items():
            rings = [item['segmentation'] for item in annotations]
            if (len(annotations) > MAX_ANNOTATIONS or any(len(item) > MAX_RINGS for item in rings)
                    or sum(len(ring) for item in rings for ring in item) > MAX_COORDINATES):
                raise HTTPException(422, f"COCO exceeds polygon limits for {rows[identifier]['file_name']}")
        state = read_state(store)
        by_id = itemgetter('id')
        changes = {identifier: annotations for identifier, annotations in matched.items()
                   if sorted(annotations, key=by_id)
                   != sorted(imported_state(rows[identifier], state)['annotations'], key=by_id)}
        if conflicts := changes.keys() & state['reviews'].keys():
            filenames = ', '.join(sorted(rows[identifier]['file_name'] for identifier in conflicts))
            raise HTTPException(409, f'COCO conflicts with local edits or approvals: {filenames}')
        if changes:
            imports = {identifier: {'sha256': rows[identifier]['sha256'], 'annotations': annotations,
                                    'revision': imported_state(rows[identifier], state)['revision'] + 1}
                       for identifier, annotations in changes.items()}
            write_json(state_path(store), {**state, 'imports': {**state['imports'], **imports}})
        return {'images': len(matched), 'annotations': sum(map(len, matched.values()))}


def segment_router(store):
    router = APIRouter(prefix='/api/segment')

    @router.get('/images')
    def list_images():
        with store.lock:
            state = read_state(store)
            return [public_review(row, current_review(row, state)) for row in catalog(store)]

    @router.get('/images/{identifier}')
    def image_details(identifier: str):
        with store.lock:
            row = frame_for(store, identifier)
            return public_review(row, current_review(row, read_state(store)), detail=True)

    @router.get('/images/{identifier}/{kind}')
    def preview(identifier: str, kind: str):
        from backend.images import preview_png

        if kind not in {'preview', 'thumbnail'}:
            raise HTTPException(404, 'Segment image view not found')
        row = frame_for(store, identifier)
        source = verify_source(store, row)
        destination = store.out / 'segment-preview' / f"{row['sha256']}-{kind}.png"
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            path = preview_png(source, destination, max_size=(320, 240) if kind == 'thumbnail' else None)
        except (ValueError, OSError) as error:
            raise HTTPException(422, 'Unable to render TIFF preview') from error
        return FileResponse(path, media_type='image/png')

    @router.put('/images/{identifier}/annotations')
    def save_annotations(identifier: str, draft: Draft):
        with store.lock:
            row = frame_for(store, identifier)
            verify_source(store, row)
            state = read_state(store)
            review = checked_review(row, state, draft.revision)
            annotations, next_id = edited_annotations(draft, review, (row['width'], row['height']))
            if annotations == review['annotations']:
                return public_review(row, review, detail=True)
            updated = {**review, 'revision': review['revision'] + 1, 'annotations': annotations,
                       'next_id': next_id, 'review_status': 'pending', 'stamps': []}
            return save_review(store, state, row, updated)

    @router.post('/images/{identifier}/approve')
    def approve(identifier: str, request: ReviewRevision):
        with store.lock:
            row = frame_for(store, identifier)
            verify_source(store, row)
            state = read_state(store)
            review = checked_review(row, state, request.revision)
            if review['review_status'] == 'approved':
                return public_review(row, review, detail=True)
            train = row['group'] == 'f971-scratch' and row['split'] == 'train'
            try:
                stamps = publish_stamps(store, row, review) if train else []
            except (ValueError, OSError) as error:
                LOG.exception('Could not build approved stamps for %s', identifier)
                raise HTTPException(422, 'Unable to build approved stamps. See the server log for details.') from error
            updated = {**review, 'revision': review['revision'] + 1, 'review_status': 'approved',
                       'published_before': train, 'stamps': stamps, 'history': [*review['history'], *stamps]}
            return save_review(store, state, row, updated)

    @router.post('/import')
    def upload_coco(file: UploadFile):
        content = file.file.read(MAX_IMPORT_BYTES + 1)
        if len(content) > MAX_IMPORT_BYTES:
            raise HTTPException(413, 'COCO import exceeds 16 MiB')
        try:
            coco = json.loads(content)
        except (ValueError, RecursionError) as error:
            raise HTTPException(422, 'Use a valid COCO JSON file') from error
        return import_annotations(store, coco)

    return router
