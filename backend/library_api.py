"""Bounded pair imports and reviewed stamp library operations."""
import json
from typing import Annotated, Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import Field, model_validator

from backend.library import Library, MAX_IMAGE_BYTES, MAX_JSON_BYTES, MAX_MASK_BYTES
from backend.models import Annotations, Approval, StrictModel


class AugmentRequest(StrictModel):
    stamp_ids: list[Annotated[str, Field(min_length=1, max_length=150)]] = Field(min_length=1, max_length=100)
    copies: int = Field(default=1, strict=True, ge=1, le=100)
    seed: int = Field(default=0, strict=True, ge=0, lt=2**32)

    @model_validator(mode='after')
    def bounded(self):
        if len(self.stamp_ids) * self.copies > 500 or len(set(self.stamp_ids)) != len(self.stamp_ids):
            raise ValueError('Select unique stamps and no more than 500 variants')
        return self


def read_upload(upload, limit, description):
    content = upload.file.read(limit + 1)
    if not content or len(content) > limit:
        raise HTTPException(413, f'{description} is empty or exceeds {limit // 1024 // 1024} MiB')
    return content


def library_router(store, jobs):
    router = APIRouter(prefix='/api/library')
    library = Library(store)

    @router.post('/import')
    def import_pair(image: Annotated[UploadFile, File()], mode: Annotated[Literal['fullframe', 'crop'], Form()],
                    source_id: Annotated[str, Form(min_length=1, max_length=150)],
                    confirmed_train: Annotated[bool, Form()],
                    annotations: Annotated[UploadFile | None, File()] = None,
                    mask: Annotated[UploadFile | None, File()] = None):
        if (mode == 'fullframe' and (annotations is None or mask is not None)
                or mode == 'crop' and (mask is None or annotations is not None)):
            raise HTTPException(422, 'Use TIFF + COCO for fullframe, or TIFF + PNG mask for crop')
        coco = None
        if annotations is not None:
            if not (annotations.filename or '').lower().endswith('.json'):
                raise HTTPException(422, 'Use a COCO .json file')
            try:
                coco = json.loads(read_upload(annotations, MAX_JSON_BYTES, 'COCO JSON'))
            except (ValueError, RecursionError) as error:
                raise HTTPException(422, 'Invalid COCO JSON') from error
        return library.import_pair(read_upload(image, MAX_IMAGE_BYTES, 'TIFF image'), mode=mode,
                                   source_id=source_id, confirmed_train=confirmed_train,
                                   filename=image.filename or '', annotations=coco,
                                   mask_bytes=read_upload(mask, MAX_MASK_BYTES, 'Mask') if mask else None)

    @router.post('/augment')
    def augment(request: AugmentRequest):
        payload = request.model_dump()
        for identifier in request.stamp_ids:
            library.verify_stamp(library.lookup(identifier))
        return jobs.submit_operation('augment', payload)

    @router.get('/{identifier}')
    def detail(identifier: str):
        try:
            return library.lookup(identifier)
        except FileNotFoundError as error:
            raise HTTPException(404, 'Stamp not found') from error

    @router.put('/{identifier}/annotations')
    def annotations(identifier: str, request: Annotations):
        return library.annotations(identifier, request.polygons)

    @router.post('/{identifier}/approve')
    def approve(identifier: str, request: Approval):
        return library.approve(identifier)

    return router
