import io
import json

import numpy as np
import pytest
from PIL import Image

from backend.store import Store, read_json, write_json
from scratchforge.dataset import sha


def images():
    pixels = np.arange(80 * 60 * 3, dtype=np.uint8).reshape(60, 80, 3)
    rgb = Image.fromarray(pixels)
    mask = Image.new('L', rgb.size)
    mask.paste(255, (12, 22, 66, 29))
    return rgb, mask


def encoded(image, fmt):
    stream = io.BytesIO()
    image.save(stream, format=fmt)
    return stream.getvalue()


def test_bundle_native_tiff_and_checksums(tmp_path):
    from scratchforge.stamps import write_bundle
    rgb, mask = images()
    record = write_bundle(tmp_path, 'fixture', rgb, mask)
    for name in ('stamp_crop.tif', 'mask.png', 'preview_alpha.png', 'polygon.json', 'metadata.json'):
        assert (tmp_path / 'fixture' / name).is_file()
    with Image.open(tmp_path / record['rgb']) as native:
        assert native.mode == 'RGB' and native.format == 'TIFF'
        assert np.array_equal(np.asarray(native), np.asarray(rgb))
    with Image.open(tmp_path / record['preview_alpha']) as preview:
        assert np.array_equal(np.asarray(preview.getchannel('A')), np.asarray(mask))
    assert sha(tmp_path / record['rgb']) == record['rgb_sha256']
    assert sha(tmp_path / record['metadata']) == record['metadata_sha256']
    assert read_json(tmp_path / record['polygon'])['images'][0]['width'] == 80
    assert write_bundle(tmp_path, 'fixture', rgb, mask) == record
    with pytest.raises(ValueError, match='immutable'):
        write_bundle(tmp_path, 'fixture', Image.new('RGB', rgb.size), mask)


def test_bundle_rejects_unsafe_or_invalid_mask(tmp_path):
    from scratchforge.stamps import write_bundle
    rgb, mask = images()
    with pytest.raises(ValueError):
        write_bundle(tmp_path, '../escape', rgb, mask)
    for invalid in (Image.new('L', rgb.size), Image.new('L', rgb.size, 128), Image.new('L', (2, 2), 255)):
        with pytest.raises(ValueError):
            write_bundle(tmp_path, 'invalid', rgb, invalid)
    assert not (tmp_path / 'invalid').exists()


def test_source_extraction_uses_shared_native_bundle(tmp_path):
    from scripts.build_library import make_stamp
    rgb, _ = images()
    source = tmp_path / 'original.tif'
    rgb.save(source, format='TIFF')
    row = {'id': 'f971-0123456789abcdef', 'status': 'NG', 'split': 'train',
           'path': source.name, 'source_image': source.name, 'source_image_id': 1,
           'sha256': sha(source), 'width': rgb.width, 'height': rgb.height}
    annotation = {'id': 9, 'segmentation': [[12, 22, 65, 22, 65, 28, 12, 28]]}
    output = tmp_path / 'data/library'
    output.mkdir(parents=True)
    record = make_stamp(tmp_path, row, annotation, output)
    assert record['rgb'].endswith('/stamp_crop.tif')
    assert (output / record['polygon']).is_file()
    with Image.open(output / record['rgb']) as crop:
        assert np.array_equal(np.asarray(crop), np.asarray(rgb.crop(record['roi'])))
    with Image.open(output / record['mask']) as mask:
        assert np.count_nonzero(np.asarray(mask)) == 53 * 6


def test_legacy_jpeg_extraction_never_claims_native_tiff(tmp_path):
    from scripts.build_library import make_stamp
    rgb, _ = images()
    source = tmp_path / 'legacy.jpg'
    rgb.save(source, format='JPEG')
    row = {'id': 'f971-0123456789abcdef', 'status': 'NG', 'split': 'train',
           'path': source.name, 'source_image': source.name, 'source_image_id': 1,
           'sha256': sha(source), 'width': rgb.width, 'height': rgb.height}
    output = tmp_path / 'data/library'
    output.mkdir(parents=True)
    record = make_stamp(tmp_path, row, {'id': 9, 'segmentation': [[12, 22, 65, 22, 65, 28, 12, 28]]}, output)
    assert record['rgb'].endswith('.png')
    assert record['origin'] == 'legacy'
    assert record['native_parent_tiff'] is False and record['warnings']
    assert 'preview_alpha' not in record


def test_import_review_revision_augmentation_and_provenance(tmp_path):
    from backend.library import Library, merge_library
    service = Library(Store(tmp_path))
    rgb, mask = images()
    first = service.import_pair(encoded(rgb, 'TIFF'), mode='crop', source_id='new-train-frame',
                                confirmed_train=True, mask_bytes=encoded(mask, 'PNG'))[0]
    assert first['review_status'] == 'PENDING_VISUAL_REVIEW'
    service.verify_stamp(first)
    approved = service.approve(first['id'])
    assert approved['review_status'] == 'APPROVED'
    variants = service.augment({'stamp_ids': [first['id']], 'copies': 2, 'seed': 44})
    assert len(variants) == 2 and all(row['origin'] == 'augment' for row in variants)
    for row in variants:
        service.verify_stamp(row)
        assert row['parent_stamp_id'] == first['id']
        assert row['review_status'] == 'PENDING_VISUAL_REVIEW'
    revised = service.annotations(first['id'], [[15, 22, 65, 22, 65, 28, 15, 28]])
    assert revised['id'] != first['id'] and revised['review_status'] == 'PENDING_VISUAL_REVIEW'
    assert first['id'] not in {row['id'] for row in merge_library(service.store, [])}
    assert service.lookup(first['id'])['review_status'] == 'APPROVED'
    assert service.lookup(revised['id'])['id'] == revised['id']
    with Image.open(tmp_path / 'data/library' / variants[0]['rgb']) as left:
        again = service.augment({'stamp_ids': [first['id']], 'copies': 1, 'seed': 44})[0]
        with Image.open(tmp_path / 'data/library' / again['rgb']) as right:
            assert np.array_equal(np.asarray(left), np.asarray(right))


def test_import_rejects_eval_unknown_declaration_and_bad_coco_atomically(tmp_path):
    from backend.library import Library
    store = Store(tmp_path)
    service = Library(store)
    rgb, mask = images()
    data = encoded(rgb, 'TIFF')
    write_json(tmp_path / 'data/fullframe/index.json', [
        {'id': 'eval', 'source_image': 'eval.tif', 'path': 'eval.tif', 'split': 'val', 'sha256': '0' * 64}])
    for source_id, declared in [('new', False), ('eval', True), ('eval.tif', True)]:
        with pytest.raises(ValueError):
            service.import_pair(data, mode='crop', source_id=source_id, confirmed_train=declared,
                                mask_bytes=encoded(mask, 'PNG'))
    coco = {'images': [{'id': 1, 'file_name': 'source.tif', 'width': 80, 'height': 60}],
            'categories': [{'id': 1, 'name': 'scratch'}],
            'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1,
                             'segmentation': [[10, 10, 30, 10, 30, 20, 10, 20]]}]}
    stamps = service.import_pair(data, mode='fullframe', source_id='new', confirmed_train=True, annotations=coco)
    assert len(stamps) == 1
    with Image.open(tmp_path / 'data/library' / stamps[0]['mask']) as imported_mask:
        assert np.count_nonzero(np.asarray(imported_mask)) == 20 * 10
    before = read_json(tmp_path / 'data/library/extra-index.json')
    coco['annotations'].append({**coco['annotations'][0], 'id': 2, 'segmentation': [[0, 0, 1]]})
    with pytest.raises(ValueError):
        service.import_pair(data, mode='fullframe', source_id='other', confirmed_train=True, annotations=coco)
    assert read_json(tmp_path / 'data/library/extra-index.json') == before


def test_augmentation_cancel_and_tampering(tmp_path):
    from backend.library import Library
    service = Library(Store(tmp_path))
    rgb, mask = images()
    record = service.import_pair(encoded(rgb, 'TIFF'), mode='crop', source_id='new', confirmed_train=True,
                                 mask_bytes=encoded(mask, 'PNG'))[0]
    with pytest.raises(RuntimeError, match='cancel'):
        service.augment({'stamp_ids': [record['id']], 'copies': 1, 'seed': 1}, cancelled=lambda: True)
    (tmp_path / 'data/library' / record['rgb']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        service.verify_stamp(record)


def test_import_source_aliases_and_metadata_tamper_are_rejected(tmp_path):
    from backend.library import Library
    store = Store(tmp_path)
    service = Library(store)
    rgb, mask = images()
    data = encoded(rgb, 'TIFF')
    write_json(tmp_path / 'data/fullframe/index.json', [
        {'id': 'f971-eval', 'source_image': 'Evaluation.jpg', 'path': 'Evaluation.tif',
         'split': 'test', 'sha256': '0' * 64}])
    with pytest.raises(ValueError, match='Validation/test'):
        service.import_pair(data, mode='crop', source_id='EVALUATION.TIF', confirmed_train=True,
                            mask_bytes=encoded(mask, 'PNG'))
    record = service.import_pair(data, mode='crop', source_id='new', confirmed_train=True,
                                 mask_bytes=encoded(mask, 'PNG'))[0]
    (tmp_path / 'data/library' / record['metadata']).write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        service.verify_stamp(record)


def test_augmentation_reports_percent_and_commits_before_completion(tmp_path):
    from backend.library import Library
    service = Library(Store(tmp_path))
    rgb, mask = images()
    record = service.import_pair(encoded(rgb, 'TIFF'), mode='crop', source_id='new', confirmed_train=True,
                                 mask_bytes=encoded(mask, 'PNG'))[0]
    updates = []
    def progress(percent, **details):
        updates.append((percent, details))
        if details.get('phase') == 'completed':
            assert len(read_json(service.index_path)) == 2
    service.augment({'stamp_ids': [record['id']], 'copies': 1, 'seed': 0}, progress=progress)
    assert updates[-1] == (100, {'phase': 'completed'})
    assert updates[0][0] == 99 and updates[0][1]['completed'] == 1


def test_segment_withdrawal_does_not_resurrect_extra_approval(tmp_path):
    from backend.library import merge_library
    store = Store(tmp_path)
    old = {'id': 'old', 'source_frame_id': 'frame', 'source_sha256': 'same', 'origin': 'original',
           'review_status': 'APPROVED'}
    child = {'id': 'child', 'origin': 'augment', 'parent_stamp_id': 'old'}
    write_json(tmp_path / 'data/fullframe/index.json', [{'id': 'frame', 'sha256': 'same'}])
    write_json(tmp_path / 'data/library/index.json', [old])
    write_json(tmp_path / 'data/library/extra-index.json', [old, child])
    write_json(tmp_path / 'data/segment/review.json', {'imports': {}, 'reviews': {
        'frame': {'sha256': 'same', 'published_before': True, 'stamps': [], 'history': [old]}}})
    assert merge_library(store, []) == []


def test_legacy_stamp_is_not_promoted_through_augmentation(tmp_path):
    from backend.library import Library
    from scripts.build_library import make_stamp
    store = Store(tmp_path)
    rgb, _ = images()
    source = tmp_path / 'old.jpg'
    rgb.save(source)
    row = {'id': 'frame', 'status': 'NG', 'split': 'train', 'sha256': sha(source), 'path': source.name,
           'source_image': source.name, 'source_image_id': 1, 'width': rgb.width, 'height': rgb.height}
    write_json(tmp_path / 'data/fullframe/index.json', [row])
    record = make_stamp(tmp_path, row, {'id': 1, 'segmentation': [[12, 22, 65, 22, 65, 28, 12, 28]]},
                        tmp_path / 'data/library')
    write_json(tmp_path / 'data/library/index.json', [record])
    with pytest.raises(ValueError, match='TIFF'):
        Library(store).augment({'stamp_ids': [record['id']], 'copies': 1, 'seed': 0})
