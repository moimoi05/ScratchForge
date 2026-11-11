"""Target-machine CUDA acceptance. Stop the studio before running this command."""
import argparse
import ctypes
import gc
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from backend.images import save_tiff, validate_tiff
from backend.store import Store, write_json
from scratchforge.dataset import ROOT, sha
from scripts.check_gpu import inspect_gpu


CHECKS = ('training', 'weights_reloaded', 'cancelled_without_publication',
          'fullframe_tiff', 'outside_mask_exact', 'scratch_detected')


def physical_memory_gib():
    """Report RAM without adding a platform-specific runtime dependency."""
    if os.name == 'nt':
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                        *[(name, ctypes.c_ulonglong) for name in
                          ('total_physical', 'available_physical', 'total_page', 'available_page',
                           'total_virtual', 'available_virtual', 'extended_virtual')]]
        status = MemoryStatus(); status.length = ctypes.sizeof(status)
        return round(status.total_physical/1024**3, 2) if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) else None
    return round(os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')/1024**3, 2)


def verify_native(original, generated, edit, labels, polygons, destination):
    """Check the encoded TIFF too; an in-memory RGB array alone is insufficient."""
    if generated.size != original.size or edit.size != original.size or labels.size != original.size:
        raise ValueError('Full-frame image/mask dimensions changed')
    outside = np.asarray(edit) == 0
    if not np.array_equal(np.asarray(original)[outside], np.asarray(generated)[outside]):
        raise ValueError('Pixels outside the editable mask changed')
    changed = np.any(np.asarray(original) != np.asarray(generated), axis=2)
    if not changed.any() or not labels.getbbox() or not polygons:
        raise ValueError('No actual generated scratch or usable automatic labels')
    save_tiff(generated, destination)
    with Image.open(destination) as encoded:
        dpi = tuple(round(float(value)) for value in encoded.info.get('dpi', (0, 0)))
        if (encoded.format != 'TIFF' or encoded.mode != 'RGB' or encoded.size != original.size
                or encoded.tag_v2.get(258) != (8, 8, 8) or encoded.tag_v2.get(259) != 5
                or dpi != (300, 300) or not np.array_equal(np.asarray(encoded), np.asarray(generated))):
            raise ValueError('TIFF failed RGB24, LZW, 300 DPI, dimensions or lossless pixel validation')
    return {'fullframe_tiff': True, 'outside_mask_exact': True, 'scratch_detected': True,
            'image': str(destination), 'image_sha256': sha(destination),
            'size': list(original.size), 'changed_pixels': int(changed.sum())}


def _cancel_smoke(training, payload):
    before = {row['id'] for row in training.checkpoints()}
    was_training = False

    def progress(percent, **details):
        nonlocal was_training
        if details.get('phase') == 'training' and details.get('step', 0) >= 1:
            was_training = True

    try:
        training.train({**payload, 'steps': 100000}, progress, lambda: was_training)
    except RuntimeError as error:
        if not was_training or 'cancel' not in str(error).lower():
            raise
    else:
        raise RuntimeError('Cancellation smoke completed a training job instead of cancelling')
    after = {row['id'] for row in training.checkpoints()}
    if before != after or any(training.registry.directory.glob('.*.staging')):
        raise RuntimeError('Cancelled job published a checkpoint or left staging files')
    return True


def _background(store):
    candidates = [row for row in store.index() if row.get('status') == 'OK' and row.get('split') == 'train'
                  and Path(row['path']).suffix.lower() in {'.tif', '.tiff'}]
    if not candidates:
        raise ValueError('Import a verified train OK TIFF before GPU acceptance')
    row = next((row for row in candidates if (row['width'], row['height']) == (5861, 3421)), candidates[0])
    source = store.asset(row['path'])
    if sha(source) != row.get('sha256'):
        raise ValueError('The acceptance background no longer matches its verified checksum')
    with Image.open(source) as image:
        validate_tiff(image)
        if image.size != (row['width'], row['height']):
            raise ValueError('Acceptance background is not full-resolution RGB')
        return image.copy(), row


def _release(adapter, torch):
    adapter.pipeline = None
    adapter.loaded_lora_id = None
    adapter.requested_lora_id = None
    gc.collect()
    torch.cuda.empty_cache()


def smoke_model(store, base_model, stamp_ids, steps):
    """Every successful stage below executes the actual CUDA backend, never mocks."""
    import torch
    from backend.training import Training
    from scratchforge.ai import SD15Generator, SDXLGenerator
    from scratchforge.generation import generate_scene
    from scratchforge.labels import coco_document

    training = Training(store)
    identifiers = stamp_ids or [row['id'] for row in store.library()
                               if str(row.get('review_status', '')).upper() == 'APPROVED'
                               and row.get('split') == 'train' and row.get('origin') not in {'synthetic', 'generated', 'ai'}
                               and Path(row.get('rgb', '')).suffix.lower() in {'.tif', '.tiff'}][:4]
    if not identifiers:
        raise ValueError('Approve at least one original/imported TIFF stamp for the LoRA smoke')
    original, background = _background(store)
    payload = {'stamp_ids': identifiers, 'base_model': base_model, 'steps': steps, 'seed': 42}
    cancelled = _cancel_smoke(training, payload)
    trained = training.train(payload, lambda percent, **details: print(
        json.dumps({'base_model': base_model, 'progress': percent, **details}), flush=True), lambda: False)
    adapter = SD15Generator(store.root) if base_model == 'sd15' else SDXLGenerator(store.root)
    destination = store.out/f"lora-smoke-{base_model}-{trained['lora_id']}.tif"
    scene = {'placements': [], 'brush_method': base_model, 'seed': 42, 'strength': .8, 'edit_margin': 24,
             'brushes': [{'points': [2400, 1600, 2520, 1640, 2680, 1610], 'width': 18}]}
    torch.cuda.reset_peak_memory_stats()
    try:
        adapter.set_lora(trained['lora_id'])
        generated = generate_scene(original, scene, [], store.root/'data/library', infer=adapter)
        torch.cuda.synchronize()
        if not adapter.metadata().get('lora_loaded'):
            raise RuntimeError('Inference did not acknowledge actual LoRA loading')
        checks = verify_native(original, generated.image, generated.edit, generated.label_mask,
                               generated.polygons, destination)
        generated.label_mask.save(destination.with_name(destination.stem+'-mask_full.png'))
        write_json(destination.with_name(destination.stem+'-polygon.json'),
                   coco_document(generated.polygons, original.size, destination.name))
        peak = round(torch.cuda.max_memory_allocated()/1024**3, 3)
    finally:
        _release(adapter, torch)
    return {'status': 'PASS', 'base_model': base_model, 'training': True, 'weights_reloaded': True,
            'cancelled_without_publication': cancelled, **checks, 'lora_id': trained['lora_id'],
            'training_metrics': trained['checkpoint']['metrics'], 'background_id': background['id'],
            'background_sha256': background['sha256'], 'inference_peak_allocated_gib': peak}


def run_acceptance(root, base_models, stamp_ids, steps):
    report = {'status': 'FAIL', 'recorded_at': datetime.now(timezone.utc).isoformat(),
              'machine': platform.platform(), 'physical_memory_gib': physical_memory_gib(),
              'python': sys.version, 'hardware': inspect_gpu(), 'models': [],
              'scope': 'Real CUDA training, cancellation, LoRA reload, full-frame inference and TIFF encoding',
              'halcon': {'status': 'NOT_RUN', 'message': 'Open the output TIFF in HALCON on the target machine manually'}}
    hardware = report['hardware']
    if not (hardware.get('status') == 'PASS' and hardware.get('available') and hardware.get('kernel_pass')):
        report['error'] = 'CUDA kernel evidence unavailable; no training/inference acceptance was attempted'
        return report
    store = Store(root)
    for base in base_models:
        try:
            report['models'].append(smoke_model(store, base, stamp_ids, steps))
        except Exception as error:
            report['models'].append({'base_model': base, 'status': 'FAIL', 'error': str(error)})
            gc.collect()
    if report['models'] and all(row.get('status') == 'PASS' and all(row.get(key) is True for key in CHECKS)
                                for row in report['models']):
        report['status'] = 'PASS'
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--base-model', choices=['sd15', 'sdxl', 'both'], default='both')
    parser.add_argument('--stamp-id', action='append', default=[], help='Approved real TIFF stamp ID; repeat to select several')
    parser.add_argument('--steps', type=int, default=2)
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= 100:
        parser.error('--steps must be between 1 and 100 for a smoke test')
    models = ['sd15', 'sdxl'] if args.base_model == 'both' else [args.base_model]
    report = run_acceptance(args.root.resolve(), models, args.stamp_id, args.steps)
    write_json(args.root/'reports/lora-smoke-runtime.json', report)
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
