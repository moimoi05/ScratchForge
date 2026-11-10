import numpy as np
import unittest
from PIL import Image

from scratchforge.generation import build_regions, generate_scene, safe_asset
from scratchforge.dataset import sha


def test_regions_merge_and_preserve_long_scratch():
    regions = build_regions([(10, 20, 1400, 30), (1390, 20, 1500, 40)], (1600, 100), 16)
    assert regions == [(0, 4, 1516, 56)]


def test_mock_ai_changes_only_edit_region():
    original = Image.new('RGB', (100, 90), (12, 23, 34))
    scene = {'placements': [], 'brushes': [{'points': [20, 20, 60, 40], 'width': 3}], 'engine': 'sdxl', 'seed': 2, 'strength': .65, 'edit_margin': 4}
    def infer(rgb, mask, seed, strength):
        return Image.new('RGB', rgb.size, 'red')
    rgb, guidance, edit, polygons = generate_scene(original, scene, [], None, infer)
    outside = np.asarray(edit) == 0
    assert np.array_equal(np.asarray(rgb)[outside], np.asarray(original)[outside])
    assert np.any(np.asarray(rgb)[~outside] != np.asarray(original)[~outside])
    assert guidance.getbbox() and polygons == []


def test_empty_scene_is_rejected():
    with unittest.TestCase().assertRaisesRegex(ValueError, 'scratch'):
        generate_scene(Image.new('RGB', (20, 20)), {'placements': [], 'brushes': [], 'engine': 'baseline'}, [], None)


def test_native_stamp_transform_and_asset_validation(tmp_path):
    rgb = Image.new('RGB', (20, 20), (50, 60, 70))
    mask = Image.new('L', (20, 20)); mask.paste(255, (8, 2, 11, 18))
    rgb.save(tmp_path/'stamp.png'); mask.save(tmp_path/'mask.png')
    stamp = {'id': 'one', 'split': 'train', 'rgb': 'stamp.png', 'mask': 'mask.png',
             'rgb_sha256': sha(tmp_path/'stamp.png'), 'mask_sha256': sha(tmp_path/'mask.png')}
    image = Image.new('RGB', (100, 80), (10, 20, 30))
    placement = {'stamp_id': 'one', 'x': 45, 'y': 40, 'scale': 1.2, 'rotation': 25, 'flip': True}
    scene = {'placements': [placement], 'brushes': [], 'engine': 'baseline'}
    result, guidance, edit, _ = generate_scene(image, scene, [stamp], tmp_path)
    assert guidance.getbbox() and np.array_equal(np.asarray(guidance), np.asarray(edit))
    outside = np.asarray(edit) == 0
    assert np.array_equal(np.asarray(image)[outside], np.asarray(result)[outside])
    with unittest.TestCase().assertRaisesRegex(ValueError, 'Unknown'):
        generate_scene(image, scene, [], tmp_path)
    with unittest.TestCase().assertRaisesRegex(ValueError, 'train'):
        generate_scene(image, scene, [{**stamp, 'split': 'test'}], tmp_path)
    with unittest.TestCase().assertRaises(ValueError):
        safe_asset(tmp_path, '../outside.png')
    Image.new('L', (3, 3)).save(tmp_path/'mask.png')
    stamp = {**stamp, 'mask_sha256': sha(tmp_path/'mask.png')}
    with unittest.TestCase().assertRaisesRegex(ValueError, 'dimensions'):
        generate_scene(image, scene, [stamp], tmp_path)


def test_invalid_inference_dimensions_and_missing_adapter():
    image = Image.new('RGB', (40, 40))
    scene = {'brushes': [{'points': [10, 10, 20, 20], 'width': 2}], 'engine': 'sdxl'}
    with unittest.TestCase().assertRaisesRegex(ValueError, 'unavailable'):
        generate_scene(image, scene, [], None)
    with unittest.TestCase().assertRaisesRegex(ValueError, 'dimensions'):
        generate_scene(image, scene, [], None, lambda *args: Image.new('RGB', (1, 1)))


def test_partly_offscreen_large_stamp_uses_valid_color_sample(tmp_path):
    Image.new('RGB', (20, 20), 'white').save(tmp_path/'stamp.png')
    Image.new('L', (20, 20), 255).save(tmp_path/'mask.png')
    stamp = {'id': 'one', 'split': 'train', 'rgb': 'stamp.png', 'mask': 'mask.png',
             'rgb_sha256': sha(tmp_path/'stamp.png'), 'mask_sha256': sha(tmp_path/'mask.png')}
    scene = {'placements': [{'stamp_id': 'one', 'x': -40, 'y': 40, 'scale': 8}], 'engine': 'baseline'}
    result, guidance, _, _ = generate_scene(Image.new('RGB', (100, 80)), scene, [stamp], tmp_path)
    assert result.size == (100, 80) and guidance.getbbox()


def test_stamp_assets_require_matching_checksums(tmp_path):
    Image.new('RGB', (20, 20), 'white').save(tmp_path/'stamp.png')
    Image.new('L', (20, 20), 255).save(tmp_path/'mask.png')
    stamp = {'id': 'one', 'split': 'train', 'rgb': 'stamp.png', 'mask': 'mask.png',
             'rgb_sha256': sha(tmp_path/'stamp.png'), 'mask_sha256': sha(tmp_path/'mask.png')}
    scene = {'placements': [{'stamp_id': 'one', 'x': 40, 'y': 40}], 'engine': 'baseline'}
    image = Image.new('RGB', (100, 80))
    Image.new('RGB', (20, 20), 'red').save(tmp_path/'stamp.png')
    with unittest.TestCase().assertRaisesRegex(ValueError, 'checksum'):
        generate_scene(image, scene, [stamp], tmp_path)
    stamp = {**stamp, 'rgb_sha256': sha(tmp_path/'stamp.png')}
    stamp.pop('mask_sha256')
    with unittest.TestCase().assertRaisesRegex(ValueError, 'checksum'):
        generate_scene(image, scene, [stamp], tmp_path)


def test_sdxl_offline_metadata_and_missing_model(tmp_path):
    import json
    from scratchforge.ai import SDXLGenerator
    adapter = SDXLGenerator(tmp_path)
    assert adapter.metadata()['model_available'] is False
    with unittest.TestCase().assertRaisesRegex(RuntimeError, 'model missing'):
        adapter(Image.new('RGB', (20, 20)), Image.new('L', (20, 20)), 0, .5)
    with unittest.TestCase().assertRaisesRegex(RuntimeError, 'model missing'):
        adapter._load()
    adapter.directory.mkdir(parents=True)
    (adapter.directory/'model_index.json').write_text('{}')
    (adapter.directory/'model-provenance.json').write_text(json.dumps({'resolved_revision': 'fixed-sha'}))
    assert adapter.metadata()['revision'] == 'fixed-sha'


def test_sdxl_adapter_protocol_with_cpu_fakes(tmp_path, monkeypatch):
    """Checks adapter I/O only; this does not exercise CUDA or model quality."""
    import sys
    from types import SimpleNamespace
    from scratchforge.ai import SDXLGenerator
    adapter = SDXLGenerator(tmp_path)
    adapter.directory.mkdir(parents=True)
    (adapter.directory/'model_index.json').write_text('{}')
    monkeypatch.setitem(sys.modules, 'torch', None)
    with unittest.TestCase().assertRaisesRegex(RuntimeError, 'dependencies missing'):
        adapter._load()
    class OutOfMemoryError(Exception):
        pass
    fake_torch = SimpleNamespace(float16='fp16', cuda=SimpleNamespace(is_available=lambda: False, OutOfMemoryError=OutOfMemoryError, empty_cache=lambda: None),
                                 Generator=lambda device: SimpleNamespace(manual_seed=lambda seed: seed))
    pipeline = SimpleNamespace(enable_model_cpu_offload=lambda: None, enable_vae_tiling=lambda: None)
    factory = SimpleNamespace(from_pretrained=lambda *args, **kwargs: pipeline)
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setitem(sys.modules, 'diffusers', SimpleNamespace(AutoPipelineForInpainting=factory))
    with unittest.TestCase().assertRaisesRegex(RuntimeError, 'CUDA GPU unavailable'):
        adapter._load()
    fake_torch.cuda.is_available = lambda: True
    adapter._load()
    assert adapter.pipeline is pipeline
    received = {}
    def run(**kwargs):
        received.update(kwargs)
        return SimpleNamespace(images=[kwargs['image']])
    adapter.pipeline = run
    result = adapter(Image.new('RGB', (200, 100), 'red'), Image.new('L', (200, 100), 255), 8, .7)
    assert result.size == (200, 100) and received['generator'] == 8
    assert received['image'].size == (1024, 1024)
    def out_of_memory(**kwargs):
        raise OutOfMemoryError()
    adapter.pipeline = out_of_memory
    with unittest.TestCase().assertRaisesRegex(RuntimeError, 'VRAM'):
        adapter(Image.new('RGB', (20, 30)), Image.new('L', (20, 30)), 0, .5)
