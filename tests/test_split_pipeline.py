"""The stamp branch must never enter a diffusion/VAE pipeline."""
from types import SimpleNamespace

import numpy as np
from PIL import Image

from backend.models import Request
from scratchforge.dataset import sha
from scratchforge.generation import generate_scene


def fixture_stamp(tmp_path):
    pixels = np.arange(40*30*3, dtype=np.uint8).reshape(30, 40, 3)
    Image.fromarray(pixels).save(tmp_path/'stamp.png')
    mask = Image.new('L', (40, 30)); mask.paste(255, (5, 10, 35, 14))
    mask.save(tmp_path/'mask.png')
    return {'id': 'one', 'split': 'train', 'rgb': 'stamp.png', 'mask': 'mask.png',
            'rgb_sha256': sha(tmp_path/'stamp.png'), 'mask_sha256': sha(tmp_path/'mask.png')}


def fake_harmonize(rgb, mask, background, method):
    return SimpleNamespace(image=Image.new('RGB', rgb.size, 'green'), requested_method=method,
                           effective_method='classical_mixed', warnings=('Checkpoint unavailable; Classical used.',),
                           metadata={'native_resolution': True})


def test_public_request_accepts_split_methods_and_legacy_scenes():
    common = {'image_id': 'a'*32, 'brushes': [{'points': [10, 10, 30, 30]}]}
    request = Request(**common, stamp_method='rainnet', brush_method='sd15')
    assert request.stamp_method == 'rainnet' and request.brush_method == 'sd15'
    assert Request(**common, engine='sdxl').brush_method == 'sdxl'
    assert Request(**common, engine='baseline').brush_method == 'procedural'
    assert Request(**common, engine='sdxl', brush_method='procedural').brush_method == 'procedural'


def test_stamp_only_keeps_native_roi_and_never_calls_diffusion(tmp_path):
    stamp = fixture_stamp(tmp_path)
    original = Image.new('RGB', (1300, 900), (40, 50, 60))
    seen = []
    def harmonize(rgb, mask, background, method):
        seen.append((rgb.size, mask.size, background.size, method))
        return fake_harmonize(rgb, mask, background, method)
    def infer(*args):
        raise AssertionError('A stamp entered the VAE')
    scene = {'placements': [{'stamp_id': 'one', 'x': 600, 'y': 500}], 'brushes': [],
             'stamp_method': 'pctnet', 'brush_method': 'sdxl', 'edit_margin': 10}
    output = generate_scene(original, scene, [stamp], tmp_path, infer, harmonize=harmonize)
    assert output.image.size == original.size
    assert seen == [((50, 24), (50, 24), (50, 24), 'pctnet')]
    assert output.layers[0]['requested_method'] == 'pctnet'
    assert output.layers[0]['effective_method'] == 'classical_mixed'
    assert output.warnings
    outside = np.asarray(output.edit) == 0
    assert np.array_equal(np.asarray(output.image)[outside], np.asarray(original)[outside])


def test_brush_uses_clean_original_and_stamp_wins_overlap(tmp_path):
    stamp = fixture_stamp(tmp_path)
    original = Image.new('RGB', (200, 100), (10, 20, 30))
    inputs = []
    def infer(rgb, mask, seed, strength):
        inputs.append(np.asarray(rgb))
        return Image.new('RGB', rgb.size, 'red')
    scene = {'placements': [{'stamp_id': 'one', 'x': 90, 'y': 50}],
             'brushes': [{'points': [60, 45, 120, 45], 'width': 12}],
             'stamp_method': 'harmonizer', 'brush_method': 'sdxl', 'edit_margin': 4}
    output = generate_scene(original, scene, [stamp], tmp_path, infer, harmonize=fake_harmonize)
    assert inputs and all(np.all(values == (10, 20, 30)) for values in inputs)
    assert output.image.getpixel((90, 47)) == (0, 128, 0)
    assert [layer['kind'] for layer in output.layers] == ['brush', 'stamp']


def test_procedural_brush_is_deterministic_textured_and_cpu_only(tmp_path):
    original = Image.new('RGB', (160, 100), (60, 70, 80))
    scene = {'brushes': [{'points': [20, 20, 100, 70], 'width': 7}],
             'brush_method': 'procedural', 'seed': 7, 'edit_margin': 8}
    def forbidden(*args):
        raise AssertionError('Procedural requested a model')
    first = generate_scene(original, scene, [], tmp_path, forbidden, harmonize=forbidden)
    second = generate_scene(original, scene, [], tmp_path, forbidden, harmonize=forbidden)
    assert np.array_equal(np.asarray(first.image), np.asarray(second.image))
    inside = np.asarray(first.guidance) > 0
    assert len(np.unique(np.asarray(first.image)[inside], axis=0)) > 10
    assert first.layers[0]['effective_method'] == 'procedural'


def test_sd15_loads_cpu_float32_and_512_canvas(tmp_path, monkeypatch):
    import sys
    from scratchforge.ai import SD15Generator
    adapter = SD15Generator(tmp_path)
    adapter.directory.mkdir(parents=True)
    (adapter.directory/'model_index.json').write_text('{}')
    calls = {}
    class Pipeline:
        vae = SimpleNamespace(enable_tiling=lambda: calls.update(tiling=True))
        def to(self, device): calls['device'] = device
        def enable_attention_slicing(self): calls['slicing'] = True
        def __call__(self, **kwargs):
            calls['canvas'] = kwargs['image'].size
            return SimpleNamespace(images=[kwargs['image']])
    def load(*args, **kwargs):
        calls.update(kwargs)
        return Pipeline()
    fake_torch = SimpleNamespace(float32='fp32', cuda=SimpleNamespace(is_available=lambda: False),
                                 Generator=lambda device: SimpleNamespace(manual_seed=lambda seed: seed))
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setitem(sys.modules, 'diffusers', SimpleNamespace(
        AutoPipelineForInpainting=SimpleNamespace(from_pretrained=load)))
    result = adapter(Image.new('RGB', (120, 70)), Image.new('L', (120, 70), 255), 2, .5)
    assert result.size == (120, 70)
    assert calls['device'] == 'cpu' and calls['torch_dtype'] == 'fp32'
    assert calls['local_files_only'] and calls['use_safetensors']
    assert calls['canvas'] == (512, 512) and calls['slicing']
    assert calls['tiling'] is True
