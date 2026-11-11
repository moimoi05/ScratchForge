import sys
from types import ModuleType

import numpy as np
import pytest
from PIL import Image

from scratchforge.harmonization_adapters import NativeAdapter, SourceScope, load_adapter


def test_missing_checkpoint_fails_before_import(tmp_path):
    with pytest.raises(FileNotFoundError, match='checkpoint'):
        load_adapter('pctnet', tmp_path)


def test_load_validates_provenance_and_uses_tensor_only_checkpoint(tmp_path, monkeypatch):
    import json
    torch = pytest.importorskip('torch')
    import scratchforge.harmonization_adapters as adapters
    from scratchforge.dataset import sha
    checkpoint = tmp_path/'checkpoint.pth'
    torch.save({'state_dict': {'module.weight': torch.ones(1)}}, checkpoint)
    with pytest.raises(FileNotFoundError, match='source'):
        load_adapter('rainnet', tmp_path)
    (tmp_path/'source/models').mkdir(parents=True)
    with pytest.raises(ValueError, match='provenance'):
        load_adapter('rainnet', tmp_path)
    (tmp_path/'model-provenance.json').write_text(json.dumps({'checkpoint_sha256': sha(checkpoint)}))
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
    monkeypatch.setattr(adapters, '_build_model', lambda *args: (Model(), {}))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    loaded = load_adapter('rainnet', tmp_path)
    assert loaded.model.weight.item() == 1
    assert not loaded.model.training
    assert loaded.metadata['checkpoint_sha256'] == sha(checkpoint)
    torch.save({'model': {'weight': torch.zeros(1)}}, checkpoint)
    assert adapters._state_dict(torch, checkpoint)['weight'].item() == 0
    torch.save({'unsafe': 'not a tensor'}, checkpoint)
    with pytest.raises(ValueError, match='tensor'):
        adapters._state_dict(torch, checkpoint)
    with pytest.raises(ValueError, match='Unknown'):
        load_adapter('unknown', tmp_path)


def test_model_constructors_and_variant_validation(monkeypatch):
    import scratchforge.harmonization_adapters as adapters
    from types import SimpleNamespace
    constructor = lambda **kwargs: kwargs
    modules = {'iharm.mconfigs': SimpleNamespace(ALL_MCONFIGS={
        'CNN_pct': {'model': constructor, 'params': {'input_normalization': {}}}}),
        'src.model': SimpleNamespace(Harmonizer=lambda: 'harmonizer'),
        'models.networks': SimpleNamespace(RAIN='rain', RainNet=lambda *args, **kwargs: args)}
    monkeypatch.setattr(adapters.importlib, 'import_module', lambda name: modules[name])
    assert adapters._build_model('pctnet', {}) == ({'input_normalization': {}}, {})
    assert adapters._build_model('harmonizer', {}) == ('harmonizer', {})
    assert adapters._build_model('rainnet', {}) == ((3, 3), {})
    with pytest.raises(ValueError, match='variant'):
        adapters._build_model('pctnet', {'variant': 'missing'})


def test_source_scope_restores_colliding_modules_on_error(tmp_path):
    previous = ModuleType('models')
    sys.modules['models'] = previous
    scope = SourceScope(tmp_path, 'models')
    with pytest.raises(RuntimeError):
        with scope:
            assert 'models' not in sys.modules
            sys.modules['models'] = ModuleType('models')
            raise RuntimeError('test')
    assert sys.modules['models'] is previous
    del sys.modules['models']


def test_reject_nonfinite_generated_pixels(tmp_path):
    torch = pytest.importorskip('torch')
    adapter = NativeAdapter('harmonizer', None, 'cpu', SourceScope(tmp_path, 'unused'), {}, {})
    adapter._predict = lambda image, mask: torch.full_like(image, float('nan'))
    with pytest.raises(ValueError, match='invalid native-resolution'):
        adapter(Image.new('RGB', (5, 7)), Image.new('L', (5, 7), 255))


@pytest.mark.parametrize('method', ['pctnet', 'harmonizer', 'rainnet'])
def test_native_geometry_and_normalization_roundtrip(method, tmp_path):
    torch = pytest.importorskip('torch')
    calls = []
    class Identity:
        def __call__(self, image, image_fullres, mask, mask_fullres):
            calls.append((image.shape, image_fullres.shape))
            return {'images_fullres': image_fullres[0]}
        def predict_arguments(self, image, mask):
            calls.append(image.shape)
            return []
        def restore_image(self, image, mask, arguments):
            return [image]
        def processImage(self, image, mask):
            calls.append(image.shape)
            return image
    rgb = Image.fromarray(np.random.default_rng(3).integers(0, 256, (37, 71, 3), dtype='uint8'))
    adapter = NativeAdapter(method, Identity(), 'cpu', SourceScope(tmp_path, 'unused'), {},
                            {'mean': [.485, .456, .406], 'std': [.229, .224, .225]})
    result = adapter(rgb, Image.new('L', rgb.size, 255))
    assert result.size == rgb.size
    assert np.array_equal(np.asarray(result), np.asarray(rgb))
    if method == 'pctnet':
        assert calls[0][0][-2:] == (256, 256)
        assert calls[0][1][-2:] == (37, 71)
    elif method == 'harmonizer':
        assert calls[0][-2:] == (37, 71)
    else:
        assert calls[0][-2:] == (512, 512)
