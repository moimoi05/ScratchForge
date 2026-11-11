"""Optional official harmonization architectures, loaded only on first use.

Small predictor inputs are allowed; the rendered stamp always uses native pixels.
Checkpoint deserialization is tensor-only. No downloads happen in inference.
"""
from contextlib import AbstractContextManager
import importlib
import json
from pathlib import Path
import sys
from threading import RLock

import numpy as np
from PIL import Image

from scratchforge.dataset import sha

_IMPORT_LOCK = RLock()


class SourceScope(AbstractContextManager):
    """Keep upstream generic package names out of the application's namespace."""
    def __init__(self, source, prefix):
        self.source, self.prefix = str(source), prefix
        self.modules = {}

    def _matches(self, name):
        return name == self.prefix or name.startswith(self.prefix + '.')

    def __enter__(self):
        _IMPORT_LOCK.acquire()
        self.previous = {key: value for key, value in sys.modules.copy().items() if self._matches(key)}
        for key in self.previous:
            del sys.modules[key]
        sys.modules.update(self.modules)
        self.paths = sys.path[:]
        sys.path.insert(0, self.source)
        return self

    def __exit__(self, *exc):
        self.modules = {key: value for key, value in sys.modules.copy().items() if self._matches(key)}
        for key in self.modules:
            del sys.modules[key]
        sys.modules.update(self.previous)
        sys.path[:] = self.paths
        _IMPORT_LOCK.release()
        return False


def _build_model(method, config):
    if method == 'pctnet':
        variants = importlib.import_module('iharm.mconfigs').ALL_MCONFIGS
        variant = config.get('variant', 'CNN_pct')
        if variant not in variants:
            raise ValueError('Unknown PCT-Net checkpoint variant')
        definition = variants[variant]
        return definition['model'](**definition['params']), definition['params']['input_normalization']
    if method == 'harmonizer':
        return importlib.import_module('src.model').Harmonizer(), {}
    networks = importlib.import_module('models.networks')
    return networks.RainNet(3, 3, ngf=64, norm_layer=networks.RAIN, use_dropout=True), {}


def _state_dict(torch, checkpoint):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if isinstance(state, dict):
        for key in ('state_dict', 'model'):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict) or not state or not all(
            isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
        raise ValueError('Checkpoint must contain a tensor state dictionary')
    return {key.removeprefix('module.'): value for key, value in state.items()}


def load_adapter(method, folder):
    prefixes = {'pctnet': 'iharm', 'harmonizer': 'src', 'rainnet': 'models'}
    if method not in prefixes:
        raise ValueError('Unknown harmonization model')
    folder = Path(folder)
    checkpoint, source = folder/'checkpoint.pth', folder/'source'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'{method} checkpoint missing; run scripts/setup_harmonization.py')
    if not (source/prefixes[method]).is_dir():
        raise FileNotFoundError(f'{method} official source missing; run scripts/setup_harmonization.py')
    provenance = folder/'model-provenance.json'
    metadata = json.loads(provenance.read_text('utf-8')) if provenance.is_file() else {}
    digest = sha(checkpoint)
    if metadata.get('checkpoint_sha256') != digest:
        raise ValueError('Checkpoint provenance missing or checksum changed; register it again')
    config_path = folder/'config.json'
    config = json.loads(config_path.read_text('utf-8')) if config_path.is_file() else {}
    import torch
    scope = SourceScope(source, prefixes[method])
    with scope:
        model, normalization = _build_model(method, config)
        model.load_state_dict(_state_dict(torch, checkpoint), strict=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()
    details = {**metadata, 'model_id': method, 'device': device,
               'variant': config.get('variant', 'CNN_pct') if method == 'pctnet' else method}
    return NativeAdapter(method, model, device, scope, details, normalization)


class NativeAdapter:
    def __init__(self, method, model, device, scope, metadata, normalization):
        self.method, self.model, self.device = method, model, device
        self.scope, self.metadata, self.normalization = scope, metadata, normalization

    def _predict(self, image, mask):
        import torch
        import torch.nn.functional as functional
        if self.method == 'harmonizer':
            arguments = self.model.predict_arguments(image, mask)
            return self.model.restore_image(image, mask, arguments)[-1]
        if self.method == 'pctnet':
            mean = image.new_tensor(self.normalization['mean']).view(1, 3, 1, 1)
            std = image.new_tensor(self.normalization['std']).view(1, 3, 1, 1)
            normalized = (image-mean)/std
            small = functional.interpolate(normalized, (256, 256), mode='bilinear', align_corners=False)
            small_mask = functional.interpolate(mask, (256, 256), mode='nearest')
            output = self.model(small, image_fullres=normalized, mask=small_mask,
                                mask_fullres=mask)['images_fullres']
            return output.reshape_as(image)*std+mean
        height, width = image.shape[-2:]
        padded_height, padded_width = max(512, (height+255)//256*256), max(512, (width+255)//256*256)
        padding = (0, padded_width-width, 0, padded_height-height)
        padded = functional.pad(image*2-1, padding, mode='replicate')
        padded_mask = functional.pad(mask, padding, mode='constant', value=0)
        output = self.model.processImage(padded, padded_mask)
        return (output[..., :height, :width]+1)/2

    def __call__(self, rgb, mask):
        import torch
        image = torch.from_numpy(np.array(rgb, dtype=np.float32)/255).permute(2, 0, 1)[None].to(self.device)
        support = torch.from_numpy(np.array(mask, dtype=np.float32)/255)[None, None].to(self.device)
        with self.scope, torch.inference_mode():
            output = self._predict(image, support)
        if output.shape != image.shape or not torch.isfinite(output).all():
            raise ValueError('Harmonization returned invalid native-resolution pixels')
        pixels = output[0].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        return Image.fromarray(np.rint(pixels*255).astype('uint8'))
