"""Native-resolution stamp harmonization. No diffusion/VAE path is used here."""
from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from scratchforge.dataset import ROOT

LOG = logging.getLogger(__name__)
METHODS = frozenset({'classical_mixed', 'classical_laplacian', 'pctnet', 'harmonizer', 'rainnet'})


@dataclass(frozen=True)
class HarmonizationResult:
    image: Image.Image
    requested_method: str
    effective_method: str
    warnings: tuple[str, ...]
    metadata: dict


def _reinhard(rgb, mask, background):
    """Contrast-limited Reinhard LAB transfer; a flat target must not erase texture."""
    import cv2
    source = cv2.cvtColor(np.asarray(rgb, dtype=np.float32)/255, cv2.COLOR_RGB2LAB)
    target = cv2.cvtColor(np.asarray(background, dtype=np.float32)/255, cv2.COLOR_RGB2LAB)
    selected = np.asarray(mask) > 0
    ring = (cv2.dilate(selected.astype('uint8'), np.ones((15, 15), 'uint8')) > 0) & ~selected
    reference = target[ring] if ring.any() else target[selected]
    foreground = source[selected]
    mean, deviation = foreground.mean(axis=0), foreground.std(axis=0)
    gain = np.clip(reference.std(axis=0)/np.maximum(deviation, 1e-3), .85, 1.2)
    shift = np.clip(reference.mean(axis=0)-mean, [-8, -20, -20], [8, 20, 20])
    transformed = (source-mean)*gain + mean + shift
    transformed = np.clip(transformed, [0, -127, -127], [100, 127, 127]).astype('float32')
    output = cv2.cvtColor(transformed, cv2.COLOR_LAB2RGB)
    return np.clip(np.rint(output*255), 0, 255).astype('uint8')


def _laplacian(source, background, mask):
    """Laplacian bands stay tied to native pixels; only pyramid levels are reduced."""
    import cv2
    source_levels, target_levels = [source.astype('float32')], [background.astype('float32')]
    mask_levels = [mask.astype('float32')/255]
    for _ in range(min(4, max(0, int(np.log2(min(mask.shape)))-2))):
        source_levels.append(cv2.pyrDown(source_levels[-1]))
        target_levels.append(cv2.pyrDown(target_levels[-1]))
        mask_levels.append(cv2.pyrDown(mask_levels[-1]))
    output = source_levels[-1]*mask_levels[-1][..., None] + target_levels[-1]*(1-mask_levels[-1][..., None])
    for index in range(len(source_levels)-2, -1, -1):
        size = (source_levels[index].shape[1], source_levels[index].shape[0])
        source_band = source_levels[index]-cv2.pyrUp(source_levels[index+1], dstsize=size)
        target_band = target_levels[index]-cv2.pyrUp(target_levels[index+1], dstsize=size)
        alpha = mask_levels[index][..., None]
        output = cv2.pyrUp(output, dstsize=size) + source_band*alpha + target_band*(1-alpha)
    return np.clip(np.rint(output), 0, 255).astype('uint8')


def _classical(rgb, mask, background, method):
    import cv2
    corrected = _reinhard(rgb, mask, background)
    target, binary = np.asarray(background), np.asarray(mask)
    notes = []
    if method == 'classical_mixed':
        x, y, width, height = cv2.boundingRect(binary)
        try:
            if min(width, height) < 3:
                raise ValueError('Thin stamp mask requires Laplacian blending')
            padded = lambda array: cv2.copyMakeBorder(array, 2, 2, 2, 2, cv2.BORDER_REFLECT)
            selected = cv2.copyMakeBorder(binary, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
            mixed = cv2.seamlessClone(padded(corrected[..., ::-1]), padded(target[..., ::-1]), selected,
                                      (x+width//2+2, y+height//2+2), cv2.MIXED_CLONE)
            output = mixed[2:-2, 2:-2, ::-1]
            if output[binary > 0].std() < corrected[binary > 0].std()*.25:
                raise ValueError('Mixed clone attenuated stamp texture; using Laplacian blending')
            return output, method, notes
        except (cv2.error, ValueError) as error:
            notes.append(str(error))
    return _laplacian(corrected, target, binary), 'classical_laplacian', notes


class StampHarmonizer:
    def __init__(self, root=ROOT, adapters=None):
        self.root = Path(root)
        self.adapters = dict(adapters or {})
        self._loaded = {}

    def release_except(self, method=None):
        retained = {key: value for key, value in self._loaded.items() if key == method}
        released = len(retained) != len(self._loaded)
        self._loaded = retained
        return released

    def _adapter(self, method):
        if method in self.adapters:
            return self.adapters[method], {'checkpoint': 'injected-test-adapter'}
        from scratchforge.harmonization_adapters import load_adapter
        if method not in self._loaded:
            self._loaded[method] = load_adapter(method, self.root/'data/models/harmonization'/method)
        adapter = self._loaded[method]
        return adapter, adapter.metadata

    def __call__(self, rgb, mask, background, method='classical_mixed'):
        if method not in METHODS:
            raise ValueError('Unknown stamp harmonization method')
        if rgb.size != mask.size or rgb.size != background.size:
            raise ValueError('Stamp ROI dimensions must match background and mask')
        binary = mask.convert('L').point(lambda value: 255 if value >= 128 else 0)
        if not binary.getbbox():
            raise ValueError('Stamp mask is empty')
        rgb, background = rgb.convert('RGB'), background.convert('RGB')
        metadata = {'native_resolution': True, 'native_size': list(rgb.size), 'vae_used': False}
        notes = []
        effective = method
        if method.startswith('classical_'):
            output, effective, notes = _classical(rgb, binary, background, method)
        else:
            try:
                adapter, details = self._adapter(method)
                generated = adapter(rgb, binary)
                if generated.size != rgb.size or generated.mode != 'RGB':
                    raise ValueError('Harmonization adapter returned incorrect dimensions or RGB mode')
                output = np.asarray(generated)
                metadata = {**metadata, **details}
            except Exception as error:
                LOG.warning('Stamp %s unavailable; Classical fallback: %s', method, error)
                notes.append(f'{method} unavailable: {error}. Falling back to Classical.')
                output, effective, fallback = _classical(rgb, binary, background, 'classical_mixed')
                notes.extend(fallback)
        image = Image.composite(Image.fromarray(output), background, binary)
        return HarmonizationResult(image, method, effective, tuple(notes), metadata)
