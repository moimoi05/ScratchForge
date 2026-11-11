import numpy as np
import pytest
from PIL import Image

from scratchforge.harmonization import StampHarmonizer


@pytest.fixture
def composite():
    yy, xx = np.mgrid[:65, :97]
    background = np.repeat((60 + xx // 8 + yy // 10)[..., None], 3, axis=2).astype('uint8')
    mask = np.zeros((65, 97), dtype='uint8')
    mask[15:52, 35:43] = 255
    source = background.copy()
    source[mask > 0] = np.repeat((150 + (xx[mask > 0] % 2) * 50)[:, None], 3, axis=1)
    return Image.fromarray(source), Image.fromarray(mask), Image.fromarray(background)


@pytest.mark.parametrize('method', ['classical_mixed', 'classical_laplacian'])
def test_classical_native_geometry_background_and_texture(tmp_path, composite, method):
    rgb, mask, background = composite
    original = np.asarray(rgb).copy()
    result = StampHarmonizer(tmp_path)(rgb, mask, background, method)
    outside = np.asarray(mask) == 0
    assert result.image.size == rgb.size
    assert result.effective_method.startswith('classical_')
    assert np.array_equal(np.asarray(result.image)[outside], np.asarray(background)[outside])
    assert np.asarray(result.image)[~outside].std() > 2
    assert np.array_equal(np.asarray(rgb), original)


@pytest.mark.parametrize('method', ['pctnet', 'harmonizer', 'rainnet'])
def test_missing_ai_checkpoint_explicit_classical_fallback(tmp_path, composite, method):
    result = StampHarmonizer(tmp_path)(*composite, method)
    assert result.requested_method == method
    assert result.effective_method.startswith('classical_')
    assert any('checkpoint' in warning.lower() for warning in result.warnings)
    assert result.metadata['native_resolution'] is True


def test_ai_adapter_receives_native_roi_and_only_changes_mask(tmp_path, composite):
    received = []
    def adapter(rgb, mask):
        received.append((rgb.size, mask.size))
        return Image.new('RGB', rgb.size, 'white')
    rgb, mask, background = composite
    result = StampHarmonizer(tmp_path, adapters={'pctnet': adapter})(rgb, mask, background, 'pctnet')
    assert received == [((97, 65), (97, 65))]
    assert result.effective_method == 'pctnet'
    assert result.warnings == ()
    outside = np.asarray(mask) == 0
    assert np.array_equal(np.asarray(result.image)[outside], np.asarray(background)[outside])


def test_invalid_ai_output_falls_back_with_warning(tmp_path, composite):
    result = StampHarmonizer(tmp_path, adapters={'rainnet': lambda *args: Image.new('RGB', (2, 2))})(
        *composite, 'rainnet')
    assert result.effective_method.startswith('classical_')
    assert any('dimensions' in warning for warning in result.warnings)


def test_thin_mask_uses_native_laplacian(tmp_path, composite):
    rgb, _, background = composite
    mask = Image.new('L', rgb.size)
    mask.paste(255, (40, 15, 41, 50))
    result = StampHarmonizer(tmp_path)(rgb, mask, background, 'classical_mixed')
    assert result.effective_method == 'classical_laplacian'
    assert result.warnings


def test_reject_unknown_method_empty_mask_and_size_mismatch(tmp_path, composite):
    harmonizer = StampHarmonizer(tmp_path)
    with pytest.raises(ValueError, match='Unknown'):
        harmonizer(*composite, 'imaginary')
    with pytest.raises(ValueError, match='mask'):
        harmonizer(composite[0], Image.new('L', composite[0].size), composite[2], 'classical_mixed')
    with pytest.raises(ValueError, match='dimensions'):
        harmonizer(composite[0], Image.new('L', (2, 2)), composite[2], 'classical_mixed')
