"""Native TIFF storage and browser-compatible PNG previews."""
import uuid
from pathlib import Path

from PIL import Image

MAX_TIFF_PIXELS = 64_000_000
MAX_TIFF_UPLOAD = 200 * 1024 * 1024


def validate_tiff(image: Image.Image) -> None:
    if image.format != 'TIFF' or image.mode != 'RGB':
        raise ValueError('Use a TIFF RGB image')
    if image.width * image.height > MAX_TIFF_PIXELS:
        raise ValueError('TIFF must be at most 64 megapixels')
    if image.n_frames != 1 or image.getexif().get(274, 1) != 1:
        raise ValueError('Use a single-page TIFF with standard orientation')
    if image.tag_v2.get(258, (8, 8, 8)) != (8, 8, 8):
        raise ValueError('Use an 8-bit-per-channel RGB TIFF')


def save_image(image: Image.Image, destination: Path, fmt: str, **options) -> Path:
    temporary = destination.with_suffix(f'.{uuid.uuid4().hex}.tmp')
    try:
        image.save(temporary, format=fmt, **options)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def save_tiff(image: Image.Image, destination: Path) -> Path:
    if image.mode != 'RGB':
        raise ValueError('TIFF output requires 24-bit RGB pixels')
    return save_image(image, destination, 'TIFF', compression='tiff_lzw', dpi=(300, 300))


def preview_png(source: Path, destination: Path, max_size: tuple[int, int] | None = None) -> Path:
    if destination.is_file():
        return destination
    with Image.open(source) as image:
        if image.format == 'TIFF':
            validate_tiff(image)
        preview = image.convert('RGB')
        if max_size is not None:
            preview.thumbnail(max_size, Image.Resampling.LANCZOS)
        return save_image(preview, destination, 'PNG', compress_level=1)
