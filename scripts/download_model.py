"""Download a reproducible safetensors-only SDXL snapshot; never upload images."""
import argparse
import json
from pathlib import Path

MODEL_ID = 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1'
DEFAULT_DIR = Path(__file__).resolve().parents[1] / 'data/models/sdxl-inpaint'


def download_model(destination, revision='main', *, model_info=None, snapshot_download=None):
    if model_info is None or snapshot_download is None:
        from huggingface_hub import HfApi, snapshot_download as download
        model_info = model_info or HfApi().model_info
        snapshot_download = snapshot_download or download
    destination = Path(destination).resolve()
    metadata_path = destination / 'model-provenance.json'
    resolved = model_info(MODEL_ID, revision=revision).sha
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding='utf-8'))
        if previous.get('resolved_revision') != resolved or previous.get('model_id') != MODEL_ID:
            raise ValueError('Existing model has a different revision; choose a new destination.')
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID, revision=resolved, local_dir=str(destination),
        allow_patterns=['*.json', '*.txt', '*.model', '*.fp16.safetensors', '*LICENSE*', 'README.md'],
        ignore_patterns=['*.bin', '*.onnx', '*.msgpack', '*.h5', '*.ckpt', '*.xml', '*.tflite'],
    )
    if not (destination / 'model_index.json').is_file():
        raise OSError('Model download incomplete: model_index.json missing.')
    metadata = {'model_id': MODEL_ID, 'revision': revision, 'resolved_revision': resolved}
    temporary = metadata_path.with_suffix('.partial')
    temporary.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    temporary.replace(metadata_path)
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest', type=Path, default=DEFAULT_DIR)
    parser.add_argument('--revision', default='main')
    args = parser.parse_args()
    print(json.dumps(download_model(args.dest, args.revision), indent=2))
