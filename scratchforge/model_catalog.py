"""Allowlisted official model sources; checkpoints verified against recorded hashes."""
from scripts.setup_harmonization import SOURCES

MODEL_CATALOG = {
    'pctnet': {
        'label': 'PCT-Net', 'size_label': '8.8 MB', 'size_bytes': 8802715,
        'checkpoint_url': 'https://raw.githubusercontent.com/rakutentech/PCT-Net-Image-Harmonization/1572176ed1a72217dad7395391615329b98d30c7/pretrained_models/PCTNet_CNN.pth',
        'checkpoint_sha256': '38492a24d9a876777a30a844d7563c7514fa11c0914728e104eb28af222291aa',
        'variant': 'CNN_pct',
    },
    'harmonizer': {
        'label': 'Harmonizer', 'size_label': '19.2 MB', 'size_bytes': 19215729,
        'checkpoint_url': 'https://drive.usercontent.google.com/download?id=15XGPQHBppaYGnhsP9l7iOGZudXNw1WbA&export=download&confirm=t',
        'checkpoint_sha256': '401dd03706fc6fa37b4600ae4c643a2fa9ad0f11351acfbf030cb5d3490de6ca',
    },
    'rainnet': {
        'label': 'RainNet', 'size_label': '219 MB', 'size_bytes': 219057338,
        'checkpoint_url': 'https://drive.usercontent.google.com/download?id=1nVJFQ1iAGMeZ-ZybJm9vBQfvYs6tkctZ&export=download&confirm=t',
        'checkpoint_sha256': 'cdc8cefb31909f222d9b7166a39e4fc7ca0a65753eb68e3f0cf36010a94c1c7e',
    },
    'sdxl': {
        'label': 'SDXL Inpainting', 'size_label': '~6.9 GB',
        'repo_id': 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1',
    },
    'sd15': {
        'label': 'SD 1.5 Inpainting', 'size_label': '~2.2 GB',
        'repo_id': 'stable-diffusion-v1-5/stable-diffusion-inpainting',
    },
}
MODEL_CATALOG = {
    key: {**value, **({'source_repository': SOURCES[key][0], 'source_revision': SOURCES[key][1],
                     'checkpoint_size': value['size_bytes']}
                    if key in SOURCES else {})}
    for key, value in MODEL_CATALOG.items()
}
