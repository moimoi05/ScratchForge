"""Register local harmonization weights against pinned official source code."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

SOURCES = {
    'pctnet': ('https://github.com/rakutentech/PCT-Net-Image-Harmonization.git',
               '1572176ed1a72217dad7395391615329b98d30c7'),
    'harmonizer': ('https://github.com/ZHKKKe/Harmonizer.git',
                   '48ecd70becbff50ccaf576db0e64212dbc494e26'),
    'rainnet': ('https://github.com/junleen/RainNet.git',
                'e4b605d301682ce0e0aaf2b00efdc5318906416a'),
}
ROOT = Path(__file__).resolve().parents[1]


def _digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _git(runner, *arguments):
    return runner(['git', *map(str, arguments)], check=True, capture_output=True, text=True).stdout.strip()


def _verify_source(source, repository, revision, runner):
    if _git(runner, '-C', source, 'remote', 'get-url', 'origin') != repository:
        raise ValueError('Existing source repository does not match official repository')
    if _git(runner, '-C', source, 'status', '--porcelain'):
        raise ValueError('Existing source has uncommitted changes')
    if _git(runner, '-C', source, 'rev-parse', 'HEAD') != revision:
        raise ValueError('Existing source revision does not match pinned revision')


def register_model(model, checkpoint, *, root=ROOT, runner=subprocess.run):
    if model not in SOURCES:
        raise ValueError('Unknown harmonization model')
    checkpoint = Path(checkpoint).resolve()
    if (not checkpoint.is_file() or checkpoint.suffix.lower() not in {'.pth', '.pt', '.ckpt'}
            or checkpoint.stat().st_size == 0):
        raise ValueError('Provide a non-empty local checkpoint (.pth, .pt or .ckpt)')
    digest = _digest(checkpoint)
    folder = Path(root).resolve()/'data/models/harmonization'/model
    destination = folder/'checkpoint.pth'
    if destination.exists() and _digest(destination) != digest:
        raise ValueError('A different checkpoint is already registered; use a separate model directory')
    repository, revision = SOURCES[model]
    source = folder/'source'
    if source.exists():
        _verify_source(source, repository, revision, runner)
    else:
        folder.mkdir(parents=True, exist_ok=True)
        _git(runner, 'clone', '--no-checkout', repository, source)
        _git(runner, '-C', source, 'checkout', '--detach', revision)
        _verify_source(source, repository, revision, runner)
    if not destination.exists():
        shutil.copyfile(checkpoint, destination)
    if _digest(destination) != digest:
        raise ValueError('Registered checkpoint checksum mismatch')
    metadata = {'model': model, 'source_repository': repository, 'source_revision': revision,
                'checkpoint_sha256': digest}
    (folder/'model-provenance.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=tuple(SOURCES), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(register_model(args.model, args.checkpoint, root=args.root), indent=2))


if __name__ == '__main__':
    main()
