"""Killable offline CUDA training subprocess; only complete weights leave staging."""
import json
import sys
import traceback
from pathlib import Path

from backend.store import write_json
from scratchforge.lora_training import run_training


def main(request_path):
    request_path = Path(request_path)
    folder = request_path.parent
    request = json.loads(request_path.read_text(encoding='utf-8'))

    def progress(percent, **details):
        write_json(folder/'progress.json', {'progress': percent, **details})

    try:
        metrics = run_training(request, folder, progress, lambda: (folder/'cancel').exists())
        write_json(folder/'completed.json', {'status': 'succeeded', 'metrics': metrics})
    except Exception as error:
        traceback.print_exc()
        message = str(error)
        if 'out of memory' in message.lower():
            message = 'CUDA out of VRAM while training; no LoRA checkpoint was published'
        write_json(folder/'completed.json', {'status': 'failed', 'error': message})
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1]))
