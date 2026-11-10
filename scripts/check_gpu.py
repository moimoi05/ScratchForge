"""Record current hardware evidence; run model smoke on the target NVIDIA GPU."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scratchforge.dataset import ROOT, save_json


def inspect_gpu(torch_module=None):
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError:
            return {'status': 'NOT_AVAILABLE', 'available': False, 'kernel_pass': False,
                    'message': 'PyTorch not installed. Run setup.ps1 -Gpu on the NVIDIA machine.'}
    torch = torch_module
    report = {'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
              'available': torch.cuda.is_available(), 'kernel_pass': False,
              'status': 'NOT_AVAILABLE'}
    if report['available']:
        report.update(device=torch.cuda.get_device_name(),
                      capability=torch.cuda.get_device_capability(),
                      architectures=torch.cuda.get_arch_list(),
                      vram_gib=round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
        try:
            matrix = torch.randn(256, 256, device='cuda')
            product = matrix @ matrix
            torch.cuda.synchronize()
            report['kernel_pass'] = bool(torch.isfinite(product).all().item())
            report['status'] = 'PASS' if report['kernel_pass'] else 'FAIL'
        except RuntimeError as error:
            report.update(status='FAIL', message=str(error))
    return report


def inference_smoke(model_dir):
    import torch
    from diffusers import AutoPipelineForInpainting
    from PIL import Image, ImageDraw
    pipe = AutoPipelineForInpainting.from_pretrained(
        str(model_dir), torch_dtype=torch.float16, variant='fp16',
        local_files_only=True, use_safetensors=True,
    )
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()
    image = Image.new('RGB', (1024, 1024), (110, 110, 110))
    mask = Image.new('L', image.size)
    ImageDraw.Draw(mask).line([(300, 400), (700, 550)], fill=255, width=24)
    torch.cuda.reset_peak_memory_stats()
    result = pipe(prompt='a thin realistic scratch on an industrial metal surface',
                  image=image, mask_image=mask, strength=.99, num_inference_steps=20,
                  generator=torch.Generator(device='cpu').manual_seed(0)).images[0]
    torch.cuda.synchronize()
    output = ROOT / 'outputs/gpu-smoke.png'
    output.parent.mkdir(exist_ok=True)
    result.save(output)
    return {'status': 'PASS', 'image': str(output),
            'peak_allocated_gib': round(torch.cuda.max_memory_allocated()/1024**3, 2),
            'note': 'Synthetic hardware smoke only; F971 visual acceptance still required.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-cuda', action='store_true')
    parser.add_argument('--inference-smoke', action='store_true')
    parser.add_argument('--model-dir', type=Path, default=ROOT/'data/models/sdxl-inpaint')
    args = parser.parse_args()
    report = inspect_gpu()
    if args.inference_smoke:
        if report['status'] != 'PASS':
            report['inference'] = {'status': 'NOT_RUN', 'message': 'CUDA kernel check did not pass.'}
        else:
            try:
                report['inference'] = inference_smoke(args.model_dir)
            except Exception as error:
                report['inference'] = {'status': 'FAIL', 'message': str(error)}
    save_json(ROOT/'reports/gpu.json', report)
    print(json.dumps(report, indent=2))
    if (args.require_cuda and report['status'] != 'PASS') or (
        args.inference_smoke and report.get('inference', {}).get('status') != 'PASS'
    ):
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
