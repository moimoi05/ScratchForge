"""Lazy, offline-only SDXL adapter. CUDA validation requires the real target GPU."""
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from scratchforge.dataset import ROOT

MODEL_ID = 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1'
PROMPT = 'photorealistic subtle industrial surface scratch, matching the surrounding material, lighting, color and texture, precise fine abrasion'
NEGATIVE = 'drawing, star, text, logo, paint, decoration, blurry, artificial, cartoon'


class SDXLGenerator:
    model_id = MODEL_ID
    directory_name = 'sdxl-inpaint'
    directory_env = 'SCRATCHFORGE_MODEL_DIR'
    canvas_size = 1024
    requires_cuda = True

    def __init__(self, root=ROOT):
        self.directory = Path(os.environ.get(self.directory_env, Path(root)/'data/models'/self.directory_name))
        self.pipeline = None

    def metadata(self):
        path = self.directory / 'model-provenance.json'
        recorded = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        return {'model_id': self.model_id, 'model_available': (self.directory/'model_index.json').is_file(),
                'revision': recorded.get('resolved_revision') or recorded.get('revision') or 'local-unversioned',
                'fine_tuned_f971': False}

    def _load(self):
        if not self.metadata()['model_available']:
            raise RuntimeError(f'{self.directory_name} model missing. Run scripts/download_model.py --model {self.directory_name.split("-")[0]}.')
        try:
            import torch
            from diffusers import AutoPipelineForInpainting
        except ImportError as error:
            raise RuntimeError('AI dependencies missing. Run setup.ps1 -Gpu.') from error
        cuda = torch.cuda.is_available()
        if self.requires_cuda and not cuda:
            raise RuntimeError('CUDA GPU unavailable. Run the GPU environment check on the NVIDIA machine.')
        dtype = torch.float16 if cuda else torch.float32
        pipeline = AutoPipelineForInpainting.from_pretrained(str(self.directory), torch_dtype=dtype,
                                                            variant='fp16', local_files_only=True, use_safetensors=True)
        if cuda:
            if os.environ.get('SCRATCHFORGE_CPU_OFFLOAD') == '1':
                pipeline.enable_model_cpu_offload()
            else:
                pipeline.to('cuda')
        else:
            pipeline.to('cpu')
            pipeline.enable_attention_slicing()
        vae = getattr(pipeline, 'vae', None)
        if vae is not None and hasattr(vae, 'enable_tiling'):
            vae.enable_tiling()
        self.pipeline = pipeline

    def __call__(self, rgb, mask, seed, strength):
        if self.pipeline is None:
            self._load()
        import torch
        # One complete context region is resized with aspect preserved and edge padding.
        canvas = self.canvas_size
        ratio = canvas / max(rgb.size)
        size = (max(1, round(rgb.width*ratio)), max(1, round(rgb.height*ratio)))
        resized = np.asarray(rgb.resize(size, Image.Resampling.LANCZOS))
        image = Image.fromarray(np.pad(resized, ((0, canvas-size[1]), (0, canvas-size[0]), (0, 0)), mode='edge'))
        small_mask = mask.resize(size, Image.Resampling.NEAREST)
        padded_mask = Image.new('L', (canvas, canvas)); padded_mask.paste(small_mask, (0, 0))
        try:
            generated = self.pipeline(prompt=PROMPT, negative_prompt=NEGATIVE, image=image,
                                      mask_image=padded_mask, strength=strength, num_inference_steps=30,
                                      guidance_scale=6, generator=torch.Generator(device='cpu').manual_seed(seed)).images[0]
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            raise RuntimeError('GPU ran out of VRAM. Close other GPU apps and retry; no baseline substitution was made.') from error
        return generated.crop((0, 0, *size)).resize(rgb.size, Image.Resampling.LANCZOS)


class SD15Generator(SDXLGenerator):
    model_id = 'stable-diffusion-v1-5/stable-diffusion-inpainting'
    directory_name = 'sd15-inpaint'
    directory_env = 'SCRATCHFORGE_SD15_MODEL_DIR'
    canvas_size = 512
    requires_cuda = False


def procedural_scratch(rgb, mask, seed, strength):
    """Native-pixel textured abrasion; deterministic and independent of Torch/CUDA."""
    values = np.asarray(rgb, dtype=np.float32)
    support = np.asarray(mask, dtype=np.float32)/255
    rng = np.random.default_rng(seed)
    grain = rng.normal(0, 6, support.shape).astype(np.float32)
    core = np.asarray(mask.filter(ImageFilter.MinFilter(3)), dtype=np.float32)/255
    ridge = (support-core)*13
    contrast = 20+strength*25
    # Retain original surface variation and add fine bright/dark abrasion contrast.
    changed = np.clip(values*.87 + (contrast+grain+ridge)[..., None], 0, 255)
    output = values*(1-support[..., None]) + changed*support[..., None]
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8))
