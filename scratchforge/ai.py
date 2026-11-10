"""Lazy, offline-only SDXL adapter. CUDA validation requires the real target GPU."""
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from scratchforge.dataset import ROOT

MODEL_ID = 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1'
PROMPT = 'photorealistic subtle industrial surface scratch, matching the surrounding material, lighting, color and texture, precise fine abrasion'
NEGATIVE = 'drawing, star, text, logo, paint, decoration, blurry, artificial, cartoon'


class SDXLGenerator:
    def __init__(self, root=ROOT):
        self.directory = Path(os.environ.get('SCRATCHFORGE_MODEL_DIR', Path(root)/'data/models/sdxl-inpaint'))
        self.pipeline = None

    def metadata(self):
        path = self.directory / 'model-provenance.json'
        recorded = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        return {'model_id': MODEL_ID, 'model_available': (self.directory/'model_index.json').is_file(),
                'revision': recorded.get('resolved_revision') or recorded.get('revision') or 'local-unversioned',
                'fine_tuned_f971': False}

    def _load(self):
        if not self.metadata()['model_available']:
            raise RuntimeError('SDXL model missing. Run scripts/download_model.py on the GPU machine.')
        try:
            import torch
            from diffusers import AutoPipelineForInpainting
        except ImportError as error:
            raise RuntimeError('AI dependencies missing. Run setup.ps1 -Gpu.') from error
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU unavailable. Run the GPU environment check on the NVIDIA machine.')
        pipeline = AutoPipelineForInpainting.from_pretrained(str(self.directory), torch_dtype=torch.float16,
                                                            variant='fp16', local_files_only=True, use_safetensors=True)
        pipeline.enable_model_cpu_offload(); pipeline.enable_vae_tiling()
        self.pipeline = pipeline

    def __call__(self, rgb, mask, seed, strength):
        if self.pipeline is None:
            self._load()
        import torch
        # One complete context region is resized with aspect preserved and edge padding.
        ratio = 1024 / max(rgb.size)
        size = (max(1, round(rgb.width*ratio)), max(1, round(rgb.height*ratio)))
        resized = np.asarray(rgb.resize(size, Image.Resampling.LANCZOS))
        image = Image.fromarray(np.pad(resized, ((0, 1024-size[1]), (0, 1024-size[0]), (0, 0)), mode='edge'))
        small_mask = mask.resize(size, Image.Resampling.NEAREST)
        padded_mask = Image.new('L', (1024, 1024)); padded_mask.paste(small_mask, (0, 0))
        try:
            generated = self.pipeline(prompt=PROMPT, negative_prompt=NEGATIVE, image=image,
                                      mask_image=padded_mask, strength=strength, num_inference_steps=30,
                                      guidance_scale=6, generator=torch.Generator(device='cpu').manual_seed(seed)).images[0]
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            raise RuntimeError('GPU ran out of VRAM. Close other GPU apps and retry; no baseline substitution was made.') from error
        return generated.crop((0, 0, *size)).resize(rgb.size, Image.Resampling.LANCZOS)
