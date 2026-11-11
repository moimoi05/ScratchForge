"""Nine-channel inpainting LoRA training; imports Torch/PEFT only when requested."""
import gc
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scratchforge.ai import PROMPT


def prepare_stamp(rgb, mask, resolution):
    if rgb.size != mask.size:
        raise ValueError('Training RGB/mask dimensions differ')
    if not mask.getbbox():
        raise ValueError('Training mask is empty')
    ratio = resolution / max(rgb.size)
    size = tuple(max(1, round(value * ratio)) for value in rgb.size)
    pixels = np.asarray(rgb.convert('RGB').resize(size, Image.Resampling.LANCZOS))
    padded = np.pad(pixels, ((0, resolution-size[1]), (0, resolution-size[0]), (0, 0)), mode='edge')
    binary = Image.new('L', (resolution, resolution))
    binary.paste(mask.convert('L').resize(size, Image.Resampling.NEAREST).point(lambda p: 255 if p >= 128 else 0))
    return Image.fromarray(padded), binary


def inpainting_input(noisy_latents, mask, masked_latents):
    import torch
    import torch.nn.functional as functional

    if noisy_latents.shape != masked_latents.shape or noisy_latents.shape[1] != 4 or mask.shape[1] != 1:
        raise ValueError('Inpainting requires 4 latent + 1 mask + 4 masked-image channels')
    resized = functional.adaptive_max_pool2d(mask, noisy_latents.shape[-2:])
    return torch.cat((noisy_latents, resized.to(noisy_latents.dtype), masked_latents), dim=1)


def masked_noise_loss(prediction, target, mask):
    import torch
    import torch.nn.functional as functional

    if prediction.shape != target.shape:
        raise ValueError('Noise prediction dimensions differ from target')
    selected = functional.adaptive_max_pool2d(mask.float(), prediction.shape[-2:])
    weights = .05 + .95 * selected
    loss = ((prediction.float()-target.float()).square()*weights).sum() / (weights.sum()*prediction.shape[1])
    if not torch.isfinite(loss):
        raise ValueError('Training loss is not finite; no checkpoint was published')
    return loss


def _prompt_embeddings(pipeline, base_model):
    import torch

    with torch.no_grad():
        result = pipeline.encode_prompt(PROMPT, device=torch.device('cpu'), num_images_per_prompt=1,
                                        do_classifier_free_guidance=False)
    return result[0].detach().cpu(), result[2].detach().cpu() if base_model == 'sdxl' else None


def _cache_latents(pipeline, request, progress, cancelled, device='cuda'):
    import torch

    cache = []
    resolution = request['resolution']
    pipeline.vae.to(device=device, dtype=torch.float32)
    for index, source in enumerate(request['sources']):
        if cancelled():
            raise RuntimeError('LoRA training cancelled')
        directory = Path(request['library_directory'])
        with Image.open(directory/source['rgb']) as rgb, Image.open(directory/source['mask']) as mask:
            image, binary = prepare_stamp(rgb, mask, resolution)
        pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).float().to(device)/127.5-1
        selected = torch.from_numpy(np.array(binary)).unsqueeze(0).unsqueeze(0).float().to(device)/255
        with torch.no_grad():
            latent = pipeline.vae.encode(pixels).latent_dist.mode()*pipeline.vae.config.scaling_factor
            masked = pipeline.vae.encode(pixels*(1-selected)).latent_dist.mode()*pipeline.vae.config.scaling_factor
        if not torch.isfinite(latent).all() or not torch.isfinite(masked).all():
            raise ValueError('VAE produced non-finite training latents')
        cache.append((latent.detach().cpu(), selected.detach().cpu(), masked.detach().cpu()))
        progress(5*(index+1)/len(request['sources']), phase='encoding', completed=index+1,
                 total=len(request['sources']))
    pipeline.vae.to('cpu')
    torch.cuda.empty_cache()
    return cache


def run_training(request, output, progress, cancelled):
    try:
        import torch
        from diffusers import AutoPipelineForInpainting
        import peft  # Verify optional training dependency before loading large weights.
    except ImportError as error:
        raise RuntimeError('LoRA dependencies missing; install requirements-ai.txt on the CUDA machine') from error
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable; LoRA training requires the NVIDIA target machine')
    torch.manual_seed(request['seed'])
    torch.cuda.manual_seed_all(request['seed'])
    np.random.seed(request['seed'])
    pipeline = AutoPipelineForInpainting.from_pretrained(
        request['model_directory'], torch_dtype=torch.float16, variant='fp16',
        local_files_only=True, use_safetensors=True)
    return _train_pipeline(pipeline, request, output, progress, cancelled)


def _train_pipeline(pipeline, request, output, progress, cancelled, device='cuda'):
    import torch
    from diffusers import DDPMScheduler
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig, get_peft_model_state_dict

    dtype = torch.float16 if device == 'cuda' else torch.float32
    if pipeline.unet.config.in_channels != 9:
        raise ValueError('Selected base is not a nine-channel inpainting UNet')
    for module in (pipeline.vae, pipeline.text_encoder, getattr(pipeline, 'text_encoder_2', None), pipeline.unet):
        if module is not None:
            module.requires_grad_(False)
    pipeline.text_encoder.to(dtype=torch.float32)
    if getattr(pipeline, 'text_encoder_2', None) is not None:
        pipeline.text_encoder_2.to(dtype=torch.float32)
    embeddings, pooled = _prompt_embeddings(pipeline, request['base_model'])
    cache = _cache_latents(pipeline, request, progress, cancelled, device)
    pipeline.unet.add_adapter(LoraConfig(r=8, lora_alpha=8, init_lora_weights='gaussian',
                                        target_modules=['to_k', 'to_q', 'to_v', 'to_out.0']))
    pipeline.unet.enable_gradient_checkpointing()
    pipeline.unet.train().to(device)
    parameters = [parameter for parameter in pipeline.unet.parameters() if parameter.requires_grad]
    for parameter in parameters:
        parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    scaler = torch.amp.GradScaler(device, enabled=device == 'cuda')
    generator = torch.Generator(device=device).manual_seed(request['seed'])
    embeddings = embeddings.to(device=device, dtype=dtype)
    added = None
    if pooled is not None:
        resolution = request['resolution']
        added = {'text_embeds': pooled.to(device=device, dtype=dtype),
                 'time_ids': torch.tensor([[resolution, resolution, 0, 0, resolution, resolution]],
                                          device=device, dtype=dtype)}
    losses = _optimize(pipeline.unet, cache, optimizer, scheduler, scaler, generator, embeddings,
                       added, request['steps'], progress, cancelled, device)
    weights = convert_state_dict_to_diffusers(get_peft_model_state_dict(pipeline.unet))
    pipeline.save_lora_weights(str(output), unet_lora_layers=weights,
                               weight_name='pytorch_lora_weights.safetensors', safe_serialization=True)
    del pipeline, cache, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return {'final_loss': losses[-1], 'initial_loss': losses[0], 'steps': request['steps'],
            'rank': 8, 'gradient_accumulation': 4, 'resolution': request['resolution']}


def _optimize(unet, cache, optimizer, scheduler, scaler, generator, embeddings, added, steps, progress, cancelled, device='cuda'):
    import torch

    losses = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        value = 0.0
        for micro in range(4):
            if cancelled():
                raise RuntimeError('LoRA training cancelled')
            dtype = torch.float16 if device == 'cuda' else torch.float32
            index = int(torch.randint(len(cache), (1,), generator=generator, device=device).item())
            latent, mask, masked = (tensor.to(device=device, dtype=dtype) for tensor in cache[index])
            noise = torch.randn(latent.shape, generator=generator, device=device, dtype=latent.dtype)
            timestep = torch.randint(0, scheduler.config.num_train_timesteps, (1,),
                                     generator=generator, device=device).long()
            noisy = scheduler.add_noise(latent, noise, timestep)
            model_input = inpainting_input(noisy, mask, masked)
            with torch.autocast(device, dtype=torch.float16, enabled=device == 'cuda'):
                prediction = unet(model_input, timestep, encoder_hidden_states=embeddings,
                                  added_cond_kwargs=added, return_dict=False)[0]
                if scheduler.config.prediction_type not in {'epsilon', 'v_prediction'}:
                    raise ValueError('Unsupported inpainting scheduler prediction type')
                target = noise if scheduler.config.prediction_type == 'epsilon' else scheduler.get_velocity(latent, noise, timestep)
                loss = masked_noise_loss(prediction, target, mask)/4
            scaler.scale(loss).backward()
            value += float(loss.detach())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in unet.parameters() if p.requires_grad], 1.0,
                                       error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        losses.append(value)
        progress(5+94*(step+1)/steps, phase='training', step=step+1, total=steps, loss=value)
    return losses
