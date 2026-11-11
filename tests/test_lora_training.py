import numpy as np
import pytest
from PIL import Image

from scratchforge.lora_training import inpainting_input, masked_noise_loss, prepare_stamp


def test_resize_preserves_complete_instance_and_binary_mask():
    rgb = Image.new('RGB', (160, 20), (10, 20, 30))
    mask = Image.new('L', rgb.size)
    mask.paste(255, (0, 8, 160, 12))
    image, binary = prepare_stamp(rgb, mask, 64)
    assert image.size == binary.size == (64, 64)
    assert set(np.unique(binary)) <= {0, 255}
    assert binary.getbbox()[0] == 0 and binary.getbbox()[2] == 64


def test_inpainting_is_nine_channels_and_masked_loss_focuses_scratch():
    import torch
    noisy = torch.zeros((1, 4, 8, 8))
    mask = torch.zeros((1, 1, 64, 64)); mask[:, :, :32] = 1
    masked = torch.ones_like(noisy)
    combined = inpainting_input(noisy, mask, masked)
    assert combined.shape == (1, 9, 8, 8)
    assert torch.all(combined[:, 5:] == 1)
    prediction = torch.zeros_like(noisy, requires_grad=True)
    target = torch.ones_like(noisy)
    loss = masked_noise_loss(prediction, target, combined[:, 4:5])
    loss.backward()
    assert prediction.grad[:, :, :4].abs().mean() > prediction.grad[:, :, 4:].abs().mean()
    assert torch.isfinite(loss)
    with pytest.raises(ValueError, match='finite'):
        masked_noise_loss(torch.full_like(noisy, float('nan')), target, combined[:, 4:5])


def test_prepare_rejects_empty_and_mismatched_masks():
    with pytest.raises(ValueError, match='dimensions'):
        prepare_stamp(Image.new('RGB', (20, 10)), Image.new('L', (10, 10)), 64)
    with pytest.raises(ValueError, match='empty'):
        prepare_stamp(Image.new('RGB', (20, 10)), Image.new('L', (20, 10)), 64)


def test_real_cpu_lora_optimizer_changes_adapter_only():
    import torch
    from diffusers import DDPMScheduler, UNet2DConditionModel
    from peft import LoraConfig
    from scratchforge.lora_training import _optimize

    torch.manual_seed(19)
    torch.set_num_threads(1)
    unet = UNet2DConditionModel(sample_size=8, in_channels=9, out_channels=4,
        layers_per_block=1, block_out_channels=(8, 16), norm_num_groups=4,
        down_block_types=('CrossAttnDownBlock2D', 'DownBlock2D'),
        up_block_types=('UpBlock2D', 'CrossAttnUpBlock2D'), cross_attention_dim=8, attention_head_dim=4)
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=['to_q', 'to_k', 'to_v', 'to_out.0']))
    before = {name: value.detach().clone() for name, value in unet.named_parameters()}
    parameters = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=.01)
    mask = torch.ones((1, 1, 64, 64))
    cache = [(torch.randn(1, 4, 8, 8), mask, torch.randn(1, 4, 8, 8))]
    events = []
    losses = _optimize(unet, cache, optimizer, DDPMScheduler(num_train_timesteps=10),
        torch.amp.GradScaler('cpu', enabled=False), torch.Generator().manual_seed(11),
        torch.randn(1, 4, 8), None, 2, lambda percent, **kw: events.append(kw), lambda: False, device='cpu')
    assert len(losses) == 2 and events[-1]['step'] == 2
    assert any(not torch.equal(value, before[name]) for name, value in unet.named_parameters() if value.requires_grad)
    assert all(torch.equal(value, before[name]) for name, value in unet.named_parameters() if not value.requires_grad)


def test_training_pipeline_prepares_embeddings_latents_and_saves_lora(tmp_path):
    import torch
    from diffusers import DDPMScheduler, UNet2DConditionModel
    from safetensors.torch import save_file
    from types import SimpleNamespace
    from scratchforge.lora_training import _train_pipeline

    torch.set_num_threads(1)
    image = Image.new('RGB', (40, 16), (110, 111, 112))
    mask = Image.new('L', image.size); mask.paste(255, (2, 6, 38, 10))
    image.save(tmp_path/'image.tif'); mask.save(tmp_path/'mask.png')
    class VAE(torch.nn.Module):
        config = SimpleNamespace(scaling_factor=.18215)
        def encode(self, pixels):
            value = torch.nn.functional.avg_pool2d(pixels, 8)
            latent = torch.cat([value, value[:, :1]], dim=1)
            return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))
    unet = UNet2DConditionModel(sample_size=8, in_channels=9, out_channels=4,
        layers_per_block=1, block_out_channels=(8, 16), norm_num_groups=4,
        down_block_types=('CrossAttnDownBlock2D', 'DownBlock2D'),
        up_block_types=('UpBlock2D', 'CrossAttnUpBlock2D'), cross_attention_dim=8, attention_head_dim=4)
    pipeline = SimpleNamespace(unet=unet, vae=VAE(), text_encoder=torch.nn.Linear(1, 1),
        scheduler=DDPMScheduler(num_train_timesteps=10),
        encode_prompt=lambda *args, **kwargs: (torch.ones(1, 4, 8), None),
        save_lora_weights=lambda directory, **kwargs: save_file(kwargs['unet_lora_layers'],
            str(tmp_path/kwargs['weight_name'])))
    request = {'seed': 2, 'base_model': 'sd15', 'steps': 1, 'resolution': 64,
               'library_directory': str(tmp_path), 'sources': [{'rgb': 'image.tif', 'mask': 'mask.png'}]}
    events = []
    result = _train_pipeline(pipeline, request, tmp_path, lambda p, **k: events.append(k), lambda: False, device='cpu')
    assert result['steps'] == 1 and result['resolution'] == 64
    assert (tmp_path/'pytorch_lora_weights.safetensors').is_file()
    assert any(event['phase'] == 'encoding' for event in events)


def test_train_worker_persists_real_progress_and_failure(tmp_path, monkeypatch):
    import json
    from scratchforge import lora_worker

    request = tmp_path/'request.json'; request.write_text('{}')
    def train(payload, output, progress, cancelled):
        progress(25, phase='training')
        return {'final_loss': .3}
    monkeypatch.setattr(lora_worker, 'run_training', train)
    assert lora_worker.main(request) == 0
    assert json.loads((tmp_path/'progress.json').read_text())['progress'] == 25
    assert json.loads((tmp_path/'completed.json').read_text())['metrics']['final_loss'] == .3
    def failure(*args): raise RuntimeError('CUDA out of memory')
    monkeypatch.setattr(lora_worker, 'run_training', failure)
    assert lora_worker.main(request) == 1
    assert 'VRAM' in json.loads((tmp_path/'completed.json').read_text())['error']
