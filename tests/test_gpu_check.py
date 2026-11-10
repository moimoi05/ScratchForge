from types import SimpleNamespace

from scripts.check_gpu import inspect_gpu


def test_no_cuda_is_not_hardware_pass():
    torch = SimpleNamespace(__version__='test', version=SimpleNamespace(cuda=None),
                            cuda=SimpleNamespace(is_available=lambda: False))
    result = inspect_gpu(torch)
    assert result['status'] == 'NOT_AVAILABLE'
    assert result['kernel_pass'] is False


def test_cuda_requires_real_kernel_execution():
    class Tensor:
        def __matmul__(self, other):
            return self

    cuda = SimpleNamespace(is_available=lambda: True,
                           get_device_name=lambda: 'Test GPU',
                           get_device_capability=lambda: (12, 0),
                           get_arch_list=lambda: ['sm_120'],
                           get_device_properties=lambda device: SimpleNamespace(total_memory=16*1024**3),
                           synchronize=lambda: None)
    torch = SimpleNamespace(__version__='test', version=SimpleNamespace(cuda='12.8'), cuda=cuda,
                            randn=lambda *args, **kwargs: Tensor(),
                            isfinite=lambda tensor: SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: True)))
    result = inspect_gpu(torch)
    assert result['status'] == 'PASS'
    assert result['kernel_pass'] is True
    assert result['vram_gib'] == 16
