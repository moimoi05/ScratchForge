import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scratchforge.dataset import ROOT,save_json
import torch
r={'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'available':torch.cuda.is_available()}
if r['available']:
 r.update(device=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),architectures=torch.cuda.get_arch_list())
 x=torch.randn(256,256,device='cuda'); y=x@x; torch.cuda.synchronize(); r['kernel_pass']=bool(torch.isfinite(y).all().item())
save_json(ROOT/'reports/gpu.json',r); print(r)
