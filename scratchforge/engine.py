"""Stable contracts for future local GAN or mask-guided inpainting adapters."""
from typing import Protocol
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
@dataclass
class GenerationRequest:
 rgb: Image.Image
 mask: Image.Image
 seed: int=0
class LocalGenerator(Protocol):
 def load(self,checkpoint:Path)->None: ...
 def infer_roi(self,request:GenerationRequest)->Image.Image: ...
class TrainingMonitor(Protocol):
 def report(self,step:int,metrics:dict[str,float])->None: ...
class CheckpointManager(Protocol):
 def save(self,state:dict,path:Path)->None: ...
 def load(self,path:Path)->dict: ...
class QualityEvaluator(Protocol):
 def evaluate(self,original:Image.Image,generated:Image.Image,mask:Image.Image)->dict[str,float]: ...
def composite(original,generated,mask):
 if original.size!=generated.size or original.size!=mask.size: raise ValueError('Dimension mismatch')
 return Image.composite(generated.convert('RGB'),original.convert('RGB'),mask.convert('L'))
