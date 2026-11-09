import hashlib, json, shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def copy_checked(src,dst):
 dst=Path(dst); dst.parent.mkdir(parents=True,exist_ok=True)
 digest=sha(src)
 if dst.exists():
  if sha(dst)!=digest: raise ValueError(f'Conflicting destination: {dst}')
 else:
  temp=dst.with_suffix(dst.suffix+'.partial'); shutil.copyfile(src,temp)
  if sha(temp)!=digest: raise ValueError('Copy checksum failure')
  temp.replace(dst)
 return digest
def save_json(p,obj):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(obj,indent=2),encoding='utf-8')
def valid_polygons(seg,w,h):
 if not isinstance(seg,list) or not seg: return False
 for poly in seg:
  if not isinstance(poly,list) or len(poly)<6 or len(poly)%2: return False
  a=np.asarray(poly,dtype=float).reshape(-1,2)
  if not np.isfinite(a).all() or (a<0).any() or (a[:,0]>w).any() or (a[:,1]>h).any(): return False
  if abs(np.dot(a[:,0],np.roll(a[:,1],1))-np.dot(a[:,1],np.roll(a[:,0],1)))<1: return False
 return True
def mask_for(seg,size):
 if not valid_polygons(seg,*size): raise ValueError('Invalid polygon')
 m=Image.new('L',size)
 for p in seg: ImageDraw.Draw(m).polygon(list(zip(p[::2],p[1::2])),fill=255)
 return m
def crop_box(seg,size,preferred=256):
 a=np.concatenate([np.array(p).reshape(-1,2) for p in seg]); lo=np.floor(a.min(0)-16).astype(int); hi=np.ceil(a.max(0)+16).astype(int)
 extent=hi-lo; wh=np.maximum(extent,preferred); wh=np.minimum(wh,size); xy=np.maximum(0,np.minimum((lo+hi-wh)//2,np.array(size)-wh))
 return tuple(map(int,[*xy,*(xy+wh)]))
def translate(seg,x,y): return [[v-(x if i%2==0 else y) for i,v in enumerate(p)] for p in seg]
