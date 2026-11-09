import io,json,uuid,zipfile,base64,math
import numpy as np
from PIL import Image,ImageDraw
from fastapi import FastAPI,UploadFile,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
from scratchforge.dataset import ROOT,save_json
from scratchforge.engine import composite
app=FastAPI(title='ScratchForge baseline API')
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173','http://127.0.0.1:5173'],allow_methods=['*'],allow_headers=['*'])
OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
app.mount('/files',StaticFiles(directory=OUT),name='files')
app.mount('/library-files',StaticFiles(directory=ROOT/'data/library'),name='library')
def library(): return json.loads((ROOT/'data/library/index.json').read_text())
@app.get('/api/health')
def health(): return {'engine':'baseline compositing','trained':False}
@app.get('/api/library')
def get_library(): return library()
@app.get('/api/dataset')
def dataset(): return json.loads((ROOT/'reports/verification.json').read_text())
@app.get('/api/training')
def training(): return {'status':'not trained','checkpoints':[],'message':'Engine interfaces ready; training requires separate explicit action.'}
@app.get('/api/exports')
def exports(): return [p.name for p in OUT.glob('*.zip')]
@app.post('/api/upload')
async def upload(file:UploadFile):
 content=await file.read(40*1024*1024+1)
 if len(content)>40*1024*1024: raise HTTPException(413,'Max 40 MiB')
 try:
  im=Image.open(io.BytesIO(content))
  if im.mode!='RGB': raise ValueError('Upload an RGB image')
  if im.width*im.height>50_000_000: raise ValueError('Max 50 megapixels')
  token=uuid.uuid4().hex; im.save(OUT/f'{token}-input.png')
  return dict(id=token,width=im.width,height=im.height,url=f'/files/{token}-input.png')
 except Exception as e: raise HTTPException(400,str(e))
class Placement(BaseModel):
 stamp_id:str
 x:float
 y:float
 scale:float=Field(default=1,ge=.05,le=8)
 rotation:float=0
 flip:bool=False
class Brush(BaseModel):
 points:list[float]
 width:float=Field(default=12,ge=1,le=512)
class Request(BaseModel):
 image_id:str=Field(pattern='^[a-f0-9]{32}$')
 placements:list[Placement]=Field(default_factory=list,max_length=100)
 brushes:list[Brush]=Field(default_factory=list,max_length=100)
def render(im,req):
 w,h=im.size; mask=Image.new('L',im.size); result=im.copy(); instances=[]; lookup={s['id']:s for s in library()}
 for p in req.placements:
  if p.stamp_id not in lookup: raise ValueError('Unknown stamp')
  if not all(math.isfinite(v) for v in (p.x,p.y,p.rotation)): raise ValueError('Nonfinite coordinates')
  s=lookup[p.stamp_id]; rgb=Image.open(ROOT/'data/library'/s['rgb']).convert('RGBA'); m=Image.open(ROOT/'data/library'/s['mask']); rgb.putalpha(m)
  # Inverse affine samples directly into original-image coordinates; x/y is stamp center.
  theta=math.radians(p.rotation); c=math.cos(theta); sn=math.sin(theta); flip=-1 if p.flip else 1
  a=flip*c/p.scale; b=flip*sn/p.scale; d=-sn/p.scale; e=c/p.scale
  warped=rgb.transform(im.size,Image.Transform.AFFINE,(a,b,rgb.width/2-a*p.x-b*p.y,d,e,rgb.height/2-d*p.x-e*p.y),Image.Resampling.BILINEAR)
  binary=warped.getchannel('A').point(lambda v:255 if v>=128 else 0); result.paste(warped.convert('RGB'),(0,0),binary); mask=np.maximum(np.array(mask),np.array(binary)); mask=Image.fromarray(mask)
  instances.append(binary)
 for b in req.brushes:
  if len(b.points)<4 or len(b.points)%2 or not all(math.isfinite(v) for v in b.points): raise ValueError('Invalid brush points')
  m=Image.new('L',im.size); ImageDraw.Draw(m).line(list(zip(b.points[::2],b.points[1::2])),fill=255,width=round(b.width),joint='curve'); result.paste((180,155,115),(0,0,w,h),m); mask=Image.fromarray(np.maximum(np.array(mask),np.array(m))); instances.append(m)
 return composite(im,result,mask),mask,instances
@app.post('/api/generate')
def generate(req:Request):
 path=OUT/f'{req.image_id}-input.png'
 if not path.exists(): raise HTTPException(404,'Image missing')
 try: rgb,mask,instances=render(Image.open(path).convert('RGB'),req)
 except ValueError as e: raise HTTPException(422,str(e))
 token=uuid.uuid4().hex; rgb.save(OUT/f'{token}.png'); mask.save(OUT/f'{token}-mask.png'); anns=[]
 # Exact COCO uncompressed RLE, preserving every edited pixel including holes.
 for m in instances:
  arr=(np.array(m)>0).astype(np.uint8); counts=[]; last=0; count=0
  for v in arr.flatten(order='F'):
   if v==last: count+=1
   else: counts.append(count); count=1; last=int(v)
  counts.append(count); ys,xs=np.where(arr)
  if len(xs): anns.append(dict(id=len(anns)+1,image_id=1,category_id=1,segmentation={'size':[rgb.height,rgb.width],'counts':counts},area=int(arr.sum()),bbox=[int(xs.min()),int(ys.min()),int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)],iscrowd=0))
 annotation=dict(info={'engine':'baseline compositing; no trained AI'},images=[dict(id=1,file_name=f'{token}.png',width=rgb.width,height=rgb.height)],categories=[dict(id=1,name='scratch')],annotations=anns)
 save_json(OUT/f'{token}.json',annotation); save_json(OUT/f'{token}-scene.json',req.model_dump())
 with zipfile.ZipFile(OUT/f'{token}.zip','w',zipfile.ZIP_DEFLATED) as z:
  for suffix in ('.png','-mask.png','.json','-scene.json'): z.write(OUT/f'{token}{suffix}',f'{token}{suffix}')
 return dict(preview=f'/files/{token}.png',mask=f'/files/{token}-mask.png',export=f'/files/{token}.zip',annotation=f'/files/{token}.json',engine='baseline')
