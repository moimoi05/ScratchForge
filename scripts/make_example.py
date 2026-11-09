import sys,json,io
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from backend.main import app,library
from scratchforge.dataset import *
c=TestClient(app); rows=json.loads((ROOT/'data/manifest.json').read_text()); r=next(r for r in rows if r['domain']=='clean' and r['split']=='train'); source=ROOT/r['source_path']
with Image.open(source) as im:
 b=io.BytesIO();im.convert('RGB').save(b,format='PNG');w,h=im.size
up=c.post('/api/upload',files={'file':('ok.png',b.getvalue(),'image/png')}).json(); stamps=library()
request={'image_id':up['id'],'placements':[{'stamp_id':stamps[0]['id'],'x':w*.45,'y':h*.3,'scale':1,'rotation':0},{'stamp_id':stamps[6]['id'],'x':w*.65,'y':h*.4,'scale':1.2,'rotation':30}]}
r=c.post('/api/generate',json=request);r.raise_for_status();result=r.json()
with Image.open(source) as original,Image.open(ROOT/'outputs'/Path(result['preview']).name) as generated,Image.open(ROOT/'outputs'/Path(result['mask']).name) as mask:
 outside=np.array(mask)==0;assert np.array_equal(np.array(original.convert('RGB'))[outside],np.array(generated)[outside])
save_json(ROOT/'reports/example.json',dict(source=str(source),request=request,result=result,outside_mask_exact=True));print(result)
