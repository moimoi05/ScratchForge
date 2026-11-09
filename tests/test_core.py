import io,json
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from scratchforge.dataset import *
from backend.main import app,render,Request,Placement,Brush,library,OUT

def test_copy_collision_and_idempotence(tmp_path):
 a=tmp_path/'a';b=tmp_path/'b';a.write_bytes(b'original');copy_checked(a,b);copy_checked(a,b);assert b.read_bytes()==b'original'
 a.write_bytes(b'changed')
 with pytest.raises(ValueError):copy_checked(a,b)
 assert b.read_bytes()==b'original'
@pytest.mark.parametrize('seg',[[[1,2,3]],[[0,0,1,1,2,2]],[[0,0,999,0,1,2]],{}])
def test_invalid_polygon(seg): assert not valid_polygons(seg,20,20)
def test_crop_mask_coordinate():
 seg=[[300,300,310,300,310,700,300,700]];box=crop_box(seg,(1000,1000));x,y,x2,y2=box;assert x2-x>=256 and y2-y>=432
 local=translate(seg,x,y);m=mask_for(local,(x2-x,y2-y));assert m.getpixel((305-x,500-y))==255
 assert translate(local,-x,-y)==seg
 assert crop_box([[0,0,10,0,10,10]],(100,100),512)==(0,0,100,100)
def test_train_library_original_pixel_exact():
 c=json.loads((ROOT/'data/raw/F971/NG/annotations/coco.json').read_text()); anns={a['id']:a for a in c['annotations']}
 for s in library():
  assert s['split']=='train' and s['id'].startswith('original-');x,y,x2,y2=s['roi'];a=anns[s['source_annotation_id']]
  assert s['segmentation']==translate(a['segmentation'],x,y)
  with Image.open(ROOT/'data/raw/F971/NG/images'/s['source_image']) as src,Image.open(ROOT/'data/library'/s['rgb']) as roi,Image.open(ROOT/'data/library'/s['mask']) as m:
   assert np.array_equal(np.array(src.convert('RGB').crop((x,y,x2,y2))),np.array(roi));assert np.array_equal(np.array(m),np.array(mask_for(s['segmentation'],roi.size)))
def test_multi_stamp_unchanged_outside_mask():
 im=Image.new('RGB',(700,600),(23,41,57));s=library()[0]
 req=Request(image_id='a'*32,placements=[Placement(stamp_id=s['id'],x=180,y=200),Placement(stamp_id=s['id'],x=450,y=350,rotation=37,scale=.7,flip=True)],brushes=[Brush(points=[25,25,100,90],width=7)])
 rgb,m,instances=render(im,req);mask=np.array(m)>0;assert len(instances)==3 and mask.any();assert np.array_equal(np.array(im)[~mask],np.array(rgb)[~mask])
def test_api_export_and_validation():
 client=TestClient(app);assert client.get('/api/health').json()['trained'] is False
 assert client.post('/api/upload',files={'file':('x',b'invalid')}).status_code==400
 b=io.BytesIO();Image.new('RGB',(600,500),(30,40,50)).save(b,format='PNG')
 up=client.post('/api/upload',files={'file':('ok.png',b.getvalue(),'image/png')});assert up.status_code==200
 sid=library()[0]['id'];r=client.post('/api/generate',json={'image_id':up.json()['id'],'placements':[{'stamp_id':sid,'x':180,'y':180},{'stamp_id':sid,'x':400,'y':300,'rotation':25,'scale':.5}]});assert r.status_code==200,r.text
 assert client.get(r.json()['export']).status_code==200
 c=client.get(r.json()['annotation']).json();assert len(c['annotations'])==2
 for a in c['annotations']: assert sum(a['segmentation']['counts'])==600*500
 assert client.post('/api/generate',json={'image_id':'../../bad'}).status_code==422

def test_all_original_audit_instances():
 rows=json.loads((ROOT/'data/roi_audit/index.json').read_text()); assert len(rows)==245
 c=json.loads((ROOT/'data/raw/F971/NG/annotations/coco.json').read_text());anns={a['id']:a for a in c['annotations']}
 for r in rows:
  x,y,x2,y2=r['roi'];a=anns[r['annotation_id']]
  with Image.open(ROOT/'data/raw/F971/NG/images'/r['source_image']) as src,Image.open(ROOT/'data/roi_audit'/r['rgb']) as rgb,Image.open(ROOT/'data/roi_audit'/r['mask']) as m:
   assert np.array_equal(np.array(src.convert('RGB').crop((x,y,x2,y2))),np.array(rgb))
   assert np.array_equal(np.array(m),np.array(mask_for(translate(a['segmentation'],x,y),rgb.size)))
