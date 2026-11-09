import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scratchforge.dataset import *
def verify():
 rows=json.loads((ROOT/'data/manifest.json').read_text()); assert len(rows)==114
 assert sum(r['clean_replacement'] for r in rows)==39
 splits={}
 for r in rows:
  assert sha(ROOT/r['path'])==r['sha256']; assert splits.setdefault(r['source_image'],r['split'])==r['split']
  with Image.open(ROOT/r['path']) as im,Image.open(ROOT/r['source_path']) as src:
   x,y=r['crop_offset']; assert im.mode=='RGB' and im.size==(r['width'],r['height'])
   assert np.array_equal(np.asarray(im),np.asarray(src.convert('RGB').crop((x,y,x+im.width,y+im.height))))
 for r in json.loads((ROOT/'data/provenance.json').read_text()): assert sha(ROOT/r['new_path'])==r['sha256'] and sha(r['old_path'])==r['sha256']
 c=json.loads((ROOT/'data/curated/coco.json').read_text()); ims={r['id']:r for r in rows}
 for a in c['annotations']: assert valid_polygons(a['segmentation'],ims[a['image_id']]['width'],ims[a['image_id']]['height'])
 result=dict(status='PASS',patches=114,clean_replacements=39,polygons=len(c['annotations']),checks=['sha256 all raw and curated','RGB source pixel equality','source split integrity','polygon bounds'])
 save_json(ROOT/'reports/verification.json',result); print(result)
if __name__=='__main__': verify()
