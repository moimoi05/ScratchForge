"""All original polygons for review, partitioned by verified split. Never training input."""
import sys,json,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scratchforge.dataset import *
def build():
 c=json.loads((ROOT/'data/raw/F971/NG/annotations/coco.json').read_text());images={i['id']:i for i in c['images']}
 rows=list(csv.DictReader((ROOT/'data/reference/source_split_leakage_check.csv').open(encoding='utf-8-sig')));splits={r['source_image']:{'validation':'val'}.get(r['source_split_assignment'],r['source_split_assignment']) for r in rows if r['leakage']=='NO'}
 out=ROOT/'data/roi_audit';out.mkdir(exist_ok=True);entries=[];errors=[]
 for a in c['annotations']:
  i=images[a['image_id']];split=splits.get(i['file_name'],'HOLD');seg=a['segmentation']
  try:
   with Image.open(ROOT/'data/raw/F971/NG/images'/i['file_name']) as im:
    if im.size!=(i['width'],i['height']) or not valid_polygons(seg,*im.size): raise ValueError('Invalid COCO mapping/polygon')
    pts=np.concatenate([np.array(p).reshape(-1,2) for p in seg]);boundary=(pts.min(0)<=0).any() or (pts.max(0)>=np.array(im.size)-1).any()
    if boundary: split='HOLD'
    box=crop_box(seg,im.size,512 if np.ptp(pts,axis=0).max()>224 else 256);x,y,x2,y2=box;folder=out/split;folder.mkdir(exist_ok=True)
    rgb=im.convert('RGB').crop(box);mask=mask_for(translate(seg,x,y),rgb.size);rgb.save(folder/f"{a['id']}.png");mask.save(folder/f"{a['id']}-mask.png")
    entries.append(dict(annotation_id=a['id'],source_image=i['file_name'],split=split,roi=box,rgb=f"{split}/{a['id']}.png",mask=f"{split}/{a['id']}-mask.png",review='PENDING',usage='audit_only_not_training'))
  except ValueError as e: errors.append(dict(annotation=a['id'],error=str(e)))
 save_json(out/'index.json',entries)
 for split in ('train','val','test','HOLD'):
  subset=[r for r in entries if r['split']==split]
  for page in range(0,len(subset),16):
   sheet=Image.new('RGB',(1200,760),'#222222');d=ImageDraw.Draw(sheet)
   for k,r in enumerate(subset[page:page+16]):
    rgb=Image.open(out/r['rgb']);m=Image.open(out/r['mask']);rgb=Image.blend(rgb,Image.composite(Image.new('RGB',rgb.size,'orange'),rgb,m),.5);rgb.thumbnail((190,155));m.thumbnail((95,155));x=(k%4)*300;y=(k//4)*190;sheet.paste(rgb,(x,y));sheet.paste(m,(x+195,y));d.text((x,y+160),str(r['annotation_id'])+' '+split,fill='white')
   sheet.save(out/f'{split}-{page//16+1}.jpg')
 save_json(ROOT/'reports/roi_audit.json',dict(total=len(entries),errors=errors,counts={s:sum(r['split']==s for r in entries) for s in ('train','val','test','HOLD')},policy='review only; no training consumption'))
 print('Audit ROIs',len(entries),'errors',len(errors))
if __name__=='__main__':build()
