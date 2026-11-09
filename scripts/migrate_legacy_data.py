"""Read-only legacy importer; collisions fail closed. Run from repository root."""
import argparse,csv,json,re,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scratchforge.dataset import *
def migrate(source,dest):
 source=Path(source).resolve(); dest=Path(dest).resolve()
 if source==dest or source in dest.parents or dest in source.parents: raise ValueError('Source/destination overlap')
 raw=[p for name in ('F971','S942') for p in (source/name).rglob('*') if p.is_file()]
 required=sum(p.stat().st_size for p in raw if not (dest/'data/raw'/p.relative_to(source)).exists())
 if shutil.disk_usage(dest).free<required+2*1024**3: raise ValueError('Insufficient disk space')
 provenance=[]
 copy_checked(source/'reports/final_patch_review_v2/source_split_leakage_check.csv',dest/'data/reference/source_split_leakage_check.csv')
 for p in raw:
  out=dest/'data/raw'/p.relative_to(source); digest=copy_checked(p,out)
  provenance.append(dict(old_path=str(p),new_path=out.relative_to(dest).as_posix(),sha256=digest,bytes=p.stat().st_size))
 rows=[]
 for split in ('train','val','test'):
  rows+=list(csv.DictReader((source/f'training/manifests/{split}.csv').open(encoding='utf-8-sig')))
 if len(rows)!=114: raise ValueError(f'Expected 114, found {len(rows)}')
 clean={r['final_rgb_file']:r for r in csv.DictReader((source/'reports/final_patch_review_v2/clean_final_manifest.csv').open())}
 holds={r['patch_name'] for r in csv.DictReader((source/'reports/final_patch_review_v2/scratch_31_dispositions.csv').open()) if r['visual_disposition']!='KEEP'}
 holds|={r['patch_name'] for r in csv.DictReader((source/'reports/final_patch_review_v2/edge_patch_dispositions.csv').open())}
 sources={}; manifest=[]; coco=json.loads((source/'F971/NG/annotations/coco.json').read_text()); images={i['file_name']:i for i in coco['images']}; annotations=[]; replacements=0
 for n,r in enumerate(rows):
  split={'validation':'val'}.get(r['split'],r['split']); name=r['source_image']
  if name in sources and sources[name]!=split: raise ValueError('Source split leakage')
  sources[name]=split
  if r['filename'] in holds: raise ValueError('HOLD/DROP in curated manifest')
  p=source/r['patch_path']; matching=[v for k,v in clean.items() if Path(k).name==p.name]
  if r['domain']=='clean':
   if len(matching)!=1: raise ValueError('Unmapped clean crop')
   x,y=int(matching[0]['crop_x']),int(matching[0]['crop_y']); replacements+=matching[0]['disposition']=='REPLACE_FROM_SOURCE_RGB'
  else:
   m=re.search(r'_x(\d+)_y(\d+)',p.stem)
   if not m: raise ValueError('Missing crop offset')
   x,y=map(int,m.groups())
  candidates=list((source/'F971').rglob(name))
  if len(candidates)!=1: raise ValueError(f'Ambiguous source {name}')
  with Image.open(p) as patch,Image.open(candidates[0]) as original:
   w,h=patch.size
   if patch.mode!='RGB' or x+w>original.width or y+h>original.height: raise ValueError('Invalid RGB crop')
   if not np.array_equal(np.asarray(patch),np.asarray(original.convert('RGB').crop((x,y,x+w,y+h)))): raise ValueError(f'Unverified offset {p}')
  out=dest/'data/curated/patches'/split/p.name; digest=copy_checked(p,out)
  record=dict(usage='reference_only_not_training',id=n+1,source_image=name,source_path=(Path('data/raw')/candidates[0].relative_to(source)).as_posix(),old_path=str(p),path=out.relative_to(dest).as_posix(),split=split,domain=r['domain'],width=w,height=h,crop_offset=[x,y],sha256=digest,clean_replacement=bool(matching and matching[0]['disposition']=='REPLACE_FROM_SOURCE_RGB'))
  manifest.append(record)
  if r['domain']=='defect':
   im=images[name]
   with Image.open(candidates[0]) as original:
    if original.size!=(im['width'],im['height']): raise ValueError('COCO dimensions mismatch')
   for a in coco['annotations']:
    if a['image_id']!=im['id']: continue
    if not valid_polygons(a['segmentation'],im['width'],im['height']): raise ValueError('Invalid original polygon')
    pts=np.concatenate([np.array(q).reshape(-1,2) for q in a['segmentation']]); lo=pts.min(0); hi=pts.max(0)
    if hi[0]<x or lo[0]>x+w or hi[1]<y or lo[1]>y+h: continue
    if lo[0]<x or lo[1]<y or hi[0]>x+w or hi[1]>y+h: raise ValueError(f'Clipped polygon in approved crop {p.name}: {a["id"]}')
    annotations.append(dict(id=len(annotations)+1,image_id=n+1,source_annotation_id=a['id'],category_id=1,segmentation=translate(a['segmentation'],x,y),iscrowd=0))
 if replacements!=39: raise ValueError(f'Expected 39 replacements, got {replacements}')
 for raw_record in provenance:
  raw_path=dest/raw_record['new_path']; raw_record['source_image']=raw_path.name; raw_record['split']=sources.get(raw_path.name)
  if raw_path.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.tif','.tiff'):
   with Image.open(raw_path) as im: raw_record.update(width=im.width,height=im.height,mode=im.mode)
 save_json(dest/'data/manifest.json',manifest); save_json(dest/'data/provenance.json',provenance)
 save_json(dest/'data/curated/coco.json',dict(images=[dict(id=r['id'],file_name=r['path'],width=r['width'],height=r['height']) for r in manifest],annotations=annotations,categories=[dict(id=1,name='scratch')]))
 save_json(dest/'reports/migration.json',dict(patches=len(rows),replacements=replacements,raw_files=len(raw),required_bytes=required,polygons=len(annotations),split_counts={s:sum(r['split']==s for r in manifest) for s in ('train','val','test')}))
 print('Migration complete',len(rows),'patches',len(annotations),'polygons')
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--source',default=r'D:\RTC\RD\GAN_Project'); p.add_argument('--dest',default=str(ROOT)); a=p.parse_args(); migrate(a.source,a.dest)
