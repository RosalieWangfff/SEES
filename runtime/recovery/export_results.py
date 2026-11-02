"""Export trained weights as well as scores; never delete source results."""
from datetime import datetime
import json,shutil,zipfile
from .queue import ROOT,sha
def main():
 files=[]
 for name in ('results_rebuilt','recovered_checkpoints','glp/results','results_predicted_english','test_results','local_test_results','reports'):
  for p in (ROOT/name).rglob('*'):
   if p.is_file() and p.name not in ('resume.pt','resume.tmp.pt') and '.tmp' not in p.name and p.suffix!='.lock' and not (p.name=='final.pt' and p.parent.name.startswith('fold')):files.append(p)
 assert any(p.name=='best.pt' for p in files),'No trained checkpoints to export yet.'
 size=sum(p.stat().st_size for p in files)
 assert shutil.disk_usage(ROOT).free>size+10*2**30,'Need enough space for archive plus 10 GiB reserve; download the result folders directly instead.'
 dest=ROOT/'exports';dest.mkdir(exist_ok=True)
 target=dest/('RESULTS_WITH_CHECKPOINTS_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.zip')
 manifest={}
 with zipfile.ZipFile(target.with_suffix('.tmp'), 'w',zipfile.ZIP_STORED,allowZip64=True) as z:
  for i,p in enumerate(files):
   name=p.relative_to(ROOT).as_posix();manifest[name]=dict(bytes=p.stat().st_size,sha256=sha(p));z.write(p,name)
   if p.name=='best.pt':print('BACKUP',name,flush=True)
  z.writestr('BACKUP_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
 target.with_suffix('.tmp').replace(target)
 print('BACKUP READY:',target,'GiB=',round(size/2**30,2),flush=True)
if __name__=='__main__':main()
