import csv,json
from collections import defaultdict
from statistics import mean,stdev
from .queue import ROOT,read,sha
def main():
 rows=[]
 candidates=[f for source in ('test_results','local_test_results') for f in (ROOT/source).glob('*/*/metrics.json')]
 seen=set()
 for f in candidates:
  p=read(f);d=p['provenance'];done=read(f.parent/'complete.json')
  assert done['complete'] and done['predictions_sha']==sha(f.parent/'predictions.npz')
  key=(d['dataset'],d['arm'],d['seed']);assert key not in seen,('Duplicate result',key);seen.add(key)
  rows.append(dict(dataset=d['dataset'],arm=d['arm'],seed=d['seed'],weighted=p['metrics']['weighted'],macro=p['metrics']['macro'],split='test'))
 out=ROOT/'reports';out.mkdir(exist_ok=True)
 with (out/'test_runs.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=['dataset','arm','seed','weighted','macro','split']);w.writeheader();w.writerows(rows)
 groups=defaultdict(list)
 for r in rows:groups[r['dataset'],r['arm']].append(r)
 lines=['Test results only. Partial seed sets remain explicitly labelled; validation scores are not substituted.','', '| Dataset | Model | n | W-F1 mean ± sample SD | Macro-F1 mean ± sample SD |','|---|---|---:|---:|---:|']
 for (d,a),rs in sorted(groups.items()):
  def fmt(k):
   v=[r[k] for r in rs];return f'{mean(v):.4f} ± {stdev(v):.4f}' if len(v)>1 else f'{v[0]:.4f} (single seed)'
  lines.append(f'| {d} | {a} | {len(rs)} | {fmt("weighted")} | {fmt("macro")} |')
 (out/'TEST_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
 print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
