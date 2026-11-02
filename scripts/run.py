"""Read-only planning by default; --execute is the user's explicit GPU launch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from prepare import generate

DS=('meld','dailydialog','emorynlp')
SEEDS=(1008,42,43)

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()

def verify(p,ds,arm,seed,kind):
    c=read(p/'complete.json'); q=read(p/'protocol.json')
    assert c.get('complete') and not c.get('diagnostic_only',False) and not q['test_used'],p
    assert (q['dataset'],q['arm'],q['seed'])==(ds,arm,seed),p
    for f,key in [('best.pt','checkpoint_sha'),('protocol.json','protocol_sha'),
                  ('predictions.npz','predictions_sha' if kind=='glp' else 'prediction_sha')]:
        assert c[key]==sha(p/f), ('Changed artifact',p/f)
    return q,c['checkpoint_sha']

def candidates(root,extras,ds,arm,seed,kind):
    if kind=='glp':
        paths=[root/'results_nc_glp'/ds/f'text_s{seed}']
        if seed==1008: paths += [root/'glp/results'/ds/'text']
    else:
        paths=[root/f/ds/'core_prior_ablation/windows/coupled'/ds/f'{arm}_s{seed}'
               for f in ('results_rebuilt','results_predicted_english','results_predicted_structure_matrix','results_nc_revision')]
        paths += [root/'recovered_checkpoints'/ds/f'{arm}_s{seed}']
        for extra in extras:
            paths += list(extra.rglob(f'{arm}_s{seed}/complete.json'))
        paths=[p.parent if p.name=='complete.json' else p for p in paths]
    found=[]
    for p in dict.fromkeys(paths):
        if (p/'complete.json').exists():
            q=read(p/'protocol.json')
            if q.get('dataset')!=ds:continue
            _,h=verify(p,ds,arm,seed,kind); found.append((p,h))
    if len({h for _,h in found})>1:
        raise RuntimeError('Multiple non-identical completed checkpoints; identify the published run: '+str([str(p) for p,_ in found]))
    return found[0][0] if found else None

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=('cross','glp','relations','evaluate'),required=True)
    p.add_argument('--datasets',nargs='+',choices=DS,default=list(DS))
    p.add_argument('--checkpoint-root',action='append',type=Path,default=[])
    p.add_argument('--manifest',type=Path)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args();root=a.root.resolve()
    if a.execute and os.name!='nt':p.error('Existing recovery evaluator targets Windows; use the original Windows server environment.')
    files=generate(root)
    for name,source in files.items():
        path=root/'nc_revision'/name
        if not path.exists() or path.read_text(encoding='utf-8')!=source:
            raise RuntimeError('Run prepare.py --root ... --install first. Source mismatch: '+str(path))
    env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([str(root),str(root/'glp'),env.get('PYTHONPATH','')])
    def call(module,args):
        subprocess.run([sys.executable,'-X','utf8','-u','-m',module,*args],cwd=root,env=env,check=True)
    if a.stage=='evaluate':
        if not a.manifest:p.error('--manifest required')
        for ds in a.datasets:call('nc_revision.evaluate',['--dataset',ds,'--manifest',str(a.manifest.resolve()),*(['--execute'] if a.execute else [])])
        return
    arms=('G1','G_word') if a.stage=='cross' else ('text',) if a.stage=='glp' else ('G2__shuffle_rel',)
    kind='glp' if a.stage=='glp' else 'ours';runs=[];missing=[]
    for ds in a.datasets:
        for arm in arms:
            for seed in SEEDS:
                path=candidates(root,a.checkpoint_root,ds,arm,seed,kind)
                if path is None:
                    missing.append((ds,arm,seed))
                    if kind=='glp':
                        path=root/'results_nc_glp'/ds/f'text_s{seed}'
                        if a.execute:
                            args=['--dataset',ds,'--arm','text','--seed',str(seed),'--microbatch','24']
                            # One integration probe per dataset, not one per seed.
                            if seed==SEEDS[0] or not any((root/'results_nc_glp'/ds).glob('text_s*_probe_b24/complete.json')):
                                call('nc_revision.glp_train',args+['--probe'])
                            call('nc_revision.glp_train',args)
                    else:
                        path=root/'results_nc_revision'/ds/'core_prior_ablation/windows/coupled'/ds/f'{arm}_s{seed}'
                        if a.execute:
                            args=['--dataset',ds,'--arm',arm,'--scaffold','coupled','--seed',str(seed),'--execute']
                            if not any(path.parent.glob(arm+'_s*_probe_v4/complete.json')):
                                call('nc_revision.train',args+['--probe'])
                            call('nc_revision.train',args)
                    if a.execute: verify(path,ds,arm,seed,kind)
                runs.append(dict(dataset=ds,arm=arm,seed=seed,kind=kind,path=str(path)))
    report=root/'nc_revision_reports';report.mkdir(exist_ok=True)
    dest=report/(a.stage+'_manifest.json')
    content=json.dumps(runs,indent=2)
    if dest.exists() and dest.read_text()!=content:
        raise RuntimeError('Manifest changed; inspect existing file before replacing: '+str(dest))
    dest.write_text(content,encoding='utf-8')
    print(json.dumps(dict(stage=a.stage,missing_training=missing,training_started=a.execute and bool(missing),manifest=str(dest)),indent=2))

if __name__=='__main__':main()
