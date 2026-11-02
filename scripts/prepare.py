"""Build an isolated revision package against the existing recovery installation.

No training on invocation. Never edits the original recovery/aligned sources.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

def replace(s, old, new, count=1):
    assert s.count(old) == count, ('Unsupported source version', old[:100], s.count(old))
    return s.replace(old, new)

def generate(root):
    base = root / 'recovery'
    train = (base/'train.py').read_text(encoding='utf-8')
    train = train.replace('ROOT/"results_rebuilt"', 'ROOT/"results_nc_revision"')
    train = replace(train, 'from recovery.model import ARMS, PriorEvidenceModel', 'from nc_revision.model import ARMS, PriorEvidenceModel')
    train = replace(train, '    sources = {name:digest(ROOT/name) for name in sorted(relative)}',
        "    relative.update('nc_revision/'+n for n in ('train.py','model.py'))\n    sources = {name:digest(ROOT/name) for name in sorted(relative)}")
    train = replace(train, "seconds=time.perf_counter()-epoch_start)",
        "seconds=time.perf_counter()-epoch_start, peak_vram_gib=torch.cuda.max_memory_allocated()/2**30)")

    glp = (root/'glp/aligned/train.py').read_text(encoding='utf-8')
    for name in ('common','data','batching','model','resume'):
        glp = glp.replace('from .'+name+' import', 'from aligned.'+name+' import')
    glp = replace(glp, '    a = p.parse_args()',
        "    p.add_argument('--seed',type=int,choices=(1008,42,43),required=True)\n    a = p.parse_args()")
    glp = replace(glp, '    seed(1008)', '    seed(a.seed)')
    glp = glp.replace('1008*100+epoch', 'a.seed*100+epoch')
    glp = replace(glp, 'arm=a.arm, seed=1008,', 'arm=a.arm, seed=a.seed,')
    glp = replace(glp, "out = ROOT/'results'/a.dataset/(a.arm+suffix+(f'_probe_b{a.microbatch}' if a.probe else ''))",
        "out = ROOT.parent/'results_nc_glp'/a.dataset/(a.arm+suffix+f'_s{a.seed}'+(f'_probe_b{a.microbatch}' if a.probe else ''))")
    glp = replace(glp, "    protocol = dict(dataset=a.dataset,", "    sources['../nc_revision/glp_train.py'] = sha(Path(__file__))\n    protocol = dict(dataset=a.dataset,")
    glp = 'from pathlib import Path\n'+glp
    glp = replace(glp, '    buckets = {k:[]', "    protocol['parameters'] = sum(p.numel() for p in model.parameters())\n    # Count stored separately: do not change the resume protocol after it is written.\n    write(out/'parameter_count.json', {'total':protocol.pop('parameters')})\n    buckets = {k:[]")

    ev = (base/'evaluate.py').read_text(encoding='utf-8')
    ev = replace(ev, 'from .queue import', 'from recovery.queue import')
    ev = replace(ev, 'from .test_features import', 'from recovery.test_features import')
    ev = replace(ev, 'from .model import', 'from nc_revision.model import')
    ev = replace(ev, "p.add_argument('--dataset',choices=DS,required=True);", "p.add_argument('--dataset',choices=DS,required=True);p.add_argument('--manifest',type=Path,required=True);")
    start = ev.index(' ds=a.dataset;arms=')
    end = ev.index(' locks=[]', start)
    ev = ev[:start]+" ds=a.dataset\n runs=[dict(r,path=Path(r['path'])) for r in read(a.manifest) if r['dataset']==ds]\n assert runs, 'Empty evaluation manifest'\n for r in runs:\n  old=recovered(ds,r['arm'],r['seed']) if r['kind']=='ours' else None\n  r['recovered']=bool(old and old['checkpoint_sha']==sha(r['path']/'best.pt'))\n"+ev[end:]
    ev = ev.replace("if recovered(ds,r['arm'],r['seed']) and", "if r['recovered'] and")
    ev = ev.replace("if recovered(ds,r['arm'],r['seed']):", "if r['recovered']:")
    ev = replace(ev, "directory=ROOT/('local_test_results' if a.recovered_root else 'test_results')/ds;", "directory=ROOT/'test_results_nc_revision'/ds;")
    ev = replace(ev, "lockpath=directory/('locked_recovered.json' if a.recovered_root else 'locked_all.json' if a.ablations else 'locked_main.json')",
        "lockpath=directory/('locked_'+sha(a.manifest)[:12]+'.json')")
    ev = replace(ev, "'path':str(r['path']) if a.recovered_root else str(r['path'].relative_to(ROOT))", "'path':str(r['path'])")
    ev = replace(ev, ' from .summarize import main as summary\n summary()', '')
    # Batch size stays as in the original GLP protocol; this is seed replication.
    return {'__init__.py':'', 'train.py':train, 'glp_train.py':glp, 'evaluate.py':ev,
            'model.py':(HERE/'model.py').read_text(encoding='utf-8')}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--install',action='store_true')
    a=p.parse_args(); files=generate(a.root.resolve())
    for n,s in files.items(): ast.parse(s, filename=n)
    out=a.root/'nc_revision'
    for n,s in files.items():
        f=out/n
        if f.exists() and f.read_text(encoding='utf-8')!=s:
            raise RuntimeError('Existing revision source differs: '+str(f))
    if a.install:
        out.mkdir(exist_ok=True)
        for n,s in files.items(): (out/n).write_text(s,encoding='utf-8')
    print(json.dumps({'source_compatible':True,'installed':a.install,'training_started':False,
        'files':{n:hashlib.sha256(s.encode()).hexdigest() for n,s in files.items()}},indent=2))

if __name__=='__main__': main()
