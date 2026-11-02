"""Install isolated paired trainer; optionally run probe, training, or tests."""
import argparse, ast, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
DS=('meld','dailydialog','emorynlp')
SEEDS=(1008,42,43)
ARMS=('G2__dep_typed','G2__no_evidence')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def sub(source,old,new,n=1):
    if source.count(old)!=n:
        raise RuntimeError('Unexpected trainer version: '+old[:100]+' count='+str(source.count(old)))
    return source.replace(old,new)


def generate(root):
    train=(root/'nc_revision/train.py').read_text(encoding='utf8')
    train=sub(train,'from nc_revision.model import ARMS, PriorEvidenceModel','from evidence_control_v2.model import ARMS, PriorEvidenceModel, install_control')
    train=sub(train,'ROOT/"results_nc_revision"','ROOT/"results_evidence_control_v2"',2)
    train=sub(train,'model = install_model(PriorEvidenceModel(text, data.dim, data.classes, args.arm, args.scaffold, args.seed)).cuda()',
        'model = install_control(PriorEvidenceModel(text, data.dim, data.classes, args.arm, args.scaffold, args.seed)).cuda()')
    train=sub(train,"    sources = {name:digest(ROOT/name) for name in sorted(relative)}",
        "    relative.update('evidence_control_v2/'+n for n in ('train.py','model.py'))\n    sources = {name:digest(ROOT/name) for name in sorted(relative)}")
    train=sub(train,"    protocol['recovery_protocol']='frozen_selection_reconstruction_v1'",
        "    protocol['evidence_control']='paired_retrained_v1; no_evidence zeros evidence input while preserving prior projection'\n    protocol['recovery_protocol']='paired_fresh_training_v1'")
    ev=(root/'nc_revision/evaluate.py').read_text(encoding='utf8')
    ev=sub(ev,' from nc_revision.model import PriorEvidenceModel',' from evidence_control_v2.model import PriorEvidenceModel, install_control')
    ev=sub(ev,'model=install_model(constructor(ExtensionModel(base,1024,len(labels)) if base else None,1024,len(labels),r[\'arm\'],\'coupled\',r[\'seed\']))',
           'model=install_control(constructor(ExtensionModel(base,1024,len(labels)) if base else None,1024,len(labels),r[\'arm\'],\'coupled\',r[\'seed\']))')
    ev=sub(ev,"directory=ROOT/'test_results_nc_revision'/ds;","directory=ROOT/'test_results_evidence_control_v2'/ds;")
    # Fresh matched controls must never claim identity with recovered old runs.
    ev=sub(ev,"  old=recovered(ds,r['arm'],r['seed']) if r['kind']=='ours' else None\n  r['recovered']=bool(old and old['checkpoint_sha']==sha(r['path']/'best.pt'))",
           "  r['recovered']=False")
    model=(HERE/'no_evidence_model.py').read_text(encoding='utf8')
    files={'__init__.py':'','model.py':model,'train.py':train,'evaluate.py':ev}
    for n,s in files.items():ast.parse(s,filename=n)
    return files


def path(root,ds,arm,seed):
    return root/'results_evidence_control_v2'/ds/'core_prior_ablation'/'windows'/'coupled'/ds/f'{arm}_s{seed}'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--stage',required=True,choices=('install','probe','train','test'))
    p.add_argument('--datasets',nargs='+',choices=DS,default=list(DS))
    p.add_argument('--execute',action='store_true')
    a=p.parse_args();root=a.root.resolve()
    if not root.is_dir():raise RuntimeError('Project root missing')
    files=generate(root); package=root/'evidence_control_v2'
    if a.stage=='install':
        if not a.execute:
            print('PREVIEW only; pass --execute to install. No training.');return
        package.mkdir(exist_ok=True)
        for name,content in files.items():
            dest=package/name
            if dest.exists() and dest.read_text(encoding='utf8')!=content:
                raise RuntimeError('Existing file differs; refusing overwrite: '+str(dest))
        for name,content in files.items():(package/name).write_text(content,encoding='utf8')
        print(json.dumps({'installed':True,'training_started':False,'files':{n:digest(package/n) for n in files}},indent=2));return
    for name,content in files.items():
        if not (package/name).is_file() or (package/name).read_text(encoding='utf8')!=content:
            raise RuntimeError('Install package first, or source changed: '+name)
    if not a.execute:
        print('DRY RUN:',a.stage,a.datasets,'No training or test.');return
    if os.name!='nt':raise RuntimeError('The training/evaluation environment requires Windows.')
    if shutil.disk_usage(root).free < 20*1024**3:raise RuntimeError('Less than 20 GiB free; checkpoint saves require disk space')
    env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join((str(root),str(root/'glp'),env.get('PYTHONPATH','')))
    def call(module,args):
        command=[sys.executable,'-X','utf8','-u','-m',module,*args]
        print('START',subprocess.list2cmdline(command),flush=True)
        subprocess.run(command,cwd=root,env=env,check=True)
    if a.stage=='probe':
        for ds in a.datasets:
            for arm in ARMS:call('evidence_control_v2.train',['--dataset',ds,'--arm',arm,'--scaffold','coupled','--seed','1008','--probe','--execute'])
    elif a.stage=='train':
        for ds in a.datasets:
            for arm in ARMS:
                for seed in SEEDS:call('evidence_control_v2.train',['--dataset',ds,'--arm',arm,'--scaffold','coupled','--seed',str(seed),'--execute'])
        print('ALL TRAINING COMPLETE; evaluate on validation before test.',flush=True)
    elif a.stage=='test':
        sys.path.insert(0,str(root))
        from recovery.queue import complete
        manifest=[]
        for ds in a.datasets:
            for arm in ARMS:
                for seed in SEEDS:
                    folder=path(root,ds,arm,seed)
                    assert complete(folder),('Missing or invalid training',str(folder))
                    manifest.append(dict(dataset=ds,arm=arm,seed=seed,kind='ours',path=str(folder)))
        output=root/'evidence_control_v2_manifest.json'
        value=json.dumps(manifest,indent=2)
        if output.exists() and output.read_text(encoding='utf8')!=value:raise RuntimeError('Manifest already exists with different contents')
        if not output.exists():output.write_text(value,encoding='utf8')
        for ds in a.datasets:call('evidence_control_v2.evaluate',['--dataset',ds,'--manifest',str(output),'--execute'])
        print('ALL TESTS COMPLETE',flush=True)

if __name__=='__main__':main()
