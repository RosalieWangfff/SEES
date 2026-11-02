"""Explicit staged recovery queue. Defaults to read-only preflight."""
import argparse, hashlib, json, shutil, subprocess, sys, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DS=('meld','dailydialog','emorynlp')
SEEDS=(1008,42,43)
MAIN='G2__dep_typed'
ABLATIONS=('G1__dep_typed','G_word__dep_typed','G2','G2__dep_untyped','G2__neg_local','G2__dep_neg_joint')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def call(args,cwd=ROOT):
 command=[sys.executable,'-X','utf8','-u',*args]
 print('RUN:',subprocess.list2cmdline(command),flush=True)
 subprocess.run(command,cwd=cwd,check=True)
def out(ds,arm,seed):return ROOT/'results_rebuilt'/ds/'core_prior_ablation/windows/coupled'/ds/f'{arm}_s{seed}'
def complete(p,glp=False):
 if not (p/'complete.json').exists():return False
 j=read(p/'complete.json')
 assert j.get('complete',glp) and not j.get('diagnostic_only',False),p
 assert j['checkpoint_sha']==sha(p/'best.pt'),p
 if 'protocol_sha' in j:assert j['protocol_sha']==sha(p/'protocol.json'),p
 assert j['predictions_sha' if glp else 'prediction_sha']==sha(p/'predictions.npz'),p
 assert zipfile.is_zipfile(p/'best.pt'),p
 return True
def recovered(ds,arm,seed):
 return next((r for r in read(ROOT/'evidence/recovered_base_manifest.json') if (r['dataset'],r['arm'],r['seed'])==(ds,arm,seed)),None)
def check(deep=False):
 m=read(ROOT/'PACKAGE_MANIFEST.json');bad=[]
 for n,r in m.items():
  f=ROOT/n
  if not f.is_file() or f.stat().st_size!=r['bytes'] or (deep and sha(f)!=r['sha256']):bad.append(n)
 assert not bad,('Package incomplete or modified',bad[:20])
 for ds in DS:
  for s in ('train','val'):
   assert (ROOT/'glp/data'/ds/(s+'.jsonl')).is_file()
  md=read(ROOT/'payload/erc_english'/ds/'metadata.json')
  assert sha(ROOT/md['checkpoint_relative'])==md['checkpoint_sha']
 for ds in DS:erc(ds)
 print(json.dumps({'package_ok':True,'files':len(m),'deep_hash_check':deep,'free_GiB':round(shutil.disk_usage(ROOT).free/2**30,2),'erc_training_runs':0,'main_training_runs':9,'glp_training_runs':3,'ablation_training_runs':51,'reused_ablation_checkpoints_in_package':3,'remote_test_runs':66,'local_supplementary_test_runs':12,'training_started':False},indent=2),flush=True)
def erc(ds):
 folder=ROOT/'results_predicted_english'/ds/'cache'
 done=read(folder/'complete.json')
 cfg=read(folder/'erc_protocol.json');md=read(ROOT/'payload/erc_english'/ds/'metadata.json')
 assert done['complete'] and done['full_model_oof'] and not done['test_used'] and not done['diagnostic_only']
 assert done['cache_sha']==sha(folder/'history_predictions.npz') and done['protocol_sha']==sha(folder/'erc_protocol.json')
 assert cfg['full_checkpoint']==md['checkpoint_sha'] and cfg['prepared_sources']==md['forecast_sources']==done['prepared_sources']
 assert cfg['input_sources']==md['raw_sources'] and cfg['labels']==md['labels']==done['labels']
 print('REUSE VERIFIED OOF CACHE:',ds,flush=True)
def train(ds,arm,seed):
 if recovered(ds,arm,seed):
  record=recovered(ds,arm,seed)
  assert record.get('package_path') and complete(ROOT/record['package_path']),('Missing packaged recovered checkpoint',record)
  print('REUSE PACKAGED CHECKPOINT; NO RETRAIN:',ds,arm,seed,flush=True);return
 p=out(ds,arm,seed)
 if complete(p):print('SKIP VERIFIED',p,flush=True);return
 assert shutil.disk_usage(ROOT).free>15*2**30,'Need at least 15 GiB free for atomic checkpoints.'
 # Each distinct model is probed once before its first production run.
 probe=out(ds,arm,1008).with_name(arm+'_s1008_probe_v4')
 if not (probe/'complete.json').exists():
  call(['-m','recovery.train','--dataset',ds,'--arm',arm,'--scaffold','coupled','--seed','1008','--probe','--execute'])
 call(['-m','recovery.train','--dataset',ds,'--arm',arm,'--scaffold','coupled','--seed',str(seed),'--execute'])
 assert complete(p)
def glp(ds):
 p=ROOT/'glp/results'/ds/'text'
 if complete(p,True):print('SKIP VERIFIED',p,flush=True);return
 assert shutil.disk_usage(ROOT).free>15*2**30,'Need 15 GiB free.'
 for probe in (True,False):
  call(['-m','aligned.train','--dataset',ds,'--arm','text','--microbatch','24',*(['--probe'] if probe else [])],ROOT/'glp')
 assert complete(p,True)
def main():
 p=argparse.ArgumentParser()
 p.add_argument('--stage',choices=('check','erc','main','glp','ablations','train-all','test-main','test-all'),default='check')
 p.add_argument('--datasets',choices=DS,nargs='+',default=list(DS))
 p.add_argument('--deep',action='store_true');p.add_argument('--execute',action='store_true')
 a=p.parse_args();check(a.deep)
 if a.stage=='check' or not a.execute:return
 if a.stage in ('test-main','test-all'):
  for ds in a.datasets:call(['-m','recovery.evaluate','--dataset',ds,*(['--ablations'] if a.stage=='test-all' else []),'--execute'])
  return
 for ds in a.datasets:
  if a.stage in ('erc','main','ablations','train-all'):erc(ds)
  if a.stage in ('main','train-all'):
   for seed in SEEDS:train(ds,MAIN,seed)
  if a.stage in ('main','glp','train-all'):glp(ds)
  if a.stage in ('ablations','train-all'):
   for arm in ABLATIONS:
    for seed in SEEDS:train(ds,arm,seed)
 print('REQUESTED QUEUE COMPLETE. Keep the entire results and cache directories.',flush=True)
if __name__=='__main__':main()
