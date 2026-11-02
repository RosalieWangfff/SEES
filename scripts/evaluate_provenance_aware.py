"""Freeze validation-selected checkpoints and evaluate the held-out split once."""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,sys
from pathlib import Path
import numpy as np
from recovery.queue import ROOT,DS,SEEDS,MAIN,ABLATIONS,out,complete,sha,read,recovered
from recovery.test_features import write,prepare
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=DS,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--ablations',action='store_true');p.add_argument('--execute',action='store_true');p.add_argument('--recovered-root',type=Path);p.add_argument('--batch',type=int,default=32);a=p.parse_args()
 ds=a.dataset
 runs=[dict(r,path=Path(r['path'])) for r in read(a.manifest) if r['dataset']==ds]
 assert runs, 'Empty evaluation manifest'
 for r in runs:
  old=recovered(ds,r['arm'],r['seed']) if r['kind']=='ours' else None
  current_checkpoint=sha(r['path']/'best.pt')
  current_protocol=sha(r['path']/'protocol.json')
  r['recovered']=bool(old and old['checkpoint_sha']==current_checkpoint and old['protocol_sha']==current_protocol)
  if old and old['checkpoint_sha']==current_checkpoint and old['protocol_sha']!=current_protocol:
   r['prior_provenance']={'checkpoint_sha':old['checkpoint_sha'],'protocol_sha':old['protocol_sha'],'prediction_sha':old.get('prediction_sha'),'current_protocol_sha':current_protocol,'note':'checkpoint matches; old protocol not asserted equivalent; current run requires independent checks'}
   print(json.dumps({'dataset':ds,'arm':r['arm'],'seed':r['seed'],**r['prior_provenance']},ensure_ascii=False),flush=True)
 locks=[]
 for r in runs:
  assert complete(r['path'],r['kind']=='glp'),('Finish training before test',r)
  protocol=read(r['path']/'protocol.json')
  assert protocol['dataset']==ds and protocol['seed']==r['seed'] and protocol['arm']==r['arm'] and not protocol['test_used']
  reviewed_differences={}
  for name,h in protocol['sources'].items():
   base=ROOT/'glp' if r['kind']=='glp' else ROOT
   actual=sha(base/name)
   if r['recovered'] and name in ('formal_model/english_predicted/erc_cache.py','formal_model/train_extension_extractor.py') and actual!=h:
    # These producer/idle-check changes do not run during frozen-checkpoint inference.
    reviewed_differences[name]={'training_sha':h,'inference_package_sha':actual,'reason':'producer code not invoked in this evaluation'}
   else:assert actual==h,('Training source changed',name)
  if r['recovered']:
   original=recovered(ds,r['arm'],r['seed']);assert original and original['checkpoint_sha']==sha(r['path']/'best.pt') and original['protocol_sha']==sha(r['path']/'protocol.json')
  locks.append({**r,'path':str(r['path']),'checkpoint_sha':sha(r['path']/'best.pt'),'protocol_sha':sha(r['path']/'protocol.json'),'reviewed_inactive_producer_differences':reviewed_differences})
 print(json.dumps({'dataset':ds,'checkpoints':len(locks),'test_started':False},indent=2),flush=True)
 if not a.execute:return
 directory=ROOT/'test_results_nc_revision'/ds;directory.mkdir(parents=True,exist_ok=True)
 lockpath=directory/('locked_'+sha(a.manifest)[:12]+'.json')
 lock=dict(dataset=ds,selection='validation W-F1; frozen configuration; no test-driven reselection',runs=locks,evaluator_sha=sha(Path(__file__)))
 if lockpath.exists():assert read(lockpath)==lock,'Frozen checkpoint set changed.'
 else:write(lockpath,lock)
 import torch
 from transformers import AutoTokenizer
 from sklearn.metrics import f1_score
 from formal_model.dependency_data import DependencyFeatures
 from formal_model.english_structure_matrix.data import CoreData,annotations
 from formal_model.english_predicted.mixed_runtime import MixedHistory,install_model
 from formal_model.backbone import create_backbone
 from formal_model.extension_model import ExtensionModel
 from nc_revision.model import PriorEvidenceModel
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 (ROOT/'results').mkdir(exist_ok=True)
 import msvcrt
 handle=(ROOT/'results/extension_extractor_gpu.lock').open('a+b');handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
 inp,y,labels,cache,texts=prepare(ds)
 with np.load(cache/'sentence_erc.npz',allow_pickle=False) as z:vectors=z['values'].copy();erc=z['p'].argmax(1)
 index={t:i for i,t in enumerate(texts)}
 # Gold history labels are not loaded or passed to either forecasting model.
 rows=[dict(r,emotions=[int(erc[index[t]]) for t in r['text']]) for r in inp]
 data=CoreData.__new__(CoreData);data.rows={'test':rows};data.classes=len(labels)
 data.word_features=DependencyFeatures(cache);data.syntax={}
 for line in (cache/'parses.jsonl').open(encoding='utf-8'):
  r=json.loads(line);data.syntax[r['text']]=annotations(r['words'])
 tok=AutoTokenizer.from_pretrained(str(ROOT/'baselines/assets/roberta-large'),local_files_only=True)
 prepared=MixedHistory(inp,vectors,index,tok,len(labels));prepared.actual_microbatch=a.batch
 sys.path.insert(0,str(ROOT/'glp'))
 from aligned.model import make_glp,MatchedGLP
 from aligned.batching import install,many
 from aligned.data import Inputs
 glp_data=Inputs.__new__(Inputs);glp_data.rows={'test':[dict(inputs=r) for r in rows]};glp_data.vectors=torch.from_numpy(vectors);glp_data.index=index;glp_data.tokenizer=tok
 local=[(' '+tok.sep_token+' ').join(('I: ' if sp==r['target'] else 'Other: ')+t for sp,t in zip(r['speakers'][-2:],r['text'][-2:])) for r in rows]
 enc=tok(local,max_length=512,truncation=True);glp_data.tokens={'test':[{k:v[i] for k,v in enc.items()} for i in range(len(rows))]}
 for r,l in zip(runs,locks):
  dest=directory/(r['arm']+'_s'+str(r['seed']));dest.mkdir(exist_ok=True)
  provenance=dict(checkpoint_sha=l['checkpoint_sha'],training_protocol_sha=l['protocol_sha'],test_cache_protocol_sha=sha(cache/'protocol.json'),evaluator_sha=sha(Path(__file__)),dataset=ds,split='test',arm=r['arm'],seed=r['seed'],labels=labels,inference_batch=a.batch if r['kind']=='ours' else 24)
  if (dest/'complete.json').exists():
   done=read(dest/'complete.json');assert done['provenance']==provenance and done['predictions_sha']==sha(dest/'predictions.npz');print('SKIP VERIFIED TEST',dest,flush=True);continue
  torch.manual_seed(r['seed']);torch.cuda.manual_seed_all(r['seed'])
  if r['kind']=='ours':
   base=None
   if r['arm']!='P0':
    base,_=create_backbone(classes=len(labels),asset=ROOT/'baselines/assets/roberta-large')
    base.ei_roberta.model.config.output_hidden_states=False
   if a.recovered_root:
    from formal_model.core_prior_ablation.model import PriorEvidenceModel as OldModel
    constructor=OldModel
   else:constructor=PriorEvidenceModel
   model=install_model(constructor(ExtensionModel(base,1024,len(labels)) if base else None,1024,len(labels),r['arm'],'coupled',r['seed']))
  else:
   base=make_glp(ROOT/'baselines/assets/roberta-large',len(labels));base.ei_roberta.model.config.output_hidden_states=False;install(base)
   model=MatchedGLP(base,len(labels),False)
  state=torch.load(r['path']/'best.pt',map_location='cpu',weights_only=True,mmap=True);model.load_state_dict(state,strict=True);del state
  model.cuda().eval();prob=np.empty((len(rows),len(labels)),np.float32)
  size=a.batch if r['kind']=='ours' else 24
  with torch.inference_mode():
   for start in range(0,len(rows),size):
    ids=list(range(start,min(start+size,len(rows))))
    with torch.autocast('cuda',dtype=torch.float16):
     if r['kind']=='ours':
      own,lengths=data.own_history('test',ids,'cuda')
      if r['arm']=='P0':logits=model(own,lengths)['logits']
      else:
       local,batch,_=prepared.collate(ids);words=data.words('test',ids,'cuda')
       logits=model(own,lengths,local,batch,words)['logits']
     else:
      inputs,own,lengths=many(glp_data,'test',ids,'cuda');logits=model(inputs,own,lengths)
    prob[ids]=logits.float().softmax(-1).cpu().numpy()
  assert np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=1e-5)
  metrics={m:100*f1_score(y,prob.argmax(1),labels=range(len(labels)),average=m,zero_division=0) for m in ('weighted','macro')}
  tmp=dest/'predictions.tmp.npz';np.savez_compressed(tmp,ids=np.array([r['id'] for r in rows]),y=y,p=prob);tmp.replace(dest/'predictions.npz')
  write(dest/'metrics.json',dict(provenance=provenance,metrics=metrics,rows=len(rows)))
  write(dest/'complete.json',dict(complete=True,provenance=provenance,predictions_sha=sha(dest/'predictions.npz')))
  print('TEST RESULT',ds,r['arm'],r['seed'],metrics,flush=True)
  del model,base;torch.cuda.empty_cache()
 handle.close()

if __name__=='__main__':main()
