"""Held-out historical features. Future text is excluded before any encoder call."""
import hashlib,json,os
from pathlib import Path
import numpy as np
from .queue import ROOT,read,sha
def write(p,j):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf-8');t.replace(p)
def rows(dataset):
 import pandas as pd
 from formal_model.english_predicted.english_data import LABELS
 path=ROOT/f'baselines/DialogueGLP/data/{dataset}/processed/test_full.csv' if dataset!='dailydialog' else ROOT/'baselines/DialogueGLP/data/daily_dialogue/test.csv'
 df=pd.read_csv(path);labels=LABELS[dataset];result=[];targets=[]
 if dataset=='dailydialog':
  df=df.sort_values(['Dialogue_ID','Utterance_ID']);df['Speaker']=df.groupby('Dialogue_ID',sort=False).cumcount()%2
 for key,g in df.groupby('Dialogue_ID',sort=True):
  values=g.sort_values('Utterance_ID').to_dict('records')
  if len(values)<2:continue
  history=values[:-1];target=values[-1]['Speaker'];speakers=list(dict.fromkeys([v['Speaker'] for v in history]+[target]))
  result.append(dict(id=str(key),text=[str(v['Utterance']) for v in history],speakers=[speakers.index(v['Speaker']) for v in history],target=speakers.index(target)))
  targets.append(labels.index(values[-1]['Emotion']))
 assert len({r['id'] for r in result})==len(result)
 return result,np.asarray(targets,dtype=np.int64),labels,path
def prepare(ds):
 import torch
 from transformers import AutoTokenizer
 from formal_model.english_predicted.extractor import EnglishExtractor
 from formal_model.prepare_extension_syntax import record_from_doc
 from formal_model.extension_syntax_data import align_contextual_words
 inp,y,labels,source=rows(ds)
 texts=sorted({t for r in inp for t in r['text']});folder=ROOT/'test_cache'/ds;folder.mkdir(parents=True,exist_ok=True)
 md=read(ROOT/'payload/erc_english'/ds/'metadata.json');ck=ROOT/md['checkpoint_relative'];assert sha(ck)==md['checkpoint_sha']
 protocol=dict(dataset=ds,split='test',source_sha=sha(source),extractor_sha=md['checkpoint_sha'],code_sha=sha(Path(__file__)),texts_sha=hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest(),future_text_used=False,sentence_pooling='masked_bf16_single' if ds=='dailydialog' else 'legacy_unmasked_bf16_batch4',word_encoding='gpu_fp32' if ds=='dailydialog' else 'cpu_fp32',erc='full_classifier_masked_bf16_batch4_fixed_on_both_machines',labels=labels)
 pp=folder/'protocol.json'
 if pp.exists():assert read(pp)==protocol,'Test cache protocol differs; do not mix caches.'
 else:write(pp,protocol)
 if (folder/'complete.json').exists():
  done=read(folder/'complete.json');assert done['protocol_sha']==sha(pp)
  for n,h in done['files'].items():assert sha(folder/n)==h,n
  return inp,y,labels,folder,texts
 # Parser configuration matches the existing training producers.
 import stanza
 parser=read(ROOT/'formal_model/data/dailydialog_dependency_v1/manifest.json')['parser']
 assert stanza.__version__==parser['stanza_version']
 resource=ROOT/'data/stanza_resources'
 for name,h in parser['assets_sha256'].items():
  p=resource/'resources.json' if name=='resources' else resource/'en'/name/(('conll17' if name=='pretrain' else parser['processors'][name])+'.pt')
  assert sha(p)==h,('Parser resource mismatch',name)
 known={}
 parse=folder/'parses.jsonl'
 if parse.exists():
  for line in parse.open(encoding='utf-8'):
   r=json.loads(line);known[r['text']]=r
 missing=[t for t in texts if t not in known]
 if missing:
  nlp=stanza.Pipeline(lang='en',dir=str(resource),package=None,processors=parser['processors'],download_method=None,use_gpu=False,verbose=False)
  for start in range(0,len(missing),16):
   batch=missing[start:start+16]
   for t,doc in zip(batch,nlp.bulk_process(batch)):known[t]=record_from_doc(t,doc)
   # Atomic whole-file replacement makes an interrupted parse recoverable.
   tmp=parse.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(known[t],ensure_ascii=False)+'\n' for t in texts if t in known),encoding='utf-8');tmp.replace(parse)
   if start%160==0:print('TEST PARSE',ds,len(known),'/',len(texts),flush=True)
  del nlp
 records=[known[t] for t in texts];parse_sha=sha(parse)
 torch.set_num_threads(4);torch.manual_seed(42);torch.use_deterministic_algorithms(True)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 asset=ROOT/'baselines/assets/roberta-large';tok=AutoTokenizer.from_pretrained(str(asset),local_files_only=True,use_fast=True)
 model=EnglishExtractor(dict(asset=str(asset),classes=len(labels)))
 model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True,mmap=True),strict=True);model.cuda().eval()
 sentences=np.empty((len(texts),1024),np.float32);prob=np.empty((len(texts),len(labels)),np.float32)
 with torch.inference_mode():
  for start in range(0,len(texts),4):
   pack=tok(texts[start:start+4],padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda')
   with torch.autocast('cuda',dtype=torch.bfloat16):z=model(pack)
   prob[start:start+len(z)]=z.float().softmax(-1).cpu().numpy()
  size=1 if ds=='dailydialog' else 4
  for start in range(0,len(texts),size):
   pack=tok(texts[start:start+size],padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda')
   with torch.autocast('cuda',dtype=torch.bfloat16):
    if ds=='dailydialog':v=model.feature(pack)
    else:
     h=model.model(**pack).last_hidden_state
     v=(h*model.attention(h).softmax(dim=1)).sum(dim=1)
   sentences[start:start+len(v)]=v.float().cpu().numpy()
 assert np.isfinite(sentences).all() and np.isfinite(prob).all()
 np.savez_compressed(folder/'sentence_erc.npz',values=sentences,p=prob)
 # Legacy English word caches were CPU float32; retain that protocol.
 device='cuda' if ds=='dailydialog' else 'cpu';model.to(device);torch.cuda.empty_cache()
 parts=folder/'word_features';parts.mkdir(exist_ok=True);files={}
 with torch.inference_mode():
  for i,r in enumerate(records):
   key=hashlib.sha256(r['text'].encode()).hexdigest();p=parts/(key+'.npz')
   if p.exists():
    with np.load(p,allow_pickle=False) as z:assert str(z['parse_sha'])==parse_sha and str(z['checkpoint_sha'])==md['checkpoint_sha']
   else:
    pack=tok(r['text'],truncation=True,max_length=512,return_offsets_mapping=True,return_tensors='pt');offset=pack.pop('offset_mapping')[0].numpy()
    h=model.model(**pack.to(device)).last_hidden_state[0].float().cpu().numpy()
    if ds=='dailydialog':v,valid,cut=align_contextual_words(r,offset,h)
    else:
     v=np.zeros((len(r['words']),1024),np.float16);valid=np.zeros(len(v),bool);content=offset[:,1]>offset[:,0]
     for j,w in enumerate(r['words']):
      overlap=content&(offset[:,0]<w['end'])&(offset[:,1]>w['start'])
      if overlap.any() and offset[overlap,1].max()>=w['end']:v[j]=h[overlap].mean(0);valid[j]=True
     cut=bool(content.any() and offset[content,1].max()<len(r['text'].rstrip()))
    tmp=p.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,values=v,valid=valid,truncated=cut,parse_sha=parse_sha,checkpoint_sha=md['checkpoint_sha'])
    tmp.replace(p)
   files[str(p.relative_to(folder))]=sha(p)
   if i%100==0:print('TEST WORDS',ds,i+1,'/',len(texts),flush=True)
 del model;torch.cuda.empty_cache()
 write(folder/'word_encoding_complete.json',dict(complete=True,parse_sha=parse_sha,checkpoint_sha=md['checkpoint_sha']))
 files.update({n:sha(folder/n) for n in ('parses.jsonl','sentence_erc.npz','word_encoding_complete.json')})
 write(folder/'complete.json',dict(complete=True,protocol_sha=sha(pp),files=files))
 return inp,y,labels,folder,texts
