"""DialogueGLP-repository DialogInfer-S adaptation. Test is a separate command.

Uses history text embeddings and speaker identity only. Never feeds future text or
gold/predicted historical emotion labels to the baseline. Train/val feature cache
and held-out test feature preparation are the existing SEES protocol.
"""
import argparse, ast, hashlib, importlib.util, json, math, os, random, sys, time, types
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score

HERE=Path(__file__).resolve()
DATASETS=('meld','dailydialog','emorynlp')
SEEDS=(1008,42,43)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()

def dump(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf8')
    tmp.replace(path)

def save(path,obj):
    tmp=path.with_suffix('.tmp.pt');torch.save(obj,tmp);tmp.replace(path)

def load_upstream(root, classes):
    """Load actual DialogueGLP repository classes, not a newly invented model."""
    from types import SimpleNamespace
    from aligned.batching import dialogue_forward
    source=root/'baselines/DialogueGLP/model'
    spec=importlib.util.spec_from_file_location('dialoginfer_source',source/'dialogue_infer.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    tree=ast.parse((source/'base_models.py').read_text(encoding='utf8'))
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='DialogueInferModel')
    scope={'torch':torch,'nn':nn,'DialogueInfer':mod.DialogueInfer}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'upstream_DialogueInferModel','exec'),scope)
    model=scope['DialogueInferModel'](SimpleNamespace(input_size=1024,hidden_size=1024,
                                         fc_dropout=.2,target_size=classes))
    # Only adaptation: mask padded history; valid single-item updates match upstream.
    model.model.forward=types.MethodType(dialogue_forward,model.model)
    return model

def train_data(root,ds):
    from aligned.common import rows, read, feature_path
    data={s:rows(ds,s) for s in ('train','val')}
    spec=read(root/'glp/data'/ds/'contract.json')
    texts=sorted({t for records in data.values() for r in records for t in r['inputs']['text']})
    with np.load(feature_path(ds),allow_pickle=False) as z:
        expected=hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest()
        assert str(z['sha'])==expected,'Train/validation feature cache mismatch'
        vectors=torch.from_numpy(z['values'].astype('float32'))
    assert vectors.shape==(len(texts),spec['dim']) and spec['dim']==1024
    assert torch.isfinite(vectors).all()
    return data,vectors,{t:i for i,t in enumerate(texts)},spec

def batch(rows,vectors,index,device):
    xs=[r['inputs'] for r in rows]
    assert all(len(x['text'])>0 and len(x['text'])==len(x['speakers']) for x in xs)
    values=torch.nn.utils.rnn.pad_sequence([vectors[[index[t] for t in x['text']]] for x in xs],batch_first=True).to(device)
    role=torch.nn.utils.rnn.pad_sequence([torch.tensor([int(s==x['target']) for s in x['speakers']],dtype=torch.float32) for x in xs],batch_first=True).to(device)
    lengths=torch.tensor([len(x['text']) for x in xs],device=device)
    valid=torch.arange(values.shape[1],device=device)[None,:]<lengths[:,None]
    return dict(embs=list(values.unbind(1)),speakers=[s[:,None] for s in role.unbind(1)],valid=list(valid.unbind(1)))

def metric(y,p,k):
    return {m:float(100*f1_score(y,p.argmax(1),labels=range(k),average=m,zero_division=0)) for m in ('weighted','macro')}

def predict(model,rows,vectors,index,batch_size,device):
    model.eval();out=[]
    with torch.inference_mode():
        for i in range(0,len(rows),batch_size):
            logits=model(batch(rows[i:i+batch_size],vectors,index,device))
            out.append(logits.float().softmax(-1).cpu().numpy())
    return np.concatenate(out)

def run(a):
    root=a.root.resolve();sys.path[:0]=[str(root),str(root/'glp')]
    from aligned.common import feature_path
    from aligned.batching import dialogue_forward
    data,vectors,index,spec=train_data(root,a.dataset)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    assert device=='cuda','CUDA required for actual runs'
    assert sys.platform=='win32','This project evaluation protocol currently supports Windows only'
    k=len(spec['labels']); out=root/'results_nc_dialoginfer'/a.dataset/f's{a.seed}'
    upstream=root/'baselines/DialogueGLP/model'
    protocol=dict(dataset=a.dataset,arm='DialogInfer-S-adapted',seed=a.seed,
       input='past dialogue utterance features and relative speaker identity; no history emotion or target text',
       data_sha={s:sha(root/'glp/data'/a.dataset/(s+'.jsonl')) for s in ('train','val')},
       features_sha=sha(feature_path(a.dataset)),labels=spec['labels'],epochs=16 if a.dataset=='dailydialog' else 5,
       effective_batch=32,lr=0.001,optimizer='AdamW',weight_decay=0.,hidden=1024,dropout=.2,
       feature_encoder='existing fixed RoBERTa utterance feature cache',selection='validation weighted F1, earliest tie',
       no_target_utterance=True,test_used=False,implementation='DialogueGLP repository DialogueInferModel; padding-mask adaptation',
       source_sha={str(p.relative_to(root)):sha(p) for p in [upstream/'base_models.py',upstream/'dialogue_infer.py',root/'glp/aligned/batching.py']},
       script_sha=sha(HERE), torch=str(torch.__version__),device=torch.cuda.get_device_name(0))
    pfile=out/'protocol.json'
    if a.mode=='train':
        out.mkdir(parents=True,exist_ok=True)
        if pfile.exists(): assert json.loads(pfile.read_text(encoding='utf8'))==protocol,'Existing protocol differs; do not overwrite'
        else: dump(pfile,protocol)
        if (out/'complete.json').exists():
            done=json.loads((out/'complete.json').read_text(encoding='utf8'))
            assert done['checkpoint_sha']==sha(out/'best.pt') and done['protocol_sha']==sha(pfile)
            print('Already complete:',out);return
        import msvcrt
        lock=(root/'results/extension_extractor_gpu.lock');lock.parent.mkdir(exist_ok=True)
        handle=lock.open('a+b');handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        model=load_upstream(root,k).to(device)
        # Padding and sample-order equivalence before any update.
        lens=sorted(range(len(data['train'])),key=lambda i:len(data['train'][i]['inputs']['text']))
        sample=[data['train'][i] for i in dict.fromkeys((lens[0],lens[len(lens)//2],lens[-1]))]
        model.eval()
        with torch.no_grad():
            group=model(batch(sample,vectors,index,device));singles=torch.cat([model(batch([x],vectors,index,device)) for x in sample])
        assert torch.allclose(group,singles,atol=1e-5,rtol=1e-4),'Batch padding parity failed'
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.)
        from transformers import get_cosine_schedule_with_warmup
        scheduler=get_cosine_schedule_with_warmup(opt,0,protocol['epochs']*math.ceil(len(data['train'])/32))
        start=0;best=-1.;history=[]
        if (out/'last.pt').exists():
            ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
            assert ck['protocol_sha']==sha(pfile);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
            start=ck['epoch'];best=ck['best'];history=ck['history'];torch.set_rng_state(ck['rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
        for epoch in range(start,protocol['epochs']):
            model.train(); tic=time.monotonic(); order=torch.randperm(len(data['train']),generator=torch.Generator().manual_seed(a.seed*100+epoch)).tolist(); loss_sum=0.
            for i in range(0,len(order),32):
                rows=[data['train'][j] for j in order[i:i+32]]
                y=torch.tensor([r['label'] for r in rows],device=device);opt.zero_grad(set_to_none=True)
                loss=nn.functional.cross_entropy(model(batch(rows,vectors,index,device)),y);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();scheduler.step();loss_sum+=loss.item()*len(rows)
            p=predict(model,data['val'],vectors,index,32,device);y=np.array([r['label'] for r in data['val']]);m=metric(y,p,k)
            rec=dict(epoch=epoch+1,loss=loss_sum/len(order),metrics=m,seconds=time.monotonic()-tic,peak_vram_gib=torch.cuda.max_memory_allocated()/2**30);history.append(rec);print(json.dumps(rec),flush=True)
            if m['weighted']>best:
                best=m['weighted'];save(out/'best.pt',model.state_dict());dump(out/'best.json',rec)
            save(out/'last.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch+1,best=best,history=history,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),protocol_sha=sha(pfile)))
            dump(out/'history.json',history)
        dump(out/'complete.json',dict(complete=True,protocol_sha=sha(pfile),checkpoint_sha=sha(out/'best.pt'),test_used=False))
        handle.close();print('TRAIN COMPLETE',out);return
    # Test only after an explicitly completed training run; read test data here, never in train.
    done=json.loads((out/'complete.json').read_text(encoding='utf8'))
    assert done['protocol_sha']==sha(pfile) and done['checkpoint_sha']==sha(out/'best.pt')
    assert json.loads(pfile.read_text(encoding='utf8'))==protocol,'Source or environment changed since training'
    import msvcrt
    lock=(root/'results/extension_extractor_gpu.lock');lock.parent.mkdir(exist_ok=True)
    handle=lock.open('a+b');handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
    from recovery.test_features import prepare
    inp,y,labels,cache,texts=prepare(a.dataset)
    assert labels==spec['labels']
    with np.load(cache/'sentence_erc.npz',allow_pickle=False) as z: tv=torch.from_numpy(z['values'].copy().astype('float32'))
    test=[dict(inputs=x) for x in inp];ti={t:i for i,t in enumerate(texts)}
    model=load_upstream(root,k).to(device);model.load_state_dict(torch.load(out/'best.pt',map_location='cpu',weights_only=True));p=predict(model,test,tv,ti,32,device)
    dest=root/'test_results_nc_revision'/a.dataset/f'DialogInfer-S-adapted_s{a.seed}';dest.mkdir(parents=True,exist_ok=True)
    provenance=dict(dataset=a.dataset,arm='DialogInfer-S-adapted',seed=a.seed,labels=labels,split='test',checkpoint_sha=sha(out/'best.pt'),training_protocol_sha=sha(pfile),test_cache_protocol_sha=sha(cache/'protocol.json'),evaluator_sha=sha(HERE))
    if (dest/'complete.json').exists():
        old=json.loads((dest/'complete.json').read_text(encoding='utf8'));assert old['provenance']==provenance and old['predictions_sha']==sha(dest/'predictions.npz');print('TEST ALREADY COMPLETE',dest);handle.close();return
    np.savez_compressed(dest/'predictions.npz',ids=np.array([x['id'] for x in inp]),y=y,p=p)
    dump(dest/'metrics.json',dict(provenance=provenance,metrics=metric(y,p,k),rows=len(y)))
    dump(dest/'complete.json',dict(complete=True,provenance=provenance,predictions_sha=sha(dest/'predictions.npz')))
    print('TEST COMPLETE',dest,metric(y,p,k));handle.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--dataset',choices=DATASETS,required=True);parser.add_argument('--seed',type=int,choices=SEEDS,required=True);parser.add_argument('--mode',choices=('train','test'),required=True)
    run(parser.parse_args())
