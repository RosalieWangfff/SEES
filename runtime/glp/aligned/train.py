"""Windows/8GB matched GLP training. Dev selection only; no test entry point."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import argparse
import math
import platform
import random
import time
import numpy as np
import torch
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from .common import ROOT, DATASETS, read, write, sha, model_path, metrics, feature_path, initial_path
from .data import Inputs
from .batching import many, install, parity
from .model import make_glp, MatchedGLP
from .resume import StopController, rng_state, restore_rng

def save(path, value):
    tmp = path.with_suffix('.tmp.pt')
    torch.save(value, tmp)
    tmp.replace(path)

def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=DATASETS, required=True)
    p.add_argument('--arm', choices=('text', 'history'), required=True)
    p.add_argument('--probe', action='store_true')
    p.add_argument('--microbatch', type=int, choices=(4,8,12,24), default=24)
    p.add_argument('--initialization', choices=('shared','pretrained'), default='shared')
    p.add_argument('--checkpoint-minutes', type=float, default=30,
                   help='Save a resumable optimizer-step checkpoint this often (default: 30).')
    a = p.parse_args()
    assert torch.cuda.is_available(), 'CUDA required; no silent CPU fallback'
    import msvcrt
    lock = (ROOT/'GPU.lock').open('a+b')
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    stop = StopController(a.checkpoint_minutes)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    seed(1008)
    asset = model_path(a.dataset)
    tokenizer = AutoTokenizer.from_pretrained(asset, local_files_only=True)
    data = Inputs(a.dataset, tokenizer)
    classes = len(data.spec['labels'])
    epochs = 5 if a.dataset in ('meld', 'emorynlp') else 16
    freeze = 0 if a.dataset in ('meld', 'emorynlp') else 2
    if a.probe:
        data.rows = {split:sorted(records,key=lambda r:sum(len(t) for t in r['inputs']['text'][-2:]),reverse=True)[:48 if split=='train' else 8] for split,records in data.rows.items()}
        data.tokens = {}
        for split,records in data.rows.items():
            local=[(' '+tokenizer.sep_token+' ').join(('I: ' if sp==r['inputs']['target'] else 'Other: ')+t for sp,t in zip(r['inputs']['speakers'][-2:],r['inputs']['text'][-2:])) for r in records]
            enc=tokenizer(local,max_length=512,truncation=True)
            data.tokens[split]=[{k:v[i] for k,v in enc.items()} for i in range(len(local))]
        epochs, freeze = 1, 0
    suffix = '_pretrained' if a.initialization=='pretrained' else ''
    out = ROOT/'results'/a.dataset/(a.arm+suffix+(f'_probe_b{a.microbatch}' if a.probe else ''))
    out.mkdir(parents=True, exist_ok=True)
    initial = initial_path(a.dataset)
    use_initial = initial.exists() and a.initialization=='shared'
    sources = {str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'aligned').glob('*.py')}
    for name in ('base_models.py', 'ei_roberta.py', 'dialogue_infer.py'):
        path = ROOT/'dev_only/model'/name
        sources[str(path.relative_to(ROOT))] = sha(path)
    protocol = dict(dataset=a.dataset, arm=a.arm, seed=1008, epochs=epochs,
        freeze_epochs=freeze, microbatch=a.microbatch, activation_checkpointing=False, batching="masked_mixed_history_v1", effective_batch=24, max_length=512, local_utterances=2,
        labels=data.spec['labels'], input='gold_target_history' if a.arm=='history' else 'text_speaker_order',
        architecture='upstream GLP BaseModel; Chinese BERT adaptation for CPED',
        selection='dev Weighted-F1; earliest tie; Macro from same checkpoint',
        test_used=False, diagnostic_only=a.probe, initialization='shared B_s42 base weights (historical-gold supervised warm start)' if use_initial else 'pretrained local + random global',
        initial_sha=sha(initial) if use_initial else None, contract=data.spec,
        features_sha=sha(feature_path(a.dataset)), sources=sources,
        assets={f.name:sha(f) for f in asset.iterdir() if f.is_file() and f.suffix in ('.json','.bin','.safetensors','.txt','.model')},
        precision='fp16 GLP / fp32 history; GradScaler 1024', weight_decay=0., clip_norm=1.,
        lr=dict(local=5e-6, global_branch=1e-5, history=1e-3, other=5e-6), scheduler='cosine; no warmup',
        versions=dict(torch=str(torch.__version__), python=platform.python_version(),
                      transformers=__import__('transformers').__version__, gpu=torch.cuda.get_device_name(0)))
    if (out/'protocol.json').exists():
        assert read(out/'protocol.json') == protocol, 'Protocol changed; keep old results and use a fresh handoff version'
    else:
        write(out/'protocol.json', protocol)
    if (out/'complete.json').exists():
        done = read(out/'complete.json')
        assert done['checkpoint_replay_verified'] and done['checkpoint_sha']==sha(out/'best.pt')
        assert done['predictions_sha']==sha(out/'predictions.npz')
        print('Verified completed run; skipped:', out, flush=True)
        return
    base = make_glp(asset, classes, a.dataset=='cped')
    if use_initial:
        checkpoint = torch.load(initial, map_location='cpu', mmap=True, weights_only=True)
        base.load_state_dict({k[5:]:v for k,v in checkpoint.items() if k.startswith('base.')}, strict=True)
        del checkpoint
    base.ei_roberta.model.config.output_hidden_states = False
    base.ei_roberta.model.gradient_checkpointing_disable()
    install(base)
    model = MatchedGLP(base, classes, a.arm=='history').cuda()
    base.dialogue_infer.device = torch.device('cuda')
    buckets = {k:[] for k in ('local','global_branch','history','other')}
    for name,param in model.named_parameters():
        group = ('local' if name.startswith('base.ei_roberta.') else
                 'global_branch' if name.startswith('base.dialogue_infer.') else
                 'history' if name.startswith('prior.') else 'other')
        buckets[group].append(param)
    opt = torch.optim.AdamW([dict(params=v, lr=protocol['lr'][k]) for k,v in buckets.items() if v],
                           weight_decay=0., foreach=False)
    sch = get_cosine_schedule_with_warmup(opt, 0, epochs*math.ceil(len(data.rows['train'])/24))
    scaler = torch.cuda.amp.GradScaler(init_scale=1024.)
    start, best, history = 0, -1., []
    resume_offset = resume_total = resume_skipped = 0
    resume_seconds = 0.
    if (out/'last.pt').exists():
        old = torch.load(out/'last.pt', map_location='cpu', mmap=True, weights_only=False)
        model.load_state_dict(old['model'])
        opt.load_state_dict(old['optimizer']); sch.load_state_dict(old['scheduler'])
        scaler.load_state_dict(old['scaler'])
        best, history = old['best'], old['history']
        if 'next_offset' in old:
            start = old['epoch']
            resume_offset = old['next_offset']
            resume_total = old['total_loss']
            resume_skipped = old['skipped_updates']
            resume_seconds = old['elapsed_seconds']
            restore_rng(old['rng'])
            print(f'Resuming epoch {start+1} at sample {resume_offset}', flush=True)
        else:
            start = old['epoch']+1
        del old
    def save_progress(epoch, next_offset, total, skipped, seconds):
        save(out/'last.pt', dict(model=model.state_dict(), optimizer=opt.state_dict(),
            scheduler=sch.state_dict(), scaler=scaler.state_dict(), epoch=epoch,
            next_offset=next_offset, total_loss=total, skipped_updates=skipped,
            elapsed_seconds=seconds, best=best, history=history, rng=rng_state()))
        stop.saved()
    def forward(split, i):
        inputs, own, lengths = many(data, split, i, 'cuda')
        with torch.autocast('cuda', dtype=torch.float16):
            return model(inputs, own, lengths)
    def evaluate():
        model.eval()
        logits = []
        with torch.inference_mode():
            for start_i in range(0,len(data.rows['val']),a.microbatch):
                i=list(range(start_i,min(start_i+a.microbatch,len(data.rows['val']))))
                if stop.stop_requested():
                    raise InterruptedError('Stop requested during validation')
                logits.extend(forward('val', i).float().cpu().numpy())
        values = np.asarray(logits)
        assert np.isfinite(values).all()
        y = np.array([r['label'] for r in data.rows['val']])
        return values, y, metrics(y, values, classes)
    if a.probe:
        write(out/'batch_parity.json', parity(model,data,forward))
    for epoch in range(start, epochs):
        if not (epoch == start and resume_offset):
            seed(1008*100+epoch)
        model.train()
        model.base.ei_roberta.requires_grad_(epoch>=freeze)
        order = torch.randperm(len(data.rows['train']), generator=torch.Generator().manual_seed(1008*100+epoch)).tolist()
        tic = last_print = time.monotonic()
        total = resume_total if epoch == start else 0.
        skipped = resume_skipped if epoch == start else 0
        elapsed = resume_seconds if epoch == start else 0.
        offset_start = resume_offset if epoch == start else 0
        if stop.stop_requested():
            save_progress(epoch, offset_start, total, skipped, elapsed)
            stop.acknowledge_stop()
            raise SystemExit(75)
        for offset in range(offset_start,len(order),24):
            window = order[offset:offset+24]
            opt.zero_grad(set_to_none=True)
            for pack_start in range(0,len(window),a.microbatch):
                i=window[pack_start:pack_start+a.microbatch]
                pred = forward('train', i)
                y = torch.tensor([data.rows['train'][j]['label'] for j in i], device='cuda')
                loss = torch.nn.functional.cross_entropy(pred, y)
                scaler.scale(loss*len(i)/len(window)).backward()
                total += float(loss.detach())*len(i)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            before = scaler.get_scale()
            scaler.step(opt); scaler.update()
            if scaler.get_scale()<before:
                skipped += 1
            else:
                sch.step()
            seen = offset+len(window)
            if stop.save_due() or stop.stop_requested() or seen == len(order):
                save_progress(epoch, seen, total, skipped, elapsed+time.monotonic()-tic)
                if stop.stop_requested():
                    stop.acknowledge_stop()
                    raise SystemExit(75)
            if time.monotonic()-last_print>15 or seen==len(order):
                seconds = time.monotonic()-tic
                status = dict(epoch=epoch+1, epochs=epochs, seen=seen, total=len(order),
                               samples_per_second=(seen-offset_start)/max(seconds,.001), pid=os.getpid())
                write(out/'status.json', status)
                print(f'{a.dataset}/{a.arm} epoch {epoch+1}/{epochs} {seen}/{len(order)} '
                      f'({100*seen/len(order):.1f}%) {status["samples_per_second"]:.2f} samples/s', flush=True)
                last_print = time.monotonic()
        try:
            logits,y,result = evaluate()
        except InterruptedError:
            save_progress(epoch, len(order), total, skipped, elapsed+time.monotonic()-tic)
            stop.acknowledge_stop()
            raise SystemExit(75)
        record = dict(epoch=epoch+1, metrics=result, loss=total/len(order), skipped_updates=skipped,
                      seconds=elapsed+time.monotonic()-tic, peak_vram_gib=torch.cuda.max_memory_allocated()/2**30)
        history.append(record)
        print(record, flush=True)
        write(out/'history.json', history)
        if result['weighted']>best:
            best=result['weighted']
            save(out/'best.pt', model.state_dict())
            write(out/'best.json', record)
            np.savez(out/'predictions.npz', logits=logits,y=y,
                     ids=np.array([str(r['inputs']['id']) for r in data.rows['val']]))
        save(out/'last.pt', dict(model=model.state_dict(),optimizer=opt.state_dict(),
              scheduler=sch.state_dict(),scaler=scaler.state_dict(),epoch=epoch,best=best,history=history))
        stop.saved()
        resume_offset = resume_total = resume_skipped = 0
        resume_seconds = 0.
    model.load_state_dict(torch.load(out/'best.pt', map_location='cpu', mmap=True, weights_only=True))
    try:
        actual,y,result = evaluate()
    except InterruptedError:
        stop.acknowledge_stop()
        raise SystemExit(75)
    with np.load(out/'predictions.npz') as saved:
        assert np.array_equal(saved['y'],y)
        assert np.array_equal(saved['logits'].argmax(1),actual.argmax(1))
        assert np.allclose(saved['logits'],actual,atol=1e-6,rtol=1e-5), 'Checkpoint replay mismatch'
    assert result==read(out/'best.json')['metrics']
    write(out/'complete.json', dict(complete=True,checkpoint_replay_verified=True,test_used=False,
          diagnostic_only=a.probe,metrics=result,checkpoint_sha=sha(out/'best.pt'),
          predictions_sha=sha(out/'predictions.npz'),protocol_sha=sha(out/'protocol.json')))
    print('Completed with checkpoint replay:', out, flush=True)

if __name__=='__main__':
    main()
