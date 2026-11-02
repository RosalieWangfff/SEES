"""English full-model dialogue-fold ERC; opt-in CUDA execution only."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


from formal_model.english_predicted.english_data import load_records as english_records, full_extractor as english_extractor, paths
DATASET = 'meld'
def load_records(root):
    return english_records(root, DATASET)
def full_extractor(root):
    return english_extractor(root, DATASET)

def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--root', type=Path, default=Path('C:/FEPD_server'))
    cli.add_argument('--execute', action='store_true')
    cli.add_argument('--dataset', required=True, choices=('meld','dailydialog','emorynlp'))
    args = cli.parse_args()
    global DATASET
    DATASET = args.dataset
    root = args.root.resolve()
    labels, records = load_records(root)
    folder, checkpoint = full_extractor(root)
    report = dict(dataset=DATASET, folds=5, epochs_per_fold=4, erc_batch=32,
                  full_model_oof=True, grouping='dialogue content and speakers',
                  counts={s:len(v) for s,v in records.items()}, test_loaded=False,
                  training_started=False, full_extractor=str(checkpoint))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
    from formal_model.english_predicted.extractor import EnglishExtractor as ExtensionExtractor
    from sklearn.metrics import f1_score
    assert torch.cuda.is_available(), 'CUDA required; no CPU training fallback'
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    lock_path = root/'results/extension_extractor_gpu.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open('a+b')
    import msvcrt
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    out = root/'results_predicted_english'/DATASET/'cache'
    out.mkdir(parents=True, exist_ok=True)
    asset = root/'baselines/assets/roberta-large'
    configuration = dict(version=1, dataset=DATASET, labels=labels, folds=5, epochs=4,
        batch=32, lr=5e-6, seed=42, precision='bf16', full_model_oof=True,
        full_checkpoint=digest(checkpoint), code=digest(Path(__file__)),
        input_sources={s:digest(p) for s,p in paths(root,DATASET)[0].items()},
        prepared_sources={s:digest(p) for s,p in paths(root,DATASET)[1].items()},
        assets={p.name:digest(p) for p in asset.iterdir() if p.is_file()}, test_used=False)
    cfg_path = out/'erc_protocol.json'
    if cfg_path.exists():
        assert json.loads(cfg_path.read_text(encoding='utf-8')) == configuration, 'ERC protocol changed'
    else:
        save_json(cfg_path, configuration)
    if (out/'complete.json').exists():
        done = json.loads((out/'complete.json').read_text(encoding='utf-8'))
        assert done['protocol_sha'] == digest(cfg_path)
        assert done['cache_sha'] == digest(out/'history_predictions.npz')
        print('Verified ERC cache complete; skipping.', flush=True)
        return
    tokenizer = AutoTokenizer.from_pretrained(str(asset), local_files_only=True)
    tokens = {}
    for split, rows in records.items():
        tokens[split] = []
        for start in range(0, len(rows), 512):
            pack = tokenizer([r[1] for r in rows[start:start+512]], truncation=True, max_length=512)
            tokens[split].extend([{k:v[j] for k,v in pack.items()} for j in range(len(pack['input_ids']))])
    def batch(split, ids):
        pack = tokenizer.pad([tokens[split][i] for i in ids], padding=True, return_tensors='pt')
        return {k:v.to('cuda') for k,v in pack.items()}
    def new_model():
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        return ExtensionExtractor(dict(asset=str(asset), dim=1024, classes=len(labels))).cuda()
    actual_batch = 32
    def predict(model, split, ids):
        model.eval()
        prediction = np.zeros((len(ids), len(labels)), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(ids), actual_batch):
                selected = ids[start:start+actual_batch]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(batch(split, selected))
                prediction[start:start+len(selected)] = logits.float().softmax(-1).cpu().numpy()
        return prediction
    def save_torch(path, value):
        temporary = path.with_suffix('.tmp.pt')
        torch.save(value, temporary)
        temporary.replace(path)
    longest = sorted(range(len(tokens['train'])), key=lambda i: len(tokens['train'][i]['input_ids']), reverse=True)[:32]
    trials=[]
    for candidate in (32,16,8,4,2,1):
        probe = new_model()
        probe_optimizer = torch.optim.AdamW(probe.parameters(), lr=5e-6, weight_decay=0.)
        torch.cuda.reset_peak_memory_stats()
        passed=False
        logits=loss=target=None
        try:
            for offset in range(0,len(longest),candidate):
                sample=longest[offset:offset+candidate]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=probe(batch('train',sample))
                target=torch.tensor([records['train'][i][2] for i in sample],device='cuda')
                loss=torch.nn.functional.cross_entropy(logits.float(),target)*len(sample)/len(longest)
                loss.backward()
                del logits,loss,target
            torch.nn.utils.clip_grad_norm_(probe.parameters(),1.,error_if_nonfinite=True)
            probe_optimizer.step()
            trials.append(dict(actual_batch=candidate,passed=True,peak_gib=torch.cuda.max_memory_allocated()/2**30))
            actual_batch=candidate
            passed=True
        except torch.cuda.OutOfMemoryError:
            trials.append(dict(actual_batch=candidate,passed=False,reason='CUDA OOM'))
        finally:
            logits=loss=target=None
            probe_optimizer.zero_grad(set_to_none=True)
            del probe,probe_optimizer
            torch.cuda.empty_cache()
        if passed:break
    assert passed,'ERC does not fit even at microbatch1'
    save_json(out/'batch32_probe.json',dict(diagnostic_only=True,effective_batch=32,actual_batch=actual_batch,trials=trials))
    oof = np.zeros((len(records['train']), len(labels)), dtype=np.float32)
    for fold in range(5):
        held = [i for i,r in enumerate(records['train']) if r[3] == fold]
        fit = [i for i,r in enumerate(records['train']) if r[3] != fold]
        assert held and fit and not set(held).intersection(fit)
        output = out/f'fold{fold}'
        output.mkdir(exist_ok=True)
        execution_path=output/'execution.json'
        if execution_path.exists():
            previous=json.loads(execution_path.read_text(encoding='utf-8'))
            assert previous['actual_batch']<=actual_batch,'Resume microbatch does not fit current GPU'
            actual_batch=previous['actual_batch']
        else:
            save_json(execution_path,dict(actual_batch=actual_batch,effective_batch=32))
        if (output/'complete.json').exists():
            done = json.loads((output/'complete.json').read_text(encoding='utf-8'))
            assert done['protocol_sha'] == digest(cfg_path)
            assert done['prediction_sha'] == digest(output/'predictions.npz')
            with np.load(output/'predictions.npz', allow_pickle=False) as z:
                assert np.array_equal(z['ids'], held)
                oof[held] = z['p']
            print(f'Fold {fold} complete; skipped', flush=True)
            continue
        model = new_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 4*math.ceil(len(fit)/32))
        first = 0
        resume = output/'resume.pt'
        if resume.exists():
            state = torch.load(resume, map_location='cpu', weights_only=True)
            assert state['protocol_sha'] == digest(cfg_path)
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            first = state['epoch']
            torch.set_rng_state(state['rng'])
            torch.cuda.set_rng_state_all(state['cuda_rng'])
            del state
        for epoch in range(first, 4):
            model.train()
            order = torch.randperm(len(fit), generator=torch.Generator().manual_seed(4200+epoch)).tolist()
            order = [fit[i] for i in order]
            began = last = time.perf_counter()
            for start in range(0, len(order), 32):
                ids = order[start:start+32]
                optimizer.zero_grad(set_to_none=True)
                for offset in range(0,len(ids),actual_batch):
                    selected=ids[offset:offset+actual_batch]
                    target=torch.tensor([records['train'][i][2] for i in selected],device='cuda')
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits=model(batch('train',selected))
                    loss=torch.nn.functional.cross_entropy(logits.float(),target)*len(selected)/len(ids)
                    assert torch.isfinite(loss),'Nonfinite ERC loss'
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                now = time.perf_counter()
                if now-last >= 15 or start+len(ids) == len(order):
                    report = dict(stage='erc_oof', fold=fold, epoch=epoch+1, seen=start+len(ids),
                                  total=len(order), samples_per_second=(start+len(ids))/(now-began))
                    save_json(out/'status.json', report)
                    print(json.dumps(report), flush=True)
                    last = now
            save_torch(resume, dict(protocol_sha=digest(cfg_path), epoch=epoch+1, model=model.state_dict(),
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()))
        p = predict(model, 'train', held)
        assert np.isfinite(p).all()
        np.savez_compressed(output/'predictions.npz', ids=np.array(held), p=p)
        save_torch(output/'final.pt', model.state_dict())
        save_json(output/'complete.json', dict(protocol_sha=digest(cfg_path),
                  prediction_sha=digest(output/'predictions.npz'), epochs=4, complete=True))
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        oof[held] = p
    model = new_model()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    val = predict(model, 'val', list(range(len(records['val']))))
    assert np.isfinite(oof).all() and np.allclose(oof.sum(1), 1, atol=1e-5)
    assert np.isfinite(val).all() and np.allclose(val.sum(1), 1, atol=1e-5)
    payload = {}
    scores = {}
    for split, p in (('train', oof), ('val', val)):
        payload[split+'_keys'] = np.array([r[0] for r in records[split]])
        payload[split+'_pred'] = p.argmax(1)
        payload[split+'_p'] = p
        scores[split] = {m:100*f1_score([r[2] for r in records[split]], p.argmax(1),
                         labels=range(len(labels)), average=m, zero_division=0) for m in ('weighted','macro')}
    temporary = out/'history_predictions.tmp.npz'
    np.savez_compressed(temporary, **payload)
    temporary.replace(out/'history_predictions.npz')
    save_json(out/'complete.json', dict(complete=True, diagnostic_only=False, dataset=DATASET, labels=labels,
        protocol_sha=digest(cfg_path), cache_sha=digest(out/'history_predictions.npz'),
        prepared_sources=configuration['prepared_sources'], erc_metrics=scores, full_model_oof=True,
        validation_source='existing train-fitted validation-selected ERC', test_used=False))
    print('ERC CACHE COMPLETE', json.dumps(scores), flush=True)


if __name__ == '__main__':
    main()
