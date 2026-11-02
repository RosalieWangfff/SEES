"""Opt-in, isolated four-epoch CPED/DailyDialog utterance extractors."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

from formal_model.extension_data import ExtensionData, ROOT
from formal_model.extension_extractor_data import utterances


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_replace(source, destination):
    for attempt in range(15):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if attempt == 14:
                raise
            time.sleep(.1 * (attempt + 1))


def asset_hashes(asset):
    files = sorted(p for p in asset.rglob('*') if p.is_file() and
                   p.suffix in ('.json', '.txt', '.bin', '.safetensors', '.model'))
    assert any(p.suffix in ('.bin', '.safetensors') for p in files), 'Missing actual model weights'
    for index in asset.glob('*.index.json'):
        manifest = json.loads(index.read_text(encoding='utf-8'))
        for name in set(manifest.get('weight_map', {}).values()):
            shard = asset / name
            assert not Path(name).is_absolute() and '..' not in Path(name).parts and shard in files, 'Unhashed/missing weight shard'
    return {str(p.relative_to(asset)): digest(p) for p in files}


def verify_replay(saved_p, actual_p, saved_y, actual_y, saved_ids, actual_ids, metrics, classes):
    import numpy as np
    from sklearn.metrics import f1_score
    assert np.array_equal(saved_y, actual_y) and np.array_equal(saved_ids, actual_ids)
    assert saved_p.shape == actual_p.shape == (len(actual_y), classes)
    assert np.isfinite(actual_p).all() and np.isfinite(saved_p).all()
    assert np.allclose(actual_p.sum(1), 1., atol=1e-6, rtol=0)
    assert np.allclose(saved_p, actual_p, atol=1e-7, rtol=1e-6), 'Best-checkpoint probability replay differs'
    assert np.array_equal(saved_p.argmax(1), actual_p.argmax(1)), 'Replay predicted classes differ'
    recomputed = {a: float(100*f1_score(actual_y, actual_p.argmax(1), labels=range(classes), average=a, zero_division=0))
                  for a in ('weighted', 'macro')}
    assert all(abs(recomputed[a]-metrics[a]) <= 1e-8 for a in recomputed), 'Replay F1 differs'
    return dict(passed=True, metrics=recomputed, max_probability_difference=float(np.abs(saved_p-actual_p).max()),
                probability_atol=1e-7, probability_rtol=1e-6, identical_predictions=True)


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    atomic_replace(tmp, path)


def gpu_idle():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                            capture_output=True, text=True, check=True)
    pids = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if not pids:
        return
    # WDDM also lists Codex's idle on-device service as a compute client.
    # Only exempt that identified service when the GPU is effectively idle.
    import psutil
    blocked = []
    for pid in pids:
        try:
            process = psutil.Process(pid)
            exe = process.exe().replace('\\', '/').lower()
            args = process.cmdline()
            codex_service = ('/windowsapps/openai.codex_' in exe and
                exe.endswith('/app/chatgpt.exe') and
                '--utility-sub-type=on_device_model.mojom.OnDeviceModelService' in args)
            if not codex_service:
                blocked.append(f'{pid}: {process.name()}')
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            blocked.append(f'{pid}: inaccessible process')
    if blocked:
        raise RuntimeError('GPU has another compute client: ' + ', '.join(blocked))
    usage = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    for line in usage.stdout.splitlines():
        memory, utilization = [int(value.strip()) for value in line.split(',')]
        if memory > 256 or utilization > 5:
            raise RuntimeError('Codex GPU service is not idle; retry when GPU is idle: ' + line)
    print('GPU preflight: only idle Codex on-device service detected; continuing.', flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--dataset', required=True, choices=('cped', 'dailydialog'))
    cli.add_argument('--out', type=Path)
    cli.add_argument('--probe', action='store_true', help='32 train/16 val utterances, one epoch, isolated --out required')
    cli.add_argument('--execute', action='store_true')
    args = cli.parse_args()
    standard = Path(f'D:/FEPD-experiments/extension_{args.dataset}_v1')
    if args.probe:
        assert args.out is not None and args.out.resolve() != standard.resolve(), 'Probe needs an independent output directory'
    out = args.out or standard
    data = ExtensionData(args.dataset)
    records, sources = {}, {}
    for split in ('train', 'val'):
        records[split], sources[split] = utterances(args.dataset, split)
    full_counts = {s: len(r) for s, r in records.items()}
    if args.probe:
        records = {s: r[:32 if s == 'train' else 16] for s, r in records.items()}
    texts = data.texts[:8] if args.probe else data.texts
    epochs = 1 if args.probe else 4
    optimizer_steps = epochs * math.ceil(len(records['train']) / 16)
    print(json.dumps(dict(dataset=args.dataset, execute=args.execute, probe=args.probe,
        output=str(out), epochs=epochs, lr=5e-6, effective_batch=16, microbatch=1,
        seed=42, scheduler='cosine', warmup_steps=0, optimizer_steps=optimizer_steps,
        utterances={s: len(r) for s,r in records.items()}, full_utterances=full_counts,
        cache_texts=len(texts), full_cache_texts=len(data.texts), test_loaded=False), indent=2), flush=True)
    if not args.execute:
        return
    out.mkdir(parents=True, exist_ok=True)
    # Cooperating extractor processes share one GPU launch lock across output roots.
    import msvcrt
    lock_root = Path('D:/FEPD-experiments')
    lock_root.mkdir(parents=True, exist_ok=True)
    global_lock = (lock_root / 'extension_extractor_gpu.lock').open('a+b')
    global_lock.seek(0)
    msvcrt.locking(global_lock.fileno(), msvcrt.LK_NBLCK, 1)
    run_lock = (out / 'run.lock').open('a+b')
    run_lock.seek(0)
    msvcrt.locking(run_lock.fileno(), msvcrt.LK_NBLCK, 1)
    gpu_idle()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import numpy as np
    import torch
    from sklearn.metrics import f1_score
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
    from huggingface_hub import snapshot_download
    from formal_model.extension_extractor_model import ExtensionExtractor
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    asset = Path(data.spec['asset'])
    if not asset.is_dir():
        asset = Path(snapshot_download(data.spec['asset'], local_files_only=True))
    source_names = ('train_extension_extractor.py', 'extension_data.py',
                    'extension_extractor_data.py', 'extension_extractor_model.py')
    protocol = dict(dataset=args.dataset, labels=data.labels, spec=data.spec,
        prepared_source_sha256=data.sources, utterance_sources=sources,
        historical_text_sha256=data.text_sha, source_sha256={n: digest(ROOT/'formal_model'/n) for n in source_names},
        asset_sha256=asset_hashes(asset), epochs=epochs, formal_epochs=4,
        lr=5e-6, microbatch=1, accumulation=16, seed=42, weight_decay=0., scheduler='cosine',
        warmup_steps=0, optimizer_steps=optimizer_steps,
        replay_tolerance=dict(probability_atol=1e-7, probability_rtol=1e-6, f1_atol=1e-8),
        precision='bf16', max_tokens=512, gradient_clip=1., deterministic=True,
        selection='validation Weighted-F1; earliest tie; Macro-F1 from same checkpoint',
        probe=args.probe, counts={s: len(r) for s,r in records.items()}, cache_texts=len(texts),
        test_used=False, torch=torch.__version__, cuda=torch.version.cuda,
        device=torch.cuda.get_device_name())
    protocol_path = out/'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text(encoding='utf-8')) == protocol, 'Protocol changed; use a new output directory'
    else:
        assert not any(out.glob('*.pt')), 'Orphan checkpoints without protocol'
        save_json(protocol_path, protocol)
    protocol_sha = digest(protocol_path)
    if not args.probe and (out/'extractor_complete.json').exists():
        done = json.loads((out/'extractor_complete.json').read_text(encoding='utf-8'))
        replay = json.loads((out/'checkpoint_replay.json').read_text(encoding='utf-8'))
        assert done['checkpoint_replay_verified'] and replay['passed']
        assert done['replay_sha256'] == digest(out/'checkpoint_replay.json')
        assert done['protocol_sha256'] == replay['protocol_sha256'] == protocol_sha
        assert done['checkpoint_sha256'] == replay['checkpoint_sha256']
        assert done['cache_sha256'] == digest(out/'features.npz')
        data.load_sentence_features(out)
        print('Verified existing completed extractor and cache', flush=True)
        return
    spec = dict(data.spec, asset=str(asset))
    tok = AutoTokenizer.from_pretrained(str(asset), local_files_only=True)
    model = ExtensionExtractor(spec)
    model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0., foreach=False)
    scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=optimizer_steps)
    resume = out/'resume.pt'
    best, history, start, best_file = -1., [], 0, None

    def save_torch(path, value):
        tmp = path.with_suffix('.tmp.pt')
        torch.save(value, tmp)
        atomic_replace(tmp, path)

    if resume.exists():
        state = torch.load(resume, map_location='cpu', weights_only=True)
        assert state['protocol_sha256'] == protocol_sha
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start, best, history = state['epoch'], state['best'], state['history']
        best_file = state['best_file']
        assert digest(out/best_file) == state['best_checkpoint_sha256']
        random.setstate(state['python_rng'])
        ns = state['numpy_rng']
        np.random.set_state((ns[0], np.array(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        del state

    def pack(text):
        return tok(text, truncation=True, max_length=512, return_tensors='pt').to('cuda')

    def validation_probabilities():
        model.eval()
        probabilities = []
        with torch.inference_mode():
            for _, _, text, _ in records['val']:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(pack(text))
                probabilities.append(logits.float().softmax(-1)[0].cpu().numpy())
        return np.asarray(probabilities)

    for epoch in range(start, epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        order = torch.randperm(len(records['train']), generator=torch.Generator().manual_seed(4200+epoch)).tolist()
        total_loss = 0.
        for step, index in enumerate(order):
            _, _, text, label = records['train'][index]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(pack(text))
                loss = torch.nn.functional.cross_entropy(logits, torch.tensor([label], device='cuda'))
            assert torch.isfinite(loss)
            (loss/min(16, len(order)-(step//16)*16)).backward()
            total_loss += float(loss.detach())
            if (step+1) % 16 == 0 or step+1 == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                opt.step()
                opt.zero_grad(set_to_none=True)
                scheduler.step()
            if step % 100 == 0:
                save_json(out/'status.json', dict(pid=os.getpid(), stage='train', epoch=epoch+1, step=step+1, total=len(order)))
        p = validation_probabilities()
        y = np.asarray([r[3] for r in records['val']])
        assert np.isfinite(p).all()
        metrics = {a: float(100*f1_score(y, p.argmax(1), labels=range(data.spec['classes']), average=a, zero_division=0))
                   for a in ('weighted', 'macro')}
        history.append(dict(epoch=epoch+1, metrics=metrics, loss=total_loss/len(order)))
        if metrics['weighted'] > best:
            best = metrics['weighted']
            best_file = f'best_epoch_{epoch+1}.pt'
            save_torch(out/best_file, model.state_dict())
            prediction_path = out/f'best_epoch_{epoch+1}.npz'
            prediction_tmp = prediction_path.with_suffix('.tmp.npz')
            with prediction_tmp.open('wb') as stream:
                np.savez_compressed(stream, p=p, y=y,
                    ids=np.array([json.dumps([r[0], r[1]]) for r in records['val']]))
            atomic_replace(prediction_tmp, prediction_path)
        ns = np.random.get_state()
        save_torch(resume, dict(protocol_sha256=protocol_sha, epoch=epoch+1, best=best, history=history,
            best_file=best_file, best_checkpoint_sha256=digest(out/best_file),
            model=model.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict(),
            python_rng=random.getstate(), numpy_rng=(ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()))
        save_json(out/'history.json', history)
        print(json.dumps(history[-1]), flush=True)
    assert len(history) == epochs
    shutil.copyfile(out/best_file, out/'extractor_best.tmp.pt')
    atomic_replace(out/'extractor_best.tmp.pt', out/'extractor_best.pt')
    shutil.copyfile((out/best_file).with_suffix('.npz'), out/'validation_predictions.tmp.npz')
    atomic_replace(out/'validation_predictions.tmp.npz', out/'validation_predictions.npz')
    best_record = max(history, key=lambda row: row['metrics']['weighted'])
    save_json(out/'best.json', dict(epoch=best_record['epoch'], metrics=best_record['metrics']))
    model.load_state_dict(torch.load(out/'extractor_best.pt', map_location='cpu', weights_only=True))
    model.eval()
    ck_sha = digest(out/'extractor_best.pt')
    replay_p = validation_probabilities()
    with np.load(out/'validation_predictions.npz', allow_pickle=False) as z:
        replay = verify_replay(z['p'], replay_p, z['y'], np.array([r[3] for r in records['val']]),
            z['ids'], np.array([json.dumps([r[0], r[1]]) for r in records['val']]), best_record['metrics'], data.spec['classes'])
    replay.update(checkpoint_sha256=ck_sha, protocol_sha256=protocol_sha)
    save_json(out/'checkpoint_replay.json', replay)
    values = np.empty((len(texts), data.spec['dim']), dtype=np.float32)
    with torch.inference_mode():
        for index, text in enumerate(texts):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                value = model.feature(pack(text))
            values[index] = value[0].float().cpu().numpy()
            if index % 100 == 0:
                save_json(out/'status.json', dict(pid=os.getpid(), stage='sentence_cache', done=index+1, total=len(texts)))
    assert np.isfinite(values).all()
    text_sha = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
    cache_name = 'probe_features.npz' if args.probe else 'features.npz'
    with (out/(cache_name+'.tmp')).open('wb') as stream:
        np.savez_compressed(stream, values=values, sha=text_sha, checkpoint_sha256=ck_sha)
    atomic_replace(out/(cache_name+'.tmp'), out/cache_name)
    with np.load(out/cache_name, allow_pickle=False) as z:
        assert z['values'].shape == values.shape and np.isfinite(z['values']).all()
        assert str(z['sha']) == text_sha and str(z['checkpoint_sha256']) == ck_sha
    completion = dict(complete=not args.probe, diagnostic_only=args.probe, checkpoint_file='extractor_best.pt',
        checkpoint_sha256=ck_sha, protocol_sha256=protocol_sha, cache_sha256=digest(out/cache_name),
        epochs=epochs, texts=len(texts), checkpoint_replay_verified=True,
        replay_sha256=digest(out/'checkpoint_replay.json'), test_used=False, efficacy_verified=False)
    save_json(out/('probe_complete.json' if args.probe else 'extractor_complete.json'), completion)
    if not args.probe:
        data.load_sentence_features(out)
    if resume.exists():
        resume.unlink()
    for checkpoint in out.glob('best_epoch_*.pt'):
        checkpoint.unlink()
        checkpoint.with_suffix('.npz').unlink(missing_ok=True)
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == '__main__':
    main()
