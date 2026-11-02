"""Dedicated trainer for the FEPD gold-history prior/evidence comparison."""
import argparse
import json
import math
import os
import platform
import random
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from formal_model.core_prior.data import CoreData, ROOT, EXPERIMENTS
from formal_model.core_prior_ablation.model import ARMS, PriorEvidenceModel
from formal_model.cuda_runtime import PreparedHistory
from formal_model.extension_model import ExtensionModel, SOURCE_FILES
from formal_model.train_extension_extractor import digest, save_json, atomic_replace, asset_hashes


def save_checkpoint(path, value):
    temporary = path.with_suffix('.tmp.pt')
    torch.save(value, temporary)
    atomic_replace(temporary, path)


def gpu_lock():
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    handle = (EXPERIMENTS/'extension_extractor_gpu.lock').open('a+b')
    handle.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise SystemExit('GPU queue busy. Finish the current experiments before training core models.')
    return handle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=('meld', 'emorynlp', 'dailydialog', 'cped'))
    parser.add_argument('--arm', required=True, choices=ARMS)
    parser.add_argument('--scaffold', required=True, choices=('separate', 'coupled'))
    parser.add_argument('--seed', type=int, default=1008, choices=(1008, 42, 43))
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--parity-only', action='store_true', help='Diagnose batch parity without training')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.parity_only:
        args.probe = True
    if not args.execute:
        print(json.dumps(vars(args), indent=2)); return
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable. No CPU training fallback.')
    if args.arm.startswith('M') and platform.system() != 'Linux':
        raise SystemExit('Mamba official-kernel experiments require the separately prepared Linux environment.')
    lock = gpu_lock()  # Keep handle alive until process exit.
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    need_text = args.arm != 'P0'
    print('Validating history inputs and existing caches...', flush=True)
    data = CoreData(args.dataset, need_text)
    epochs = data.epochs if need_text else 30
    freeze = data.freeze if need_text else 0
    if args.probe:
        data.rows = {s:rows[:48 if s == 'train' else 16] for s,rows in data.rows.items()}
        epochs, freeze = 1, 0
    prepared, text, backbone_info, assets = {}, None, {}, {}
    if need_text:
        tokenizer = AutoTokenizer.from_pretrained(str(data.asset), local_files_only=True)
        for split, rows in data.rows.items():
            # This text adapter never receives history emotion labels or target y.
            inputs = [{k:r[k] for k in ('id', 'text', 'speakers', 'target')} for r in rows]
            prepared[split] = PreparedHistory(inputs, data.vectors, data.index, tokenizer, data.classes)
        if args.dataset == 'cped':
            from formal_model.chinese_backbone import create_chinese_backbone
            backbone, backbone_info = create_chinese_backbone(classes=data.classes)
        else:
            from formal_model.backbone import create_backbone
            backbone, backbone_info = create_backbone(classes=data.classes, asset=data.asset)
        text = ExtensionModel(backbone, data.dim, data.classes)
        if data.initial is not None:
            saved = torch.load(data.initial, map_location='cpu', weights_only=True, mmap=True)
            text.backbone.load_state_dict({k[5:]:v for k,v in saved.items() if k.startswith('base.')}, strict=True)
            text.gold.weight.data.copy_(saved['gold.weight'])
            del saved
        text.backbone.ei_roberta.model.config.output_hidden_states = False
        text.backbone.ei_roberta.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        assets = asset_hashes(data.asset)
    model = PriorEvidenceModel(text, data.dim, data.classes, args.arm, args.scaffold, args.seed).cuda()
    # Shared learning rates across G1/G2/M1/M2; history encoder has its own fixed rate.
    buckets = {'plm':[], 'global':[], 'text_residual':[], 'prior_evidence':[], 'other':[]}
    rates = dict(plm=5e-6, global_branch=1e-5, text_residual=1e-3, prior_evidence=1e-3, other=5e-6)
    for name, parameter in model.named_parameters():
        group = ('plm' if name.startswith('text.backbone.ei_roberta.') else
                 'global' if name.startswith('text.backbone.dialogue_infer.') else
                 'text_residual' if name.startswith('text.candidate.') else
                 'prior_evidence' if name.startswith(('prior.', 'evidence.')) else 'other')
        buckets[group].append(parameter)
    groups = [dict(params=values, lr=rates['global_branch' if key == 'global' else key], weight_decay=0.)
              for key,values in buckets.items() if values]
    optimizer = torch.optim.AdamW(groups, foreach=False)
    scaler = torch.amp.GradScaler('cuda', enabled=need_text, init_scale=1024.)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, epochs*math.ceil(len(data.rows['train'])/24))
    relative = set('formal_model/'+name for name in SOURCE_FILES)
    relative.update('formal_model/'+name for name in ('cuda_runtime.py', 'public_data.py', 'extension_data.py',
        'dependency_data.py', 'extension_syntax_data.py', 'prepare_extension_syntax.py', 'backbone.py',
        'chinese_backbone.py', 'train_extension_extractor.py'))
    relative.update('formal_model/core_prior/'+name for name in ('model.py', 'data.py', 'train.py'))
    relative.update('baselines/DialogueGLP/model/'+name for name in ('base_models.py', 'ei_roberta.py', 'dialogue_infer.py'))
    relative.update('formal_model/core_prior_ablation/'+name for name in ('model.py', 'train.py'))
    sources = {name:digest(ROOT/name) for name in sorted(relative)}
    versions = dict(torch=str(torch.__version__), numpy=np.__version__, python=platform.python_version(),
                    system=platform.system(), device=torch.cuda.get_device_name())
    if args.arm.startswith('M'):
        from importlib.metadata import version
        versions['mamba_ssm'] = version('mamba-ssm')
    protocol = dict(dataset=args.dataset, arm=args.arm, scaffold=args.scaffold, seed=args.seed,
        epochs=epochs, freeze_plm_epochs=freeze, microbatch=4, effective_batch=24,
        precision='fp16_text_fp32_prior' if need_text else 'fp32', cudnn_tf32=False,
        gradient_scaling=dict(enabled=need_text, initial_scale=1024.),
        history_condition='gold_target_history', labels=data.labels, sources=sources, data_sha=data.hashes,
        asset_sha=assets, backbone=backbone_info, versions=versions, learning_rates=rates,
        loss='cross_entropy_only', prior_hidden=32, mamba_config=dict(d_state=16,d_conv=4,expand=2),
        selection='val Weighted-F1; earliest tie; same-checkpoint Macro-F1', test_used=False,
        diagnostic_only=args.probe, word_input='all valid cached words; no syntax types or graph edges',
        p0_budget='30 epochs without PLM; P0 is a separately optimized simple baseline',
        initialization='shared B_s42' if data.initial is not None and need_text else 'fresh',
        parameters=dict(total=sum(p.numel() for p in model.parameters()),
                        prior=sum(p.numel() for p in model.prior.parameters())))
    if platform.system() == 'Windows' and args.arm in ('G_word', 'G_utterance') and not args.probe:
        reference_path = EXPERIMENTS/'core_prior/windows'/args.scaffold/args.dataset/f'G2_s{args.seed}'/'protocol.json'
        reference = json.loads(reference_path.read_text(encoding='utf-8'))
        for key, expected in reference.items():
            if key not in ('arm', 'sources'):
                assert protocol[key] == expected, f'Baseline protocol mismatch before training: {key}'
        assert all(sources[name] == expected for name, expected in reference['sources'].items()), 'Baseline source mismatch'
    suffix = '_parity_v3' if args.parity_only else '_probe_v3' if args.probe else ''
    out = EXPERIMENTS/'core_prior_ablation'/platform.system().lower()/args.scaffold/args.dataset/(f'{args.arm}_s{args.seed}'+suffix)
    out.mkdir(parents=True, exist_ok=True)
    pp = out/'protocol.json'
    if pp.exists():
        assert json.loads(pp.read_text(encoding='utf-8')) == protocol, 'Protocol changed; do not overwrite existing results.'
    else:
        save_json(pp, protocol)
    if (out/'complete.json').exists():
        done = json.loads((out/'complete.json').read_text())
        assert done['protocol_sha'] == digest(pp)
        assert done['checkpoint_sha'] == digest(out/'best.pt') and done['prediction_sha'] == digest(out/'predictions.npz')
        print('Verified completed run; skipped training.', flush=True); return

    def packs(split, order):
        if need_text:
            yield from prepared[split].groups(order, 4, 24)
        else:
            for start in range(0, len(order), 24):
                ids = order[start:start+24]
                yield [ids], len(ids)

    def forward(split, ids, mixed_precision=True):
        own, lengths = data.own_history(split, ids, 'cuda')
        local = batch = words = None
        if need_text:
            local, batch, _ = prepared[split].collate(ids)
            words = data.words(split, ids, 'cuda')
        with torch.autocast('cuda', dtype=torch.float16, enabled=mixed_precision and need_text):
            return model(own, lengths, local, batch, words)

    def evaluate():
        model.eval()
        p = torch.empty(len(data.rows['val']), data.classes, device='cuda')
        prior_p = torch.empty_like(p)
        with torch.inference_mode():
            for batches, _ in packs('val', list(range(len(p)))):
                for ids in batches:
                    output = forward('val', ids)
                    p[ids] = output['logits'].float().softmax(-1)
                    prior_p[ids] = output['prior_logits'].float().softmax(-1)
        return p.cpu().numpy(), prior_p.cpu().numpy()

    resume = out/'resume.pt'
    start, best, history = 0, -1., []
    if resume.exists():
        state = torch.load(resume, map_location='cpu', weights_only=True)
        assert state['protocol_sha'] == digest(pp)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        start, best, history = state['epoch'], state['best'], state['history']
        torch.set_rng_state(state['rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        del state
    y = np.array([r['y'] for r in data.rows['val']])
    val_ids = np.array([str(r['id']) for r in data.rows['val']])
    began = time.perf_counter()
    if args.probe:
        model.eval()
        batches, _ = next(packs('train', list(range(len(data.rows['train'])))))
        ids = max(batches, key=len)
        parity = dict(ids=ids, n=len(ids), efficacy_result=False, training_started=False)
        with torch.inference_mode():
            for precision, mixed, atol, rtol in (('fp16_text_fp32_prior', True, .02, .02), ('fp32', False, 1e-4, 1e-4)):
                batched = forward('train', ids, mixed)
                singles = [forward('train', [i], mixed) for i in ids]
                details = {}
                for key, value in batched.items():
                    a = value.float()
                    b = torch.cat([x[key].float() for x in singles])
                    details[key] = dict(max_abs_difference=float((a-b).abs().max()),
                        mean_abs_difference=float((a-b).abs().mean()),
                        passed=bool(torch.allclose(a,b,atol=atol,rtol=rtol)))
                    if key.endswith('logits'):
                        details[key]['argmax_disagreements'] = int((a.argmax(-1)!=b.argmax(-1)).sum())
                parity[precision] = dict(atol=atol,rtol=rtol,branches=details)
        save_json(out/'batch_parity.json', parity)
        print(json.dumps(parity, indent=2), flush=True)
        print(f'Parity report: {out / "batch_parity.json"}', flush=True)
        if args.parity_only:
            print('Diagnosis finished. No training, optimizer step or efficacy evaluation performed.', flush=True)
            return
        assert all(branch['passed'] for result in (parity['fp16_text_fp32_prior'], parity['fp32'])
                   for branch in result['branches'].values()), 'Batch/single mismatch; inspect saved batch_parity.json'
    for epoch in range(start, epochs):
        model.train()
        if need_text:
            model.text.backbone.ei_roberta.requires_grad_(epoch >= freeze)
        order = torch.randperm(len(data.rows['train']), generator=torch.Generator().manual_seed(args.seed*100+epoch)).tolist()
        epoch_start = last = time.perf_counter()
        seen = 0
        skipped_updates = 0
        total = torch.zeros((), device='cuda')
        for batches, count in packs('train', order):
            optimizer.zero_grad(set_to_none=True)
            for ids in batches:
                output = forward('train', ids)
                target = torch.tensor([data.rows['train'][i]['y'] for i in ids], device='cuda')
                loss = nn.functional.cross_entropy(output['logits'].float(), target)
                scaler.scale(loss*len(ids)/count).backward()
                total += loss.detach()*len(ids)
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=not need_text)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < previous_scale:
                skipped_updates += 1
            else:
                scheduler.step()
            seen += count
            now = time.perf_counter()
            if now-last >= 10 or seen == len(order):
                speed = seen/(now-epoch_start)
                save_json(out/'status.json', dict(pid=os.getpid(), epoch=epoch+1, epochs=epochs, step=seen,
                    total=len(order), samples_per_second=speed, stage='train',
                    allocated_mib=torch.cuda.memory_allocated()/2**20, reserved_mib=torch.cuda.memory_reserved()/2**20))
                print(f'{args.dataset}/{args.arm} epoch {epoch+1}/{epochs} {seen}/{len(order)} ({100*seen/len(order):.1f}%) {speed:.2f} samples/s', flush=True)
                last = now
        probabilities, prior_p = evaluate()
        assert np.isfinite(probabilities).all() and np.isfinite(prior_p).all()
        metrics = {m:100*f1_score(y, probabilities.argmax(1), labels=range(data.classes), average=m, zero_division=0)
                   for m in ('weighted','macro')}
        record = dict(epoch=epoch+1, metrics=metrics, loss=float(total.cpu())/len(order),
                      skipped_updates=skipped_updates, gradient_scale=scaler.get_scale(),
                      seconds=time.perf_counter()-epoch_start)
        history.append(record)
        if metrics['weighted'] > best:
            best = metrics['weighted']
            save_checkpoint(out/'best.pt', model.state_dict())
            np.savez_compressed(out/'predictions.npz', ids=val_ids, y=y, p=probabilities, prior_p=prior_p)
            save_json(out/'best.json', record)
        save_json(out/'history.json', history)
        save_checkpoint(resume, dict(protocol_sha=digest(pp), model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), epoch=epoch+1, best=best, history=history,
            rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()))
        print(json.dumps(record), flush=True)
    model.load_state_dict(torch.load(out/'best.pt', map_location='cpu', weights_only=True))
    replay, replay_prior = evaluate()
    with np.load(out/'predictions.npz', allow_pickle=False) as z:
        assert np.allclose(replay, z['p'], atol=1e-6, rtol=1e-5)
        assert np.array_equal(replay.argmax(1), z['p'].argmax(1)), 'Replay changes discrete predictions'
        assert np.allclose(replay_prior, z['prior_p'], atol=1e-6, rtol=1e-5)
    assert all(digest(ROOT/name) == value for name,value in sources.items()), 'Source changed during training'
    save_json(out/'complete.json', dict(complete=True, diagnostic_only=args.probe, test_used=False,
        checkpoint_replay_verified=True, protocol_sha=digest(pp), checkpoint_sha=digest(out/'best.pt'),
        prediction_sha=digest(out/'predictions.npz'), seconds=time.perf_counter()-began))
    resume.unlink(missing_ok=True)
    save_json(out/'status.json', dict(pid=os.getpid(), stage='complete', epochs=epochs))
    lock.close()


if __name__ == '__main__':
    main()
