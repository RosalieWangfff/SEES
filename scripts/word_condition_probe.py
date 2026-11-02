"""Validation-only query sensitivity probe for frozen G1 and G_word models.

Changes historical emotion IDs by a fixed cyclic shift. This is an artificial
stress test, not a realistic intervention, new training or causal proof.
Reports only within-utterance word attention; no raw dialogue is exported.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

DS = ('meld', 'dailydialog', 'emorynlp')
SEEDS = (1008, 42, 43)
ARMS = ('G1', 'G_word')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def query_scores(evidence, values, state, conditioned):
    h = evidence.word(values.float()).tanh().float()
    q = evidence.query(evidence.default_query[None] + (state if conditioned else 0))
    return evidence.word_score((h + q).tanh()).squeeze(-1).float()


def js_bits(p, q):
    p = p.double().clamp_min(1e-12)
    q = q.double().clamp_min(1e-12)
    m = (p + q) / 2
    return float((0.5 * ((p * (p / m).log()).sum() + (q * (q / m).log()).sum()) / np.log(2)).cpu())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--dataset', choices=DS, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    a = parser.parse_args()
    root, out = a.root.resolve(), a.out.resolve()
    assert root.is_dir() and not out.exists(), 'Project missing or output already exists'
    if a.device == 'cuda':
        assert torch.cuda.is_available(), 'CUDA unavailable'
    sys.path[:0] = [str(root), str(root / 'glp')]
    from formal_model.english_predicted.adapter import CoreData
    from formal_model.core_prior_ablation.model import EmotionTrajectory, EvidenceSelection
    from recovery.queue import complete

    manifest_path = root / 'nc_revision_reports' / 'cross_manifest.json'
    runs = [r for r in read(manifest_path) if r['dataset'] == a.dataset and r['arm'] in ARMS and r['seed'] in SEEDS]
    assert len(runs) == 6 and {(r['arm'], r['seed']) for r in runs} == {(arm, seed) for arm in ARMS for seed in SEEDS}
    data = CoreData(a.dataset, True)
    torch.set_num_threads(4)
    device = torch.device(a.device)
    out.mkdir(parents=True)
    summaries, provenance = [], []
    for run in runs:
        folder = Path(run['path'])
        assert complete(folder), f'Checkpoint validation failed: {folder}'
        protocol = read(folder / 'protocol.json')
        assert (protocol['dataset'], protocol['arm'], protocol['seed']) == (a.dataset, run['arm'], run['seed'])
        assert not protocol['test_used'] and not protocol['diagnostic_only']
        for key, value in data.hashes.items():
            assert protocol['data_sha'].get(key) == value, f'Data/cache mismatch: {key}'
        for name, digest in protocol['sources'].items():
            assert sha(root / name) == digest, f'Source mismatch: {name}'
        state_dict = torch.load(folder / 'best.pt', map_location='cpu', mmap=True, weights_only=True)
        prior = EmotionTrajectory(data.classes, 'gru')
        evidence = EvidenceSelection(data.dim, data.classes)
        prior.load_state_dict({k[6:]: v for k, v in state_dict.items() if k.startswith('prior.')}, strict=True)
        evidence.load_state_dict({k[9:]: v for k, v in state_dict.items() if k.startswith('evidence.')}, strict=True)
        del state_dict
        prior, evidence = prior.to(device).eval(), evidence.to(device).eval()
        all_js, all_l1, all_top = [], [], []
        skipped_empty, skipped_words, utterances = 0, 0, 0
        conditioned = run['arm'] == 'G_word'
        with torch.inference_mode():
            for i in range(len(data.rows['val'])):
                own, lengths = data.own_history('val', [i], device)
                if not int(lengths[0]):
                    skipped_empty += 1
                    continue
                altered = own.clone()
                altered[0, :int(lengths[0])] = (altered[0, :int(lengths[0])] + 1) % data.classes
                _, original_state = prior(own, lengths)
                _, altered_state = prior(altered, lengths)
                words = data.words('val', [i], device)
                scores0 = query_scores(evidence, words['values'], original_state, conditioned)
                scores1 = query_scores(evidence, words['values'], altered_state, conditioned)
                sample_js, sample_l1, sample_top = [], [], []
                # Match per-utterance softmax and valid-word masking in EvidenceSelection.
                for u in range(words['utterances']):
                    select = words['valid'] & (words['word_to_utterance'] == u)
                    if int(select.sum()) < 2:
                        continue
                    p = scores0[select].softmax(0)
                    q = scores1[select].softmax(0)
                    sample_js.append(js_bits(p, q))
                    sample_l1.append(float(torch.abs(p-q).sum().cpu()))
                    sample_top.append(float(p.argmax() != q.argmax()))
                if not sample_js:
                    skipped_words += 1
                    continue
                utterances += len(sample_js)
                all_js.append(float(np.mean(sample_js)))
                all_l1.append(float(np.mean(sample_l1)))
                all_top.append(float(np.mean(sample_top)))
                if (i + 1) % 200 == 0:
                    print(a.dataset, run['arm'], run['seed'], 'instances', i+1, flush=True)
        assert all_js, 'No eligible validation histories'
        if not conditioned:
            assert max(all_js) < 1e-12 and max(all_l1) < 1e-7, 'G1 negative control unexpectedly changed'
        summaries.append(dict(dataset=a.dataset, split='val', arm=run['arm'], seed=run['seed'],
            n_instances=len(all_js), n_utterance_occurrences=utterances, skipped_no_own_history=skipped_empty,
            skipped_no_comparable_words=skipped_words, mean_js_bits=float(np.mean(all_js)),
            median_js_bits=float(np.median(all_js)), mean_l1=float(np.mean(all_l1)),
            mean_within_history_top1_change=float(np.mean(all_top))))
        provenance.append(dict(dataset=a.dataset, arm=run['arm'], seed=run['seed'], checkpoint_sha=sha(folder/'best.pt'),
            protocol_sha=sha(folder/'protocol.json'), validation_source_sha=data.hashes['val']))
        print('RESULT', json.dumps(summaries[-1]), flush=True)
        del prior, evidence
    with (out / 'word_condition_sensitivity.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0])); w.writeheader(); w.writerows(summaries)
    (out / 'provenance.json').write_text(json.dumps(dict(runs=provenance, script_sha=sha(Path(__file__)),
        perturbation='add 1 modulo number of classes to each valid target-speaker historical emotion ID',
        split='validation', caution='Artificial stress test; no prediction effectiveness, causal explanation, significance, or realistic emotion transition inferred.'), indent=2), encoding='utf-8')
    print('DONE', out, flush=True)


if __name__ == '__main__':
    main()
