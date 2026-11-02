"""Read-only negation-edge coverage and optional subgroup analysis.

Uses the actual structure-matrix annotation rule and valid-edge filters.
Requires existing caches; does not create features, train, or run inference.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

DATASETS = ('meld', 'dailydialog', 'emorynlp')
ARMS = ('G2__neg_local', 'G2__dep_typed', 'G2', 'G2__shuffle_rel')
SEEDS = (1008, 42, 43)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def csv_write(path, rows, columns):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def edge_count(words, valid, annotations, punct_index):
    rel, neg = annotations(words)
    assert len(valid) == len(words) == len(rel) == len(neg)
    count = 0
    for i, w in enumerate(words):
        h = int(w['head'])
        if h >= 0:
            assert h < len(words), 'Invalid dependency head'
            assert words[h]['sentence'] == w['sentence'], 'Cross-sentence edge'
            if bool(valid[i]) and bool(valid[h]) and int(rel[i]) != punct_index and bool(neg[i]):
                count += 1
    return count


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--splits', nargs='+', choices=('train', 'val', 'test'), default=('train', 'val', 'test'))
    a = p.parse_args()
    root, out = a.root.resolve(), a.out.resolve()
    assert root.is_dir(), 'Project root missing'
    assert not out.exists(), 'Output folder already exists; choose a new folder'
    sys.path[:0] = [str(root), str(root / 'glp')]
    from aligned.common import rows as train_rows
    from recovery.test_features import rows as test_rows
    from formal_model.english_structure_matrix.data import annotations, RELATIONS

    out.mkdir(parents=True)
    coverage, subgroup, missing, paired, masks = [], [], [], [], {}
    for ds in DATASETS:
        for split in a.splits:
            if split == 'test':
                records, labels, names, raw_path = test_rows(ds)
                folder = root / 'test_cache' / ds
                rs = [(x['id'], x['text'], int(y)) for x, y in zip(records, labels)]
            else:
                records = train_rows(ds, split)
                rs = [(x['inputs']['id'], x['inputs']['text'], int(x['label'])) for x in records]
                folder = root / 'formal_model' / 'data' / (ds + '_dependency_v1')
            pp, cp = folder / 'parses.jsonl', folder / 'word_encoding_complete.json'
            if not pp.is_file() or not cp.is_file():
                missing.append({'dataset': ds, 'split': split, 'reason': 'Missing parse or aligned-word cache'})
                continue
            done = read(cp)
            if not done.get('complete') or digest(pp) != done['parse_sha']:
                raise ValueError(f'Parse cache hash mismatch: {folder}')
            wanted = {t for _, texts, _ in rs for t in texts}
            counts = {}
            with pp.open(encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    text = item['text']
                    if text not in wanted:
                        continue
                    assert text not in counts, 'Duplicate parse text'
                    key = hashlib.sha256(text.encode()).hexdigest()
                    with np.load(folder / 'word_features' / (key + '.npz'), allow_pickle=False) as z:
                        assert str(z['parse_sha']) == done['parse_sha']
                        assert str(z['checkpoint_sha']) == done['checkpoint_sha']
                        counts[text] = edge_count(item['words'], z['valid'], annotations, RELATIONS.index('punct'))
            absent = wanted - counts.keys()
            if absent:
                raise ValueError(f'{ds}/{split}: {len(absent)} missing parsed texts; not counted as zero')
            ids = [str(i) for i, _, _ in rs]
            assert len(ids) == len(set(ids)), 'Duplicate prediction instance ID'
            edges = np.array([sum(counts[t] for t in ts) for _, ts, _ in rs], dtype=np.int64)
            n_utter = sum(len(ts) for _, ts, _ in rs)
            hit_utter = sum(sum(counts[t] > 0 for t in ts) for _, ts, _ in rs)
            coverage.append(dict(dataset=ds, split=split, instances=len(rs), positive_instances=int((edges > 0).sum()),
                                 instance_rate=float((edges > 0).mean()) if len(edges) else 0,
                                 utterance_occurrences=n_utter, positive_utterance_occurrences=hit_utter,
                                 positive_utterance_rate=hit_utter / n_utter if n_utter else 0,
                                 active_neg_edges=int(edges.sum()), parse_sha=done['parse_sha']))
            if split == 'test':
                masks[ds] = (ids, np.array([y for _, _, y in rs]), edges > 0, folder)
            print(f'COVERAGE {ds}/{split}: {int((edges > 0).sum())}/{len(edges)} instances', flush=True)

    for ds, (ids, y, mask, folder) in masks.items():
        cache_protocol = folder / 'protocol.json'
        if not cache_protocol.is_file():
            missing.append({'dataset': ds, 'split': 'test', 'reason': 'Missing test cache protocol'})
            continue
        cache_sha = digest(cache_protocol)
        pred_sets = {}
        for arm in ARMS:
            for seed in SEEDS:
                directory = root / 'test_results_nc_revision' / ds / f'{arm}_s{seed}'
                files = [directory / n for n in ('metrics.json', 'complete.json', 'predictions.npz')]
                if not all(x.is_file() for x in files):
                    missing.append({'dataset': ds, 'split': 'test', 'reason': f'Missing test predictions: {arm} seed {seed}'})
                    continue
                metric, done = map(read, files[:2])
                prov = metric['provenance']
                if not (done.get('complete') and done['provenance'] == prov and digest(files[2]) == done['predictions_sha']):
                    raise ValueError(f'Test artifact hash/provenance mismatch: {directory}')
                if (prov['dataset'], prov['arm'], prov['seed'], prov['split'], prov['test_cache_protocol_sha']) != (ds, arm, seed, 'test', cache_sha):
                    raise ValueError(f'Test identity/cache mismatch: {directory}')
                with np.load(files[2], allow_pickle=False) as z:
                    pred_ids, truth, prob = z['ids'].astype(str), z['y'].copy(), z['p'].copy()
                assert pred_ids.tolist() == ids and np.array_equal(truth, y), f'ID or label mismatch: {directory}'
                assert np.isfinite(prob).all() and prob.shape == (len(y), len(prov['labels']))
                pred_sets[(arm, seed)] = prob.argmax(axis=1)
                for group, select in (('has_neg_edge', mask), ('no_neg_edge', ~mask)):
                    n = int(select.sum())
                    metrics = {k: (float(100 * f1_score(y[select], pred_sets[(arm, seed)][select],
                        labels=list(range(len(prov['labels']))), average=k, zero_division=0)) if n else None)
                        for k in ('weighted', 'macro')}
                    subgroup.append(dict(dataset=ds, arm=arm, seed=seed, group=group, n=n,
                                         weighted=metrics['weighted'], macro=metrics['macro'],
                                         supported_classes=int(len(np.unique(y[select]))) if n else 0))
        for seed in SEEDS:
            if ('G2__neg_local', seed) not in pred_sets or ('G2', seed) not in pred_sets:
                continue
            for group, select in (('has_neg_edge', mask), ('no_neg_edge', ~mask)):
                n = int(select.sum())
                if not n:
                    continue
                # Same test IDs/labels and cache checked; protocol equivalence is not implied.
                pair = {'dataset': ds, 'seed': seed, 'group': group, 'n': n}
                for key in ('weighted', 'macro'):
                    def score(arm):
                        return 100 * f1_score(y[select], pred_sets[(arm, seed)][select],
                            labels=list(range(prob.shape[1])), average=key, zero_division=0)
                    pair['delta_' + key] = float(score('G2__neg_local') - score('G2'))
                paired.append(pair)

    csv_write(out / 'coverage.csv', coverage, ('dataset','split','instances','positive_instances','instance_rate','utterance_occurrences','positive_utterance_occurrences','positive_utterance_rate','active_neg_edges','parse_sha'))
    csv_write(out / 'subgroup_metrics.csv', subgroup, ('dataset','arm','seed','group','n','weighted','macro','supported_classes'))
    csv_write(out / 'paired_neg_vs_no_structure.csv', paired, ('dataset','seed','group','n','delta_weighted','delta_macro'))
    (out / 'missing.json').write_text(json.dumps(missing, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'README.txt').write_text('Observational subgroup analysis, not causal evidence. Fixed full label set for macro F1; low-support groups are unstable. Missing predictions are not zeros. Same test cache/IDs/labels verified; training protocols and cross-batch comparability must be audited separately. No training or model modifications.\n', encoding='utf-8')
    print('DONE:', out, 'missing items:', len(missing), flush=True)


if __name__ == '__main__':
    main()
