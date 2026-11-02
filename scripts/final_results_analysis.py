"""Audit existing test predictions and summarize results without training."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

DATASETS = ('meld', 'dailydialog', 'emorynlp')
SEEDS = (1008, 42, 43)
ARMS = ('G2__dep_typed', 'text', 'DialogInfer-S-adapted', 'G1', 'G_word',
        'G2', 'G2__neg_local', 'G2__shuffle_rel', 'G1__dep_typed',
        'G_word__dep_typed', 'G2__dep_untyped', 'G2__dep_neg_joint')


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--legacy', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    roots = [a.root / 'test_results_nc_revision', a.root / 'test_results']
    if a.legacy:
        roots.append(a.legacy)
    runs, issues = {}, []
    for root in roots:
        if not root.is_dir():
            issues.append({'path': str(root), 'issue': 'test root not found'})
            continue
        for f in sorted(root.glob('*/*/metrics.json')):
            try:
                m, c = load(f), load(f.parent / 'complete.json')
                v = m['provenance']
                if v['dataset'] not in DATASETS or v['arm'] not in ARMS or v['seed'] not in SEEDS:
                    continue
                # Historical 'text' can refer to a different experiment. GLP-text
                # must come from the current server's test results.
                if a.legacy and root == a.legacy and v['arm'] == 'text':
                    continue
                assert v.get('split') == 'test' and c.get('complete') and c['provenance'] == v
                predfile = f.parent / 'predictions.npz'
                assert digest(predfile) == c['predictions_sha']
                with np.load(predfile, allow_pickle=False) as z:
                    ids, y, probs = z['ids'].astype(str), z['y'].astype(int), z['p']
                k = len(v['labels'])
                assert len(set(ids)) == len(ids) == len(y) and probs.shape == (len(y), k)
                assert np.isfinite(probs).all() and (probs >= 0).all() and np.allclose(probs.sum(axis=1), 1, atol=1e-5)
                assert np.all((y >= 0) & (y < k))
                pred = probs.argmax(axis=1)
                scores = {name: float(100*f1_score(y, pred, labels=range(k), average=name, zero_division=0)) for name in ('weighted', 'macro')}
                assert all(abs(scores[x] - m['metrics'][x]) < 1e-6 for x in scores)
                key = (v['dataset'], v['arm'], int(v['seed']))
                item = dict(v=v, ids=ids, y=y, pred=pred, scores=scores, path=str(f), pred_sha=c['predictions_sha'])
                if key in runs:
                    if runs[key] is None or runs[key]['pred_sha'] != item['pred_sha']:
                        issues.append({'path': str(f), 'issue': 'conflicting duplicate; excluded', 'key': list(key)})
                        runs[key] = None
                else:
                    runs[key] = item
            except Exception as e:
                issues.append({'path': str(f), 'issue': f'{type(e).__name__}: {e}'})
    runs = {k: v for k, v in runs.items() if v is not None}
    grouped = defaultdict(list)
    for (ds, arm, seed), item in runs.items():
        grouped[ds, arm].append((seed, item))
    summary, per_class, comparisons = [], [], []
    for (ds, arm), items in sorted(grouped.items()):
        vals = {metric: [i['scores'][metric] for _, i in items] for metric in ('weighted', 'macro')}
        summary.append(dict(dataset=ds, arm=arm, seeds=sorted(s for s, _ in items), n=len(items),
            **{m+'_mean': float(np.mean(v)) for m,v in vals.items()},
            **{m+'_sd': float(np.std(v, ddof=1)) if len(v)>1 else None for m,v in vals.items()}))
        for seed, i in items if arm in ('G2__dep_typed', 'text', 'DialogInfer-S-adapted') else []:
            k = len(i['v']['labels'])
            precision, recall, f1, support = precision_recall_fscore_support(i['y'], i['pred'], labels=range(k), zero_division=0)
            for j, label in enumerate(i['v']['labels']):
                per_class.append(dict(dataset=ds, arm=arm, seed=seed, label=label, support=int(support[j]),
                    precision=float(precision[j]*100), recall=float(recall[j]*100), f1=float(f1[j]*100)))
    class_groups = defaultdict(list)
    for row in per_class:
        class_groups[row['dataset'], row['arm'], row['label']].append(row)
    per_class = [dict(dataset=ds, arm=arm, label=label, n_seeds=len(rows),
                      support_by_seed=[r['support'] for r in rows],
                      f1_mean=float(np.mean([r['f1'] for r in rows])),
                      recall_mean=float(np.mean([r['recall'] for r in rows])))
                 for (ds, arm, label), rows in sorted(class_groups.items())]
    pairs = [('G2__dep_typed','text'), ('G2__dep_typed','DialogInfer-S-adapted'),
             ('G_word','G1'), ('G_word__dep_typed','G1__dep_typed'),
             ('G2__dep_typed','G2'), ('G2__dep_typed','G2__shuffle_rel'), ('G2__neg_local','G2__dep_typed')]
    for ds in DATASETS:
        for left, right in pairs:
            for seed in SEEDS:
                x, y = runs.get((ds,left,seed)), runs.get((ds,right,seed))
                if x is None or y is None:
                    continue
                match = (x['v']['labels'] == y['v']['labels'] and np.array_equal(x['ids'],y['ids']) and np.array_equal(x['y'],y['y']))
                cache_match = x['v'].get('test_cache_protocol_sha') == y['v'].get('test_cache_protocol_sha') and bool(x['v'].get('test_cache_protocol_sha'))
                comparisons.append(dict(dataset=ds, left=left, right=right, seed=seed,
                    same_ordered_ids_and_labels=bool(match), same_test_cache_sha=bool(cache_match),
                    weighted_delta=x['scores']['weighted']-y['scores']['weighted'] if match else None,
                    macro_delta=x['scores']['macro']-y['scores']['macro'] if match else None))
    missing = [{'dataset':ds,'arm':arm,'seed':seed} for ds in DATASETS for arm in ARMS[:7]
               for seed in SEEDS if (ds,arm,seed) not in runs]
    result = dict(roots=[str(r) for r in roots], n_valid_runs=len(runs), summary=summary,
                  paired_seed_comparisons=comparisons, per_class=per_class, missing=missing, issues=issues,
                  note='Matched test predictions do not establish matched training protocol or statistical significance.')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        raise FileExistsError(f'Output already exists: {a.out}')
    content = json.dumps(result, ensure_ascii=False, indent=2)
    a.out.write_text(content, encoding='utf-8')
    print('=== RUN COUNTS ===')
    print(json.dumps({'n_valid_runs': len(runs), 'roots': [str(r) for r in roots]}, ensure_ascii=False))
    for name in ('summary', 'paired_seed_comparisons', 'per_class', 'missing', 'issues'):
        print('=== ' + name.upper() + ' ===')
        for row in result[name]:
            print(json.dumps(row, ensure_ascii=False, separators=(',', ':')))
    print('=== DONE ===', a.out)


if __name__ == '__main__':
    main()
