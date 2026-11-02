"""Paired dialogue-level bootstrap for pre-specified test comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

DATASETS = ('meld', 'dailydialog', 'emorynlp')
SEEDS = (1008, 42, 43)
LEFT = 'G2__dep_typed'
RIGHTS = ('text', 'DialogInfer-S-adapted')


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def get_record(folder, ds, arm, seed):
    m = read(folder/'metrics.json')
    c = read(folder/'complete.json')
    v = m['provenance']
    assert c['complete'] and c['provenance'] == v and v['split'] == 'test'
    assert (v['dataset'], v['arm'], v['seed']) == (ds, arm, seed)
    assert sha(folder/'predictions.npz') == c['predictions_sha']
    with np.load(folder/'predictions.npz', allow_pickle=False) as z:
        ids, y, prob = z['ids'].astype(str), z['y'].astype(int), z['p']
    assert len(set(ids)) == len(ids) == len(y) and prob.shape == (len(y), len(v['labels']))
    assert np.isfinite(prob).all() and (prob >= 0).all() and np.allclose(prob.sum(1),1,atol=1e-5)
    pred = prob.argmax(1)
    score = 100*f1_score(y,pred,labels=range(len(v['labels'])),average='weighted',zero_division=0)
    assert abs(score-m['metrics']['weighted']) < 1e-6
    return {'ids':ids, 'y':y, 'pred':pred, 'provenance':v, 'prediction_sha':c['predictions_sha']}


def weighted_f1(y, pred, k):
    cm = np.bincount(y*k+pred, minlength=k*k).reshape(k,k)
    supports = cm.sum(axis=1)
    den = supports+cm.sum(axis=0)
    f = np.divide(2*np.diag(cm), den, out=np.zeros(k,dtype=float), where=den != 0)
    return 100 * float((supports*f).sum()/len(y))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--legacy',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--replicates',type=int,default=2000)
    a=p.parse_args()
    assert a.replicates >= 1000 and not a.out.exists()
    results=[]
    for ds in DATASETS:
        for right in RIGHTS:
            pairs=[]
            for seed in SEEDS:
                left=get_record(a.legacy/ds/f'{LEFT}_s{seed}', ds, LEFT, seed)
                other=get_record(a.root/'test_results_nc_revision'/ds/f'{right}_s{seed}', ds, right, seed)
                v,w=left['provenance'],other['provenance']
                assert v['labels']==w['labels'] and np.array_equal(left['ids'],other['ids']) and np.array_equal(left['y'],other['y'])
                assert v.get('test_cache_protocol_sha') and v['test_cache_protocol_sha']==w.get('test_cache_protocol_sha')
                if pairs:
                    assert np.array_equal(left['ids'],pairs[0][0]['ids']) and np.array_equal(left['y'],pairs[0][0]['y'])
                    assert v['labels']==pairs[0][0]['provenance']['labels']
                pairs.append((left,other))
            n=len(pairs[0][0]['y']); k=len(pairs[0][0]['provenance']['labels'])
            obs=[weighted_f1(l['y'],l['pred'],k)-weighted_f1(r['y'],r['pred'],k) for l,r in pairs]
            rng=np.random.default_rng(20261028)
            delta=np.empty(a.replicates)
            for b in range(a.replicates):
                idx=rng.integers(0,n,size=n)
                delta[b]=np.mean([weighted_f1(l['y'][idx],l['pred'][idx],k)-weighted_f1(r['y'][idx],r['pred'][idx],k) for l,r in pairs])
            low,high=np.percentile(delta,[2.5,97.5])
            row=dict(dataset=ds,comparison=f'{LEFT} minus {right}',metric='weighted_f1_percentage_points',
                     n_processed_dialogues=n,seeds=list(SEEDS),seed_deltas=obs,mean_delta=float(np.mean(obs)),
                     seed_delta_sample_sd=float(np.std(obs,ddof=1)),ci95_percentile=[float(low),float(high)],
                     bootstrap_replicates=a.replicates,bootstrap_seed=20261028,
                     prediction_sha=[{'seed':s,'left':l['prediction_sha'],'right':r['prediction_sha']} for s,(l,r) in zip(SEEDS,pairs)])
            results.append(row)
            print(json.dumps(row,ensure_ascii=False),flush=True)
    data=dict(results=results,script_sha=sha(Path(__file__)),
              note='Resampling processed Dialogue_ID rows; assumes these units adequately represent independent dialogues. Fixed trained models, same resample across seeds. No adjustment for multiple comparisons; not uncertainty from retraining. Training protocols of older runs are not fully verified.')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    print('DONE',a.out,flush=True)


if __name__=='__main__': main()
