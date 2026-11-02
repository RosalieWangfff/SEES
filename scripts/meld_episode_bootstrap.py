"""MELD paired bootstrap clustered by season and episode; no training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from paired_bootstrap import get_record, weighted_f1, SEEDS, LEFT, RIGHTS


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--legacy', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--replicates', type=int, default=2000)
    a = p.parse_args()
    assert not a.out.exists() and a.replicates >= 1000
    source = a.root/'baselines/DialogueGLP/data/meld/processed/test_full.csv'
    cache_protocol = a.root/'test_cache/meld/protocol.json'
    protocol = json.loads(cache_protocol.read_text(encoding='utf-8-sig'))
    assert protocol['source_sha'] == sha(source), 'MELD test source does not match test cache protocol'
    frame = pd.read_csv(source, dtype={'Dialogue_ID': str})
    assert frame[['Dialogue_ID','Season','Episode']].notna().all().all()
    grouped = frame.groupby('Dialogue_ID', sort=False)
    assert (grouped['Season'].nunique() == 1).all() and (grouped['Episode'].nunique() == 1).all()
    ids_to_cluster = {str(key): (str(g['Season'].iloc[0]), str(g['Episode'].iloc[0])) for key,g in grouped}
    results = []
    for right in RIGHTS:
        pairs=[]
        for seed in SEEDS:
            left=get_record(a.legacy/'meld'/f'{LEFT}_s{seed}', 'meld', LEFT, seed)
            other=get_record(a.root/'test_results_nc_revision'/'meld'/f'{right}_s{seed}', 'meld', right, seed)
            x,y=left['provenance'],other['provenance']
            assert x['labels']==y['labels'] and np.array_equal(left['ids'],other['ids']) and np.array_equal(left['y'],other['y'])
            assert x.get('test_cache_protocol_sha') and x['test_cache_protocol_sha']==y.get('test_cache_protocol_sha')
            assert x['test_cache_protocol_sha'] == sha(cache_protocol), 'Predictions use a different test cache protocol'
            if pairs:
                assert np.array_equal(left['ids'],pairs[0][0]['ids']) and np.array_equal(left['y'],pairs[0][0]['y'])
            pairs.append((left,other))
        ids=pairs[0][0]['ids']; assert all(x in ids_to_cluster for x in ids)
        clusters={}
        for i,key in enumerate(ids): clusters.setdefault(ids_to_cluster[key],[]).append(i)
        chunks=[np.asarray(v,dtype=int) for _,v in sorted(clusters.items())]
        k=len(pairs[0][0]['provenance']['labels'])
        obs=[weighted_f1(l['y'],l['pred'],k)-weighted_f1(r['y'],r['pred'],k) for l,r in pairs]
        rng=np.random.default_rng(20261028); deltas=np.empty(a.replicates)
        for b in range(a.replicates):
            selected=rng.integers(0,len(chunks),size=len(chunks))
            ix=np.concatenate([chunks[i] for i in selected])
            deltas[b]=np.mean([weighted_f1(l['y'][ix],l['pred'][ix],k)-weighted_f1(r['y'][ix],r['pred'][ix],k) for l,r in pairs])
        low,high=np.percentile(deltas,[2.5,97.5])
        row={'dataset':'meld','comparison':f'{LEFT} minus {right}', 'mean_delta':float(np.mean(obs)),
             'seed_deltas':obs,'ci95_percentile':[float(low),float(high)],'n_dialogues':len(ids),'n_episodes':len(chunks),
             'replicates':a.replicates,'bootstrap_seed':20261028,
             'test_source_sha':sha(source), 'prediction_sha':[{'seed':seed,'left':l['prediction_sha'],'right':r['prediction_sha']} for seed,(l,r) in zip(SEEDS,pairs)]}
        results.append(row); print(json.dumps(row,ensure_ascii=False),flush=True)
    note='Episode-cluster resampling is a sensitivity analysis; it assumes episodes are suitable clusters. Fixed models; same resamples across seeds; no multiple-comparison adjustment; source/protocol limitations remain.'
    result={'results':results,'note':note,'script_sha':sha(Path(__file__))}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print('DONE',a.out,flush=True)


if __name__=='__main__': main()
