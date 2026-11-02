"""Exploratory analysis of fixed test predictions by historical dialogue length."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from paired_bootstrap import get_record, SEEDS, LEFT, RIGHTS

DS=('meld','dailydialog','emorynlp')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--legacy',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); assert not a.out.exists()
    output=[]
    for ds in DS:
        base=a.root/'baselines/DialogueGLP/data'
        source=base/'daily_dialogue/test.csv' if ds=='dailydialog' else base/ds/'processed/test_full.csv'
        cp=a.root/'test_cache'/ds/'protocol.json'
        protocol=json.loads(cp.read_text(encoding='utf-8-sig'))
        assert protocol['source_sha']==sha(source)
        df=pd.read_csv(source,dtype={'Dialogue_ID':str})
        sizes=(df.groupby('Dialogue_ID',sort=False).size()-1).to_dict()
        by_arm={}
        for arm in (LEFT,*RIGHTS):
            records=[]
            for seed in SEEDS:
                path=(a.legacy/ds if arm==LEFT else a.root/'test_results_nc_revision'/ds)/f'{arm}_s{seed}'
                record=get_record(path,ds,arm,seed)
                assert record['provenance']['test_cache_protocol_sha']==sha(cp)
                if records:
                    assert np.array_equal(record['ids'],records[0]['ids']) and np.array_equal(record['y'],records[0]['y'])
                    assert record['provenance']['labels']==records[0]['provenance']['labels']
                records.append(record)
            by_arm[arm]=records
        ref=by_arm[LEFT][0]
        for records in by_arm.values():
            assert np.array_equal(ref['ids'],records[0]['ids']) and np.array_equal(ref['y'],records[0]['y'])
            assert ref['provenance']['labels']==records[0]['provenance']['labels']
        lens=np.array([sizes[i] for i in ref['ids']]); assert np.all(lens>=1)
        labels=ref['provenance']['labels']; k=len(labels)
        # Fixed integer ranges, not performance-selected. Zero observations are explicit.
        for name,mask in [('1-3',lens<=3),('4-7',(lens>=4)&(lens<=7)),('8+',lens>=8)]:
            supports=np.bincount(ref['y'][mask],minlength=k)
            row=dict(dataset=ds,history_turns=name,n=int(mask.sum()),label_support=dict(zip(labels,supports.tolist())),models={})
            if mask.any():
                for arm,records in by_arm.items():
                    values=[float(100*f1_score(r['y'][mask],r['pred'][mask],labels=range(k),average='weighted',zero_division=0)) for r in records]
                    row['models'][arm]={'weighted_f1_by_seed':values,'mean':float(np.mean(values)),'sample_sd':float(np.std(values,ddof=1))}
            output.append(row); print(json.dumps(row,ensure_ascii=False),flush=True)
    data={'groups':output,'seeds':list(SEEDS),'script_sha':sha(Path(__file__)),
          'note':'Exploratory test subgroup analysis; fixed boundaries are not training interventions. Distributions and sizes differ; no causal effect, significance or cross-model superiority inferred from subgroup means.'}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DONE',a.out,flush=True)

if __name__=='__main__': main()
