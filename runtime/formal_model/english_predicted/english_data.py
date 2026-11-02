"""English occurrence-level ERC inputs, aligned to existing forecasting contracts."""
import hashlib
import json
from pathlib import Path
import pandas as pd

LABELS = {'meld':['anger','disgust','fear','joy','neutral','sadness','surprise'],
 'emorynlp':['Neutral','Joyful','Peaceful','Powerful','Scared','Mad','Sad'],
 'dailydialog':['no emotion','anger','disgust','fear','happiness','sadness','surprise']}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def paths(root,dataset):
    raw={s:root/'payload/erc_english'/dataset/f'{s}.csv' for s in ('train','val')}
    forecast={s:root/'formal_model/data/dailydialog'/f'{s}.jsonl' if dataset=='dailydialog' else
              root/'baselines/DialogueGLP/data'/dataset/'processed'/f'{s}_full.csv'
              for s in ('train','val')}
    return raw,forecast


def forecasts(root,dataset):
    raw,prepared=paths(root,dataset)
    if dataset=='dailydialog':
        return {s:[dict(r['inputs'],y=r['label']) for r in map(json.loads,prepared[s].read_text(encoding='utf-8').splitlines())]
                for s in ('train','val')}
    rows={}
    for split,path in prepared.items():
        output=[]
        frame=pd.read_csv(path)
        for key,group in frame.groupby('Dialogue_ID',sort=True):
            values=group.sort_values('Utterance_ID').to_dict('records')
            if len(values)<2:continue
            history=values[:-1]; target=values[-1]['Speaker']
            speakers=list(dict.fromkeys([v['Speaker'] for v in history]+[target]))
            output.append(dict(id=str(key),text=[str(v['Utterance']) for v in history],
                speakers=[speakers.index(v['Speaker']) for v in history],target=speakers.index(target),
                emotions=[LABELS[dataset].index(v['Emotion']) for v in history],y=LABELS[dataset].index(values[-1]['Emotion'])))
        rows[split]=output
    return rows


def aligned(root,dataset):
    raw,_=paths(root,dataset)
    result={}; aliases={}; prepared=forecasts(root,dataset)
    for split,path in raw.items():
        rows=[];dialogs={};starts={};parents={}
        frame=pd.read_csv(path)
        assert not frame.duplicated(['Dialogue_ID','Utterance_ID']).any()
        for key,group in frame.groupby('Dialogue_ID',sort=True):
            values=group.sort_values('Utterance_ID').to_dict('records')
            key=str(key);dialogs[key]=values;parents[key]=key
            for i,v in enumerate(values):starts.setdefault(str(v['Utterance']),[]).append((key,i))
        def find(key):
            while parents[key]!=key:
                parents[key]=parents[parents[key]];key=parents[key]
            return key
        selected={}
        windows={}
        if dataset!='dailydialog':
            processed=pd.read_csv(paths(root,dataset)[1][split])
            windows={str(k):g.sort_values('Utterance_ID').to_dict('records') for k,g in processed.groupby('Dialogue_ID')}
        for r in prepared[split]:
            # Target text is used only to recover source occurrence identity, never as ERC input for history.
            window=windows.get(str(r['id']))
            expected_text=[str(v['Utterance']) for v in window] if window is not None else r['text']
            expected_labels=[LABELS[dataset].index(v['Emotion']) for v in window] if window is not None else r['emotions']
            candidates=[]
            for key,start in starts.get(r['text'][0],[]):
                if dataset=='dailydialog' and key!=str(r['id']):continue
                values=dialogs[key][start:start+len(expected_text)]
                if len(values)!=len(expected_text):continue
                if [str(v['Utterance']) for v in values]!=expected_text:continue
                if [LABELS[dataset].index(v['Emotion']) for v in values]!=expected_labels:continue
                if window is not None and [str(v['Speaker']) for v in values]!=[str(v['Speaker']) for v in window]:continue
                candidates.append((key,start))
            assert candidates,(dataset,split,r['id'],'unmatched original history')
            # Ambiguous duplicate sequences are kept together, never split across folds.
            representative=candidates[0][0]
            for key,_ in candidates:parents[find(key)]=find(representative)
            selected[str(r['id'])]=candidates[0]
        aliases[split]={str(r['id']):[f"{selected[str(r['id'])][0]}:{selected[str(r['id'])][1]+i}" for i in range(len(r['text']))]
                        for r in prepared[split]}
        for key,values in dialogs.items():
            fold=int(hashlib.sha256(find(key).encode()).hexdigest()[:16],16)%5
            for i,v in enumerate(values):rows.append((f'{key}:{i}',str(v['Utterance']),LABELS[dataset].index(v['Emotion']),fold))
        lookup={r[0]:r for r in rows};assert len(lookup)==len(rows)
        for r in prepared[split]:
            assert len(r['text'])==len(r['emotions'])==len(r['speakers'])
            for i,(text,label) in enumerate(zip(r['text'],r['emotions'])):
                assert lookup[aliases[split][str(r['id'])][i]][1:3]==(text,label),(dataset,split,r['id'],i)
        result[split]=rows
    return LABELS[dataset],result,aliases


def load_records(root,dataset):
    labels,records,_=aligned(root,dataset)
    return labels,records


def full_extractor(root,dataset):
    folder=root/'payload/erc_english'/dataset
    metadata=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    checkpoint=(root/metadata['checkpoint_relative']).resolve()
    assert checkpoint.is_relative_to(root.resolve())
    assert digest(checkpoint)==metadata['checkpoint_sha'], 'Full ERC checkpoint differs'
    assert metadata['labels']==LABELS[dataset]
    raw,prepared=paths(root,dataset)
    assert {s:digest(p) for s,p in raw.items()}==metadata['raw_sources']
    assert {s:digest(p) for s,p in prepared.items()}==metadata['forecast_sources']
    return folder,checkpoint
