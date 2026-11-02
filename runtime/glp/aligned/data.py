import hashlib
import json
import numpy as np
import torch
from .common import ROOT, rows, read, feature_path

class Inputs:
    def __init__(self, dataset, tokenizer):
        self.rows = {s: rows(dataset, s) for s in ('train', 'val')}
        self.spec = read(ROOT/'data'/dataset/'contract.json')
        texts = sorted({t for records in self.rows.values() for r in records for t in r['inputs']['text']})
        feature = feature_path(dataset)
        with np.load(feature, allow_pickle=False) as z:
            expected = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
            assert str(z['sha']) == expected, 'Feature/text order mismatch'
            self.vectors = torch.from_numpy(z['values'].astype('float32'))
        assert self.vectors.shape == (len(texts), self.spec['dim'])
        assert torch.isfinite(self.vectors).all()
        self.index = {s:i for i,s in enumerate(texts)}
        self.tokenizer = tokenizer
        self.tokens = {}
        for split, records in self.rows.items():
            local = []
            for r in records:
                x = r['inputs']
                local.append((' '+tokenizer.sep_token+' ').join(
                    ('I: ' if speaker == x['target'] else 'Other: ')+text
                    for speaker,text in zip(x['speakers'][-2:], x['text'][-2:])))
            enc = tokenizer(local, max_length=512, truncation=True)
            self.tokens[split] = [{k:v[i] for k,v in enc.items()} for i in range(len(local))]

    def one(self, split, i, device):
        # Microbatch one exactly preserves the upstream variable-length LSTM update semantics.
        x = self.rows[split][i]['inputs']
        emb = self.vectors[[self.index[t] for t in x['text']]].to(device)
        speakers = torch.tensor([int(s==x['target']) for s in x['speakers']], device=device).float()
        inputs = dict(inputs1=dict(embs=[e[None] for e in emb],
                                   speakers=[s.reshape(1,1) for s in speakers]),
                      inputs2={k:v.to(device) for k,v in self.tokenizer.pad(
                          [self.tokens[split][i]], return_tensors='pt').items()})
        own = [e for e,s in zip(x['emotions'], x['speakers']) if s==x['target']]
        return inputs, torch.tensor([own or [0]], device=device), torch.tensor([len(own)], device=device)
