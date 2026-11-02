"""No labels or future text enter word graphs or deterministic head corruption."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'formal_model/data/meld_dependency_v1'

class DependencyFeatures:
    def __init__(self,directory=None):
        root=Path(directory) if directory is not None else OUT
        done=json.loads((root/'word_encoding_complete.json').read_text())
        assert done['complete']
        rows=[json.loads(l) for l in (root/'parses.jsonl').read_text(encoding='utf-8').split('\n') if l.strip()]
        self.records={};self.audit={'words':0,'edges':0,'changed_edges':0,'unperturbable_edges':0,'invalid_words':0}
        for r in rows:
            text=r['text'];key=hashlib.sha256(text.encode()).hexdigest()
            with np.load(root/'word_features'/(key+'.npz')) as z:
                assert str(z['parse_sha'])==done['parse_sha'] and str(z['checkpoint_sha'])==done['checkpoint_sha']
                values=z['values'].copy();valid=z['valid'].copy()
            words=r['words'];heads=np.array([w['head'] for w in words],np.int64);shuffled=heads.copy()
            rng=np.random.default_rng(int(key[:16],16))
            for j,w in enumerate(words):
                if heads[j]<0:continue
                self.audit['edges']+=1
                candidates=[k for k,v in enumerate(words) if v['sentence']==w['sentence'] and k not in (j,heads[j]) and valid[k]]
                if valid[j] and valid[heads[j]] and candidates:
                    shuffled[j]=rng.choice(candidates);self.audit['changed_edges']+=1
                else:self.audit['unperturbable_edges']+=1
            self.audit['words']+=len(words);self.audit['invalid_words']+=int((~valid).sum())
            roles=np.array([w['role'] for w in words]);shuffled_roles=roles.copy()
            role_rng=np.random.default_rng(int(key[16:32],16))
            for sentence in sorted({w['sentence'] for w in words}):
                indices=[j for j,w in enumerate(words) if w['sentence']==sentence and valid[j]]
                shuffled_roles[indices]=role_rng.permutation(roles[indices])
            self.audit.setdefault('role_words_changed',0);self.audit['role_words_changed']+=int((roles!=shuffled_roles).sum())
            self.records[text]=(values,valid,roles,heads,shuffled,shuffled_roles)

    def collate(self,rows):
        texts=list(dict.fromkeys(t for r in rows for t in r['text']));index={t:i for i,t in enumerate(texts)}
        arrays=[[] for _ in range(7)];offset=0
        for i,t in enumerate(texts):
            values,valid,roles,heads,shuffled,shuffled_roles=self.records[t]
            for dest,src in zip(arrays,(values,valid,roles,np.where(heads>=0,heads+offset,-1),np.where(shuffled>=0,shuffled+offset,-1),np.full(len(heads),i),shuffled_roles)):dest.append(src)
            offset+=len(heads)
        lookup=np.zeros((len(rows),max(len(r['text']) for r in rows)),np.int64)
        for i,r in enumerate(rows):lookup[i,:len(r['text'])]=[index[t] for t in r['text']]
        out={k:torch.from_numpy(np.concatenate(v)) for k,v in zip(('values','valid','roles','heads','shuffled','word_to_utterance','shuffled_roles'),arrays)}
        out.update(utterances=len(texts),lookup=torch.from_numpy(lookup))
        return out
