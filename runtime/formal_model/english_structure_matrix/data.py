import json
import numpy as np
import torch
from formal_model.english_predicted.adapter import CoreData as OriginalData
from formal_model.core_prior.data import ROOT

# Fixed UD base relations; subtypes collapse to their base, unseen labels -> dep.
RELATIONS = ('acl','advcl','advmod','amod','appos','aux','case','cc','ccomp','clf',
 'compound','conj','cop','csubj','dep','det','discourse','dislocated','expl','fixed',
 'flat','goeswith','iobj','list','mark','nmod','nsubj','nummod','obj','obl','orphan',
 'parataxis','punct','reparandum','root','vocative','xcomp','neg')


def annotations(words):
    relations = np.array([RELATIONS.index(w['dependency'].split(':')[0])
        if w['dependency'].split(':')[0] in RELATIONS else RELATIONS.index('dep') for w in words], dtype=np.int64)
    neg = np.zeros(len(words), dtype=bool)
    for i,w in enumerate(words):
        token = w['text'].lower().replace('’', "'")
        relation = w['dependency'].split(':')[0]
        head = w['head']
        if not (0 <= head < len(words)) or words[head]['sentence'] != w['sentence']:
            continue
        # Conservative local relation, NOT full scope. Exclude response "No" and "not only".
        local = (token in ('not', "n't", 'never') and relation in ('advmod','neg')) or (token == 'no' and relation == 'det')
        following = [v['text'].lower() for v in words[i+1:i+3] if v['sentence']==w['sentence']]
        if token == 'not' and 'only' in following:
            local = False
        neg[i] = local
    return relations, neg


class CoreData(OriginalData):
    def __init__(self, dataset, need_text):
        super().__init__(dataset, need_text)
        self.syntax = {}
        with (ROOT/f'formal_model/data/{dataset}_dependency_v1/parses.jsonl').open(encoding='utf-8') as f:
            for line in f:
                r=json.loads(line)
                cached=self.word_features.records[r['text']]
                assert len(r['words']) == len(cached[0]) == len(cached[1]), 'Parse/word feature alignment mismatch'
                assert np.array_equal(cached[3], [w['head'] for w in r['words']]), 'Parse/cached head mismatch'
                self.syntax[r['text']] = annotations(r['words'])

    def words(self, split, ids, device):
        rows = [{k:self.rows[split][i][k] for k in ('id','text','speakers','target')} for i in ids]
        graph = self.word_features.collate(rows)
        texts = list(dict.fromkeys(t for r in rows for t in r['text']))
        rel, neg = zip(*(self.syntax[t] for t in texts))
        relations = torch.from_numpy(np.concatenate(rel))
        neg = torch.from_numpy(np.concatenate(neg))
        assert len(relations) == len(graph['values']) == len(graph['valid'])
        child = torch.arange(len(relations))
        heads = graph['heads']
        has_head = heads >= 0
        assert (heads[has_head] < len(child)).all()
        parent = heads.clamp_min(0)
        active = has_head & graph['valid'] & graph['valid'][parent]
        active &= relations != RELATIONS.index('punct')
        assert (graph['word_to_utterance'][child[active]] == graph['word_to_utterance'][parent[active]]).all()
        graph['edge_child'], graph['edge_parent'] = child[active], parent[active]
        graph['edge_relation'], graph['edge_negation'] = relations[active], neg[active]
        keys = ('values','valid','word_to_utterance','lookup','utterances',
                'edge_child','edge_parent','edge_relation','edge_negation')
        return {k:graph[k].to(device) if torch.is_tensor(graph[k]) else graph[k] for k in keys}
