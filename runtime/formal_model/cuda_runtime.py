"""CUDA batching for this project's four forecasting datasets.

Equal history lengths preserve DialogueGLP recurrent semantics. Token padding
is masked by the local encoder. Effective optimizer groups remain unchanged.
"""
from collections import defaultdict
from dataclasses import fields
import types
import numpy as np
import torch
from formal_model.model import HistoryBatch


class CheckedBatch(HistoryBatch):
    def validate(self, classes):
        # Value checks happen once on CPU in PreparedHistory, not every layer.
        b, t, _ = self.utterances.shape
        assert self.emotions.shape == self.speakers.shape == self.valid.shape == (b, t)
        assert self.target.shape == (b,)


class PreparedHistory:
    def __init__(self, rows, vectors, index, tokenizer, classes):
        self.rows = rows
        self.vectors = torch.from_numpy(np.asarray(vectors, dtype=np.float32))
        assert torch.isfinite(self.vectors).all()
        self.indices, self.speakers, self.emotions, self.topologies = [], [], [], []
        texts = []
        for row in rows:
            n = len(row['text'])
            assert n > 0 and n == len(row['speakers'])
            self.indices.append(torch.tensor([index[t] for t in row['text']]))
            self.speakers.append(torch.tensor(row['speakers'], dtype=torch.long))
            e = torch.tensor(row.get('emotions', [0]*n), dtype=torch.long)
            assert e.shape == (n,) and ((e >= 0) & (e < classes)).all()
            self.emotions.append(e)
            people=sorted(set(row['speakers']))
            mapping={s:n+j for j,s in enumerate(people)}
            u=torch.arange(n,dtype=torch.long)
            r=torch.tensor([mapping[s] for s in row['speakers']])
            self.topologies.append((people,[u,r,u[:-1]],[r,u,u[1:]],
                                    torch.tensor([int(s==row['target']) for s in people])))
            texts.append((' '+tokenizer.sep_token+' ').join(
                ('I: ' if s == row['target'] else 'Other: ')+t
                for s,t in zip(row['speakers'][-2:],row['text'][-2:])))
        self.tokens = []
        for start in range(0,len(texts),512):
            pack = tokenizer(texts[start:start+512], max_length=512, truncation=True)
            self.tokens.extend([{k:v[i] for k,v in pack.items()} for i in range(len(pack['input_ids']))])
        self.tokenizer = tokenizer

    def groups(self, order, microbatch=4, effective=24):
        # Reorder ONLY inside an optimizer step; no dropped or repeated samples.
        for start in range(0,len(order),effective):
            window = order[start:start+effective]
            buckets = defaultdict(list)
            for i in window:
                buckets[len(self.indices[i])].append(i)
            packs = []
            for bucket in buckets.values():
                bucket.sort(key=lambda i: len(self.tokens[i]['input_ids']))
                for j in range(0,len(bucket),microbatch):
                    packs.append(bucket[j:j+microbatch])
            yield packs, len(window)

    def collate(self, ids, graph_data=None, device='cuda'):
        rows = [self.rows[i] for i in ids]
        indices = torch.stack([self.indices[i] for i in ids])
        b,t = indices.shape
        empty = torch.empty(b,0,dtype=torch.long)
        batch = CheckedBatch(self.vectors[indices],torch.stack([self.emotions[i] for i in ids]),
            torch.stack([self.speakers[i] for i in ids]),torch.tensor([r['target'] for r in rows]),
            torch.ones(b,t,dtype=torch.bool),empty,empty,empty,torch.zeros(b,0,1),torch.zeros(b,0),empty.bool())
        def move(x):
            return x.pin_memory().to(device,non_blocking=True) if device == 'cuda' else x
        batch = CheckedBatch(**{f.name:move(getattr(batch,f.name)) if torch.is_tensor(getattr(batch,f.name))
                                else getattr(batch,f.name) for f in fields(batch)})
        tokens = {k:move(v) for k,v in self.tokenizer.pad([self.tokens[i] for i in ids],
                                      padding=True,return_tensors='pt').items()}
        graph = None
        if graph_data is not None:
            graph = graph_data.collate(rows)
            # Compute graph metadata on CPU, with no GPU scalar extraction.
            src, dst, role_types, positions, node_parts = [[],[],[]],[[],[],[]],[],[],[]
            offset = 0
            for i,row_id in enumerate(ids):
                people,source,dest,kind=self.topologies[row_id]
                role_types.append(kind)
                node_parts.append((i, len(people)))
                for j in range(3):
                    src[j].append(source[j]+offset);dst[j].append(dest[j]+offset)
                positions.append(torch.arange(t)+offset)
                offset += 2*t+len(people)
            graph.update(edge_src=[torch.cat(x) for x in src],edge_dst=[torch.cat(x) for x in dst],
                         role_types=torch.cat(role_types),node_positions=torch.cat(positions),node_parts=node_parts)
            graph = {k:[move(x) for x in v] if k in ('edge_src','edge_dst') else
                     move(v) if torch.is_tensor(v) else v for k,v in graph.items()}
        return tokens,batch,graph


def graph_context(self,b,g,arm):
    x = self.utterance(b.utterances).tanh().float()+self.encode_words(g,self.syntax_mode)
    emo = self.graph_emotion(b.emotions.clamp_min(0))
    if self.ablation != 'no_emotion': x=x+emo
    roles=self.role_kind(g['role_types']); pieces=[]; cursor=0
    for i,n in g['node_parts']:
        pieces.extend((x[i],roles[cursor:cursor+n],emo[i])); cursor+=n
    h=torch.cat(pieces)
    for _ in range(2):
        message=self.graph_self(h)
        for kind,(layer,src,dst) in enumerate(zip(self.relations,g['edge_src'],g['edge_dst'])):
            if not src.numel() or (kind<2 and self.ablation=='no_role') or (kind==2 and self.ablation=='no_time'): continue
            value=layer(h[src]).to(h.dtype)
            agg=torch.zeros_like(h).index_add(0,dst,value)
            count=h.new_zeros(len(h)).index_add(0,dst,h.new_ones(len(dst)))
            message=message+agg/count.clamp_min(1)[:,None]
        h=message.tanh()
    return h[g['node_positions']].reshape_as(x)


def install_graph_runtime(model):
    model.candidate.graph_context = types.MethodType(graph_context,model.candidate)
