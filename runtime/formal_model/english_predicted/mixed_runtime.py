"""Experimental mixed-length batching; never installed into production sources."""
import inspect
import textwrap
import types
import torch
from formal_model.cuda_runtime import PreparedHistory, CheckedBatch


class MixedHistory(PreparedHistory):
    actual_microbatch = 4

    def groups(self, order, microbatch=4, effective=32):
        size=self.actual_microbatch
        assert size in (4,8,12,24,32) and effective==32
        for original, count in super().groups(order,microbatch,effective):
            # Preserve original flattened traversal and optimizer-window membership.
            ids=[i for pack in original for i in pack]
            yield [ids[j:j+size] for j in range(0,len(ids),size)],count

    def collate(self, ids, graph_data=None, device='cuda'):
        if graph_data is not None:
            raise ValueError('Mixed runtime supports prior/evidence arms only, not dialogue graph arms.')
        lengths=torch.tensor([len(self.indices[i]) for i in ids])
        assert len(ids)>0 and (lengths>0).all()
        width=int(lengths.max())
        def pad(items):
            return torch.nn.utils.rnn.pad_sequence(items,batch_first=True,padding_value=0)
        vectors=pad([self.vectors[self.indices[i]] for i in ids])
        emotions=pad([self.emotions[i] for i in ids])
        speakers=pad([self.speakers[i] for i in ids])
        valid=torch.arange(width)[None,:]<lengths[:,None]
        empty=torch.empty(len(ids),0,dtype=torch.long)
        batch=CheckedBatch(vectors,emotions,speakers,torch.tensor([self.rows[i]['target'] for i in ids]),
            valid,empty,empty,empty,torch.zeros(len(ids),0,1),torch.zeros(len(ids),0),empty.bool())
        tokens=dict(self.tokenizer.pad([self.tokens[i] for i in ids],padding=True,return_tensors='pt'))
        if device!='cpu':
            from dataclasses import fields,replace
            batch=replace(batch,**{f.name:getattr(batch,f.name).to(device) for f in fields(batch) if torch.is_tensor(getattr(batch,f.name))})
            tokens={k:v.to(device) for k,v in tokens.items()}
        return tokens,batch,None


def dialogue_forward(self, embs, speakers, valid=None):
    batch_size=embs[0].shape[0]
    h_n=torch.zeros(batch_size,self.hidden_size,dtype=torch.float,device=embs[0].device)
    c_n=torch.zeros_like(h_n)
    for j,(emb,lambd) in enumerate(zip(embs,speakers)):
        h_o,c_o=h_n,c_n
        h_store,c_store=self.lstm_store(emb,(h_o,c_o))
        h_affect,c_affect=self.lstm_affect(emb,(h_o,c_o))
        h_n=lambd*h_store+(1-lambd)*h_affect
        c_n=lambd*c_store+(1-lambd)*c_affect
        if valid is not None:
            mask=valid[j][:,None]
            h_n=torch.where(mask,h_n,h_o)
            c_n=torch.where(mask,c_n,c_o)
    return self.fc(h_n),c_n


def rewritten(method, replacements):
    func=method.__func__
    source=textwrap.dedent(inspect.getsource(func))
    for old,new in replacements:
        assert source.count(old)==1, 'Unexpected method layout: '+old
        source=source.replace(old,new)
    namespace=dict(func.__globals__)
    exec(compile(source,'<mixed_batch_experiment>','exec'),namespace)
    return types.MethodType(namespace[func.__name__],method.__self__)


def install_model(model):
    if model.text is None:return model
    model.text.backbone.dialogue_infer.forward=types.MethodType(dialogue_forward,model.text.backbone.dialogue_infer)
    model.text.forward=rewritten(model.text.forward,[
        ("if not safe.valid.all():\n        raise ValueError('Use unpadded microbatch1')", "if arm not in ('none', 'backbone_only'):\n        raise ValueError('Unsupported mixed-history arm')"),
        ("base = self.backbone(dict(inputs1=inputs, inputs2=dict(local_inputs)))", "inputs['valid'] = list(safe.valid.unbind(1))\n    base = self.backbone(dict(inputs1=inputs, inputs2=dict(local_inputs)))")])
    model.evidence.forward=rewritten(model.evidence.forward,[
        ("available = counts[words['lookup']] > 0", "available = (counts[words['lookup']] > 0) & words['history_valid']")])
    original=model.forward
    def forward(self,own_emotions,own_lengths,local=None,batch=None,words=None):
        words=dict(words,history_valid=batch.valid)
        return original(own_emotions,own_lengths,local,batch,words)
    model.forward=types.MethodType(forward,model)
    return model
