"""Masked variable-history batches for the matched GLP baseline."""
import types
import torch

def many(data, split, ids, device):
    xs = [data.rows[split][i]['inputs'] for i in ids]
    pad = torch.nn.utils.rnn.pad_sequence
    lengths = torch.tensor([len(x['text']) for x in xs], device=device)
    vectors = pad([data.vectors[[data.index[t] for t in x['text']]] for x in xs],batch_first=True).to(device)
    speakers = pad([torch.tensor([int(s==x['target']) for s in x['speakers']],dtype=torch.float32) for x in xs],batch_first=True).to(device)
    valid = torch.arange(vectors.shape[1],device=device)[None,:] < lengths[:,None]
    own = [[e for e,s in zip(x['emotions'],x['speakers']) if s==x['target']] for x in xs]
    labels = pad([torch.tensor(v or [0],dtype=torch.long) for v in own],batch_first=True).to(device)
    tokens = {k:v.to(device) for k,v in data.tokenizer.pad([data.tokens[split][i] for i in ids],return_tensors='pt').items()}
    inputs = dict(inputs1=dict(embs=list(vectors.unbind(1)),speakers=[s[:,None] for s in speakers.unbind(1)],valid=list(valid.unbind(1))),inputs2=tokens)
    return inputs, labels, torch.tensor([len(v) for v in own],device=device)

def dialogue_forward(self, embs, speakers, valid=None):
    h = torch.zeros(len(embs[0]),self.hidden_size,device=embs[0].device)
    c = torch.zeros_like(h)
    for j,(emb,speaker) in enumerate(zip(embs,speakers)):
        hs,cs = self.lstm_store(emb,(h,c))
        ha,ca = self.lstm_affect(emb,(h,c))
        hn,cn = speaker*hs+(1-speaker)*ha,speaker*cs+(1-speaker)*ca
        if valid is None:
            h,c = hn,cn
        else:
            h,c = torch.where(valid[j][:,None],hn,h),torch.where(valid[j][:,None],cn,c)
    return self.fc(h),c

def install(base):
    base.dialogue_infer.forward = types.MethodType(dialogue_forward,base.dialogue_infer)

def parity(model,data,forward):
    # Varied history lengths; dropout disabled. Does not perform optimizer updates.
    candidates=sorted(range(len(data.rows['train'])),key=lambda i:len(data.rows['train'][i]['inputs']['text']))
    ids=list(dict.fromkeys([candidates[0],candidates[len(candidates)//2],candidates[-1]]))
    model.eval()
    with torch.inference_mode():
        grouped=forward('train',ids).float()
        single=torch.cat([forward('train',[i]).float() for i in ids])
    difference=float((grouped-single).abs().max())
    assert torch.allclose(grouped,single,atol=.03,rtol=.03), f'GLP batch/single mismatch: {difference}'
    return dict(ids=ids,max_abs_difference=difference,atol=.03,rtol=.03,passed=True)
