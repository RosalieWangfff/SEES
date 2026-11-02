"""History-only target-response emotion forecasting on an injected DialogueGLP backbone."""
from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn

@dataclass
class HistoryBatch:
    utterances: torch.Tensor
    emotions: torch.Tensor
    speakers: torch.Tensor
    target: torch.Tensor
    valid: torch.Tensor
    response: torch.Tensor
    antecedent: torch.Tensor
    wrong_antecedent: torch.Tensor
    attributes: torch.Tensor
    confidence: torch.Tensor
    links: torch.Tensor
    syntax: Optional[torch.Tensor] = None
    syntax_mask: Optional[torch.Tensor] = None
    wrong_syntax: Optional[torch.Tensor] = None
    wrong_syntax_mask: Optional[torch.Tensor] = None

    def validate(self, classes):
        b,t,_=self.utterances.shape
        assert self.emotions.shape==self.speakers.shape==self.valid.shape==(b,t)
        assert self.target.shape==(b,) and self.valid.dtype==torch.bool
        assert self.response.shape==self.antecedent.shape==self.wrong_antecedent.shape==self.links.shape==self.confidence.shape
        assert self.links.dtype==torch.bool and self.attributes.shape[:2]==self.links.shape
        assert ((self.emotions[self.valid]>=0)&(self.emotions[self.valid]<classes)).all()
        assert torch.isfinite(self.utterances).all() and torch.isfinite(self.attributes).all()
        assert ((self.confidence>=0)&(self.confidence<=1)).all()
        for index in (self.response,self.antecedent,self.wrong_antecedent):
            assert ((index>=0)&(index<t)).all()
            assert self.valid.gather(1,index)[self.links].all()
        assert (self.antecedent[self.links]<self.response[self.links]).all()
        assert (self.wrong_antecedent[self.links]<self.response[self.links]).all()
        assert (self.speakers.gather(1,self.antecedent)[self.links]==self.speakers.gather(1,self.wrong_antecedent)[self.links]).all()

class RoleHistory(nn.Module):
    def __init__(self,dim,hidden,classes):
        super().__init__()
        self.text=nn.Linear(dim,hidden)
        self.target_cell=nn.GRUCell(hidden,hidden)
        self.other_cell=nn.GRUCell(hidden,hidden)
        self.emotion=nn.Embedding(classes,hidden)
        self.emotion_cell=nn.GRUCell(hidden,hidden)
    def forward(self,b):
        x=self.text(b.utterances).tanh();own=b.speakers==b.target[:,None]
        ht=x.new_zeros((len(x),x.shape[-1]));ho=ht.clone();he=ht.clone()
        for i in range(x.shape[1]):
            mt=b.valid[:,i]&own[:,i];mo=b.valid[:,i]&~own[:,i]
            ht=torch.where(mt[:,None],self.target_cell(x[:,i],ht),ht)
            ho=torch.where(mo[:,None],self.other_cell(x[:,i],ho),ho)
            e=self.emotion(torch.where(b.valid[:,i],b.emotions[:,i],torch.zeros_like(b.emotions[:,i])))
            he=torch.where(mt[:,None],self.emotion_cell(e,he),he)
        return ht,ho,he

class TargetResponseModule(nn.Module):
    def __init__(self,dim,classes,attribute_dim,hidden=128,use_syntax=False):
        super().__init__();self.classes=classes;self.hidden=hidden;self.use_syntax=use_syntax
        self.history=RoleHistory(dim,hidden,classes)
        self.content=nn.Linear(dim,hidden)
        self.syntax_role=nn.Embedding(7,hidden)
        self.syntax_query=nn.Linear(attribute_dim,hidden)
        self.message=nn.Sequential(nn.Linear(7*hidden+attribute_dim,hidden),nn.Tanh(),nn.Dropout(.2))
        self.delta=nn.Linear(hidden,classes)
        self.gate=nn.Sequential(nn.Linear(2*hidden+classes,64),nn.Tanh(),nn.Linear(64,1))
        nn.init.zeros_(self.delta.weight);nn.init.zeros_(self.delta.bias)
    @staticmethod
    def pool(x,w):
        return (x*w[...,None]).sum(1)/w.sum(1,keepdim=True).clamp_min(1)
    def forward(self,b,base_logits,arm="matched",syntax_mode="typed"):
        b.validate(self.classes)
        if arm not in ("baseline","matched","independent","wrong_object"):
            raise ValueError(arm)
        h=self.hidden;own=b.speakers==b.target[:,None]
        active=b.links & ~own.gather(1,b.response) & own.gather(1,b.antecedent)
        weight=active*b.confidence
        if arm=="baseline" or not active.any():
            return base_logits,base_logits.new_zeros((len(base_logits),1)),active
        target,other,emotion=self.history(b)
        index=b.wrong_antecedent if arm=="wrong_object" else b.antecedent
        take=lambda ix:b.utterances.gather(1,ix[...,None].expand(-1,-1,b.utterances.shape[-1]))
        response=self.content(take(b.response)).tanh();obj=self.content(take(index)).tanh()
        if self.use_syntax:
            if b.syntax is None or b.syntax_mask is None:raise ValueError("Syntax inputs required")
            slots=b.syntax;mask=b.syntax_mask
            if arm=="wrong_object":
                if b.wrong_syntax is None or b.wrong_syntax_mask is None:raise ValueError("Aligned wrong-object syntax required")
                slots=torch.stack((slots[:,:,0],b.wrong_syntax),2)
                mask=torch.stack((mask[:,:,0],b.wrong_syntax_mask),2)
            assert slots.shape[-2]==7 and mask.shape==slots.shape[:-1]
            key=self.content(slots).tanh()
            if syntax_mode=="typed":key=key+self.syntax_role.weight
            elif syntax_mode!="bag":raise ValueError(syntax_mode)
            q=self.syntax_query(b.attributes)[:,:,None,None,:]
            a=((key*q).sum(-1)/h**.5).masked_fill(~mask,-1e4).softmax(-1)*mask
            v=(key*a[...,None]).sum(-2);response,obj=v[:,:,0],v[:,:,1]
        context=torch.cat((target,other,emotion),-1)
        if arm=="independent":
            r=self.pool(response,weight);o=self.pool(obj,weight);a=self.pool(b.attributes,weight)
            msg=self.message(torch.cat((r,o,torch.zeros_like(r),torch.zeros_like(r),context,a),-1))
        else:
            c=context[:,None,:].expand(-1,response.shape[1],-1)
            msg=self.pool(self.message(torch.cat((response,obj,response*obj,abs(response-obj),c,b.attributes),-1)),weight)
        gate=self.gate(torch.cat((msg,emotion,base_logits.softmax(-1)),-1)).sigmoid()
        delta=self.delta(msg)*gate*active.any(1)[:,None]
        return base_logits+delta,gate,active

class HistoryAwareGLP(nn.Module):
    """Public backbone + common historical-label input + auditable research module.

    The local token inputs must be constructed from historical text by the data adapter.
    Original GLP has no recurrent padding mask: use equal-length batches or microbatch1.
    """
    def __init__(self,backbone,dim,classes,attribute_dim,hidden=128,use_syntax=False,use_history_labels=True):
        super().__init__();self.backbone=backbone;self.use_history_labels=use_history_labels
        self.gold=nn.Embedding(classes,dim);nn.init.zeros_(self.gold.weight)
        self.relation=TargetResponseModule(dim,classes,attribute_dim,hidden,use_syntax)
    def forward(self,local_inputs,b,arm="matched",syntax_mode="typed"):
        b.validate(self.relation.classes)
        if not b.valid.all():raise ValueError("Public GLP requires unpadded microbatches")
        vectors=b.utterances+(self.gold(b.emotions) if self.use_history_labels else 0)
        global_inputs={"embs":list(vectors.unbind(1)),"speakers":[x.float()[:,None] for x in (b.speakers==b.target[:,None]).unbind(1)]}
        inputs={"inputs1":global_inputs,"inputs2":dict(local_inputs)}
        if hasattr(self.backbone,"dialogue_infer"):
            self.backbone.dialogue_infer.device=vectors.device
        base_logits=self.backbone(inputs)
        logits,gate,active=self.relation(b,base_logits,arm,syntax_mode)
        return {"logits":logits,"base_logits":base_logits,"gate":gate,"active_edges":active}
