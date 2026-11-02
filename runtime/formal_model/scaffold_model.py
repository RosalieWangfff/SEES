"""Text-only architecture screen inspired by individual/interaction modeling; not EIP reproduction."""
import torch
from formal_model.model import RoleHistory
from formal_model.predictive_state import PredictiveState
class InteractionHistory(RoleHistory):
 mode='separate'
 def forward(self,b):
  x=self.text(b.utterances).tanh();own=(b.speakers==b.target[:,None])&b.valid
  ht=x.new_zeros((len(x),x.shape[-1]));ho=ht.clone();he=ht.clone()
  for i in range(x.shape[1]):
   mt=own[:,i];mo=b.valid[:,i]&~own[:,i]
   cross=self.mode in ['coupled','coupled_retrieval']
   ti=x[:,i]+ho if cross else x[:,i];oi=x[:,i]+ht if cross else x[:,i]
   next_ht=torch.where(mt[:,None],self.target_cell(ti,ht),ht)
   next_ho=torch.where(mo[:,None],self.other_cell(oi,ho),ho)
   ht,ho=next_ht,next_ho
   e=self.emotion(torch.where(b.valid[:,i],b.emotions[:,i],torch.zeros_like(b.emotions[:,i])))
   he=torch.where(mt[:,None],self.emotion_cell(e,he),he)
  if self.mode in ['retrieval','coupled_retrieval']:
   pos=torch.arange(x.shape[1],device=x.device)[None,:];last=torch.where(own,pos,-1).max(1).values
   query=x[torch.arange(len(x),device=x.device),last.clamp_min(0)]
   query=torch.where((last>=0)[:,None],query,torch.zeros_like(query))
   other=b.valid&~own;score=(torch.nn.functional.normalize(x,dim=-1)*torch.nn.functional.normalize(query,dim=-1)[:,None,:]).sum(-1)
   weights=score.masked_fill(~other,-1e4).softmax(-1)*other;weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
   retrieved=(x*weights[:,:,None]).sum(1);ho=(ho+retrieved)/2
  return ht,ho,he
class Scaffold(PredictiveState):
 def __init__(self,dim,classes,attribute_dim=None,hidden=16,states=4):
  super().__init__(dim,classes,attribute_dim,hidden,states)
  # Reuse initialized parameters: separate arm matches the previous continuous control exactly.
  self.history.__class__=InteractionHistory
 def forward(self,b,base_logits,arm='separate'):
  assert arm in ['separate','coupled','retrieval','coupled_retrieval']
  self.history.mode=arm
  return super().forward(b,base_logits,'continuous')
