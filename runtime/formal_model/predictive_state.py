"""Emotion-conditioned predictive substates versus equal-parameter continuous coefficients."""
import torch
from torch import nn
from formal_model.model import RoleHistory
class PredictiveState(nn.Module):
 def __init__(self,dim,classes,attribute_dim=None,hidden=16,states=4):
  super().__init__();self.classes=classes;self.states=states
  self.history=RoleHistory(dim,hidden,classes)
  self.assignment=nn.Sequential(nn.Linear(3*hidden,hidden),nn.Tanh(),nn.Linear(hidden,states))
  self.transitions=nn.Parameter(torch.zeros(classes+1,states,classes))
 def forward(self,b,base_logits,arm='latent'):
  assert arm in ['latent','continuous'];b.validate(self.classes)
  context=torch.cat(self.history(b),-1);weights=self.assignment(context)
  own=(b.speakers==b.target[:,None])&b.valid;pos=torch.arange(b.valid.shape[1],device=b.valid.device)[None,:]
  last=torch.where(own,pos,-1).max(1).values;anchor=b.emotions.gather(1,last.clamp_min(0)[:,None]).squeeze(1);anchor=torch.where(last>=0,anchor,torch.full_like(anchor,self.classes))
  table=self.transitions[anchor]
  if arm=='latent':
   state=weights.softmax(-1)
   probability=(state[:,:,None]*table.softmax(-1)).sum(1)
   correction=probability.clamp_min(1e-8).log()
  else:
   state=weights.tanh()/self.states**.5
   correction=(state[:,:,None]*table).sum(1)
  correction=correction-correction.mean(-1,keepdim=True)
  return base_logits+correction,state,own.any(1)
