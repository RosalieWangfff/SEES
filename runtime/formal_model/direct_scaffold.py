"""Direct context residual removes the observed saturated four-basis bottleneck."""
import torch
from torch import nn
from formal_model.scaffold_model import InteractionHistory

class DirectScaffold(nn.Module):
    def __init__(self,dim,classes,hidden=16):
        super().__init__();self.classes=classes
        self.history=InteractionHistory(dim,hidden,classes)
        self.readout=nn.Linear(3*hidden,classes)
        nn.init.zeros_(self.readout.weight);nn.init.zeros_(self.readout.bias)

    def forward(self,b,prior,arm='coupled'):
        b.validate(self.classes);self.history.mode=arm
        context=torch.cat(self.history(b),-1)
        correction=self.readout(context);correction=correction-correction.mean(-1,keepdim=True)
        return prior+correction,context,((b.speakers==b.target[:,None])&b.valid).any(1)
