"""Role-typed word graph on the common frozen DialogueGLP scaffold."""
from dataclasses import replace
import torch
from torch import nn
from formal_model.scaffold_model import Scaffold

class DependencyModule(nn.Module):
    def __init__(self,dim=1024,classes=7,hidden=16,readout='continuous'):
        super().__init__()
        if readout=='continuous':self.scaffold=Scaffold(dim,classes,hidden=hidden)
        elif readout=='direct':
            from formal_model.direct_scaffold import DirectScaffold
            self.scaffold=DirectScaffold(dim,classes,hidden)
        else:raise ValueError(readout)
        self.word=nn.Linear(dim,hidden)
        self.role=nn.Embedding(7,hidden)
        self.self_message=nn.Linear(hidden,hidden)
        self.parent_message=nn.Linear(hidden,hidden,bias=False)
        self.child_message=nn.Linear(hidden,hidden,bias=False)
        self.output=nn.Linear(hidden,dim,bias=False)

    def encode_words(self,g,arm='edges'):
        if arm=='untyped':roles=self.role.weight.mean(0,keepdim=True)
        else:roles=self.role(g['shuffled_roles'] if arm=='shuffled_types' else g['roles'])
        x=(self.word(g['values'].float())+roles).tanh()
        valid=g['valid'];x=x*valid[:,None]
        if arm in ('types','untyped','shuffled_types'):parent=child=x
        else:
            heads=g['heads'] if arm=='edges' else g['shuffled']
            edge=(heads>=0)&valid&valid[heads.clamp_min(0)]
            parent=x[heads.clamp_min(0)]*edge[:,None]
            child=torch.zeros_like(x);counts=x.new_zeros(len(x))
            child.index_add_(0,heads[edge],x[edge]);counts.index_add_(0,heads[edge],torch.ones_like(counts[edge]))
            child=child/counts.clamp_min(1)[:,None]
        h=((self.self_message(x)+self.parent_message(parent)+self.child_message(child)).tanh()*valid[:,None]).float()
        pooled=h.new_zeros((g['utterances'],h.shape[-1]));counts=h.new_zeros(g['utterances'])
        pooled.index_add_(0,g['word_to_utterance'],h)
        counts.index_add_(0,g['word_to_utterance'],valid.float())
        pooled=pooled/counts.clamp_min(1)[:,None]
        return pooled[g['lookup']]

    def forward(self,b,prior,g,arm):
        assert arm in ('none','types','edges','shuffled','untyped','shuffled_types')
        if arm=='none':return self.scaffold(b,prior,'coupled')
        syntax=self.output(self.encode_words(g,arm))*b.valid[:,:,None]
        return self.scaffold(replace(b,utterances=b.utterances+syntax),prior,'coupled')
