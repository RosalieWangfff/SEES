import torch
from torch import nn
from .data import RELATIONS


class LocalStructure(nn.Module):
    """One-hop bidirectional messages before prior-conditioned word attention."""
    def __init__(self, mode, hidden=32):
        super().__init__()
        self.mode = mode
        self.message = nn.Linear(hidden, hidden, bias=False)
        # Same parameter budget for untyped/typed/negation branches.
        self.relation = nn.Embedding(2*len(RELATIONS)+2, hidden)
        self.output = nn.Linear(hidden, hidden, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, h, words):
        child, parent = words['edge_child'], words['edge_parent']
        rel = words['edge_relation']
        if self.mode == 'neg_local':
            keep = words['edge_negation']
            child, parent, rel = child[keep], parent[keep], rel[keep]
        src, dst = torch.cat((child,parent)), torch.cat((parent,child))
        if self.mode in ('dep_typed', 'dep_neg_joint'):
            types = torch.cat((2*rel,2*rel+1))
        else:
            types = torch.cat((torch.full_like(rel,2*len(RELATIONS)),torch.full_like(rel,2*len(RELATIONS)+1)))
        relation = self.relation(types)
        if self.mode == 'dep_neg_joint':
            # Reuse the two otherwise unused generic-direction rows: no added parameters.
            negative = torch.cat((words['edge_negation'], words['edge_negation']))
            direction = torch.cat((torch.zeros_like(rel), torch.ones_like(rel)))
            relation = relation + self.relation(2*len(RELATIONS)+direction)*negative[:,None]
        message = (self.message(h[src])+relation).tanh().float()
        pooled = h.new_zeros(h.shape).index_add(0,dst,message)
        counts = h.new_zeros(len(h)).index_add(0,dst,h.new_ones(len(dst)))
        correction = self.output(pooled/counts.clamp_min(1)[:,None]).float()
        return h + correction*words['valid'][:,None]
