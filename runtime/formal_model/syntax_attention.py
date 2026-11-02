"""Utterance-context-conditioned syntax pooling; history only."""
from dataclasses import replace
import torch
from torch import nn
from formal_model.dependency_model import DependencyModule


class SyntaxAttention(DependencyModule):
    arms = ('types_mean', 'untyped_attention', 'shuffled_attention', 'types_attention')

    def __init__(self, hidden=16):
        super().__init__(hidden=hidden, readout='direct')
        self.attn_word = nn.Linear(hidden, hidden, bias=False)
        self.attn_context = nn.Linear(hidden, hidden, bias=False)
        self.attn_score = nn.Linear(hidden, 1, bias=False)

    def pool(self, g, arm):
        assert arm in SyntaxAttention.arms
        roles = (self.role.weight.mean(0, keepdim=True) if arm == 'untyped_attention'
                 else self.role(g['shuffled_roles'] if arm == 'shuffled_attention' else g['roles']))
        valid = g['valid']
        index = g['word_to_utterance']
        x = (self.word(g['values'].float()) + roles).tanh() * valid[:, None]
        h = (self.self_message(x) + self.parent_message(x) + self.child_message(x)).tanh()
        h = h.float() * valid[:, None]
        counts = h.new_zeros(g['utterances']).index_add(0, index, valid.float())
        context = h.new_zeros((g['utterances'], h.shape[-1])).index_add(0, index, h)
        context = context / counts.clamp_min(1)[:, None]
        if arm == 'types_mean':
            weights = valid.float() / counts[index].clamp_min(1)
        else:
            scores = self.attn_score((self.attn_word(h) + self.attn_context(context[index])).tanh()).squeeze(-1)
            scores = scores.masked_fill(~valid, -1e9)
            maxima = scores.new_full((g['utterances'],), -1e9)
            maxima.scatter_reduce_(0, index, scores.detach(), reduce='amax', include_self=True)
            exps = (scores - maxima[index]).exp() * valid.float()
            denom = exps.new_zeros(g['utterances']).index_add(0, index, exps)
            weights = exps / denom[index].clamp_min(1e-12)
        pooled = h.new_zeros(context.shape).index_add(0, index, h * weights[:, None])
        return pooled[g['lookup']], weights

    def forward(self, b, prior, g, arm):
        pooled, _ = self.pool(g, arm)
        syntax = self.output(pooled) * b.valid[:, :, None]
        return self.scaffold(replace(b, utterances=b.utterances + syntax), prior, 'coupled')
