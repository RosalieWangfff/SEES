"""Matched attention/role graph candidates; observed history only."""
from dataclasses import replace
from torch import nn
from formal_model.role_syntax_emotion_graph import RoleSyntaxEmotionGraph
from formal_model.syntax_attention import SyntaxAttention


class AttentionGraph(RoleSyntaxEmotionGraph):
    arms = ('none', 'attention_only', 'graph_only', 'untyped_graph',
            'attention_direct_emotion', 'attention_graph')

    def __init__(self, hidden=16):
        super().__init__(hidden=hidden, readout='direct', syntax_mode='edges')
        self.attn_word = nn.Linear(hidden, hidden, bias=False)
        self.attn_context = nn.Linear(hidden, hidden, bias=False)
        self.attn_score = nn.Linear(hidden, 1, bias=False)

    def pool(self, g, arm):
        kind = 'untyped_attention' if arm == 'untyped_graph' else 'types_attention'
        return SyntaxAttention.pool(self, g, kind)

    def encode_words(self, g, arm='edges'):
        if '_attention_arm' in g:
            return self.pool(g, g['_attention_arm'])[0]
        return super().encode_words(g, arm)

    def forward(self, b, prior, g, arm):
        assert arm in self.arms
        if arm == 'none':
            return self.scaffold(b, prior, 'coupled')
        if arm == 'attention_only':
            context = self.pool(g, arm)[0]
        else:
            inputs = g if arm == 'graph_only' else dict(g, _attention_arm=arm)
            graph_arm = 'direct_emotion' if arm == 'attention_direct_emotion' else 'full'
            context = self.graph_context(b, inputs, graph_arm)
        augmented = replace(b, utterances=b.utterances + self.output(context)*b.valid[:, :, None])
        return self.scaffold(augmented, prior, 'coupled')
