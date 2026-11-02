"""Full-backbone wrapper for the selected syntax-attention/graph candidates.

Separate from the running original candidate sweep. This module has not yet
passed a GPU integration probe or produced full-training efficacy evidence.
"""
from torch import nn
from formal_model.attention_graph import AttentionGraph


class AttentionCandidateGLP(nn.Module):
    arms = AttentionGraph.arms

    def __init__(self, backbone, dim=1024, classes=7, hidden=16):
        super().__init__()
        assert dim == 1024 and classes == 7
        self.backbone = backbone
        self.classes = classes
        self.gold = nn.Embedding(classes, dim)
        nn.init.zeros_(self.gold.weight)
        self.candidate = AttentionGraph(hidden)

    def forward(self, local_inputs, b, g, arm='attention_graph'):
        assert arm in self.arms
        b.validate(self.classes)
        if not b.valid.all():
            raise ValueError('Upstream GLP requires unpadded microbatches')
        vectors = b.utterances + self.gold(b.emotions)
        global_inputs = {
            'embs': list(vectors.unbind(1)),
            'speakers': [s.float()[:, None] for s in
                         (b.speakers == b.target[:, None]).unbind(1)],
        }
        self.backbone.dialogue_infer.device = vectors.device
        base = self.backbone({'inputs1': global_inputs, 'inputs2': dict(local_inputs)})
        logits = self.candidate(b, base, g, arm)[0]
        return {'logits': logits, 'base_logits': base}
