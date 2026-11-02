"""Dimension-parametric text-only supplement, retaining the original graph operations.

No old source is edited. Record SOURCE_FILES with the runner's protocol: reused
forward methods remain dependencies, not an independent frozen copy of old code.
"""
from dataclasses import replace
import torch
from torch import nn
from formal_model.model import HistoryBatch
from formal_model.direct_ablation_glp import DirectAblationGraph
from formal_model.role_syntax_emotion_graph import RoleSyntaxEmotionGraph
from formal_model.history_condition_model import SoftHistoryEmbedding, training_loss

SOURCE_FILES = ('extension_model.py', 'model.py', 'direct_ablation_glp.py',
    'role_syntax_emotion_graph.py', 'attention_graph.py', 'syntax_attention.py',
    'dependency_model.py', 'direct_scaffold.py', 'scaffold_model.py',
    'predictive_state.py', 'history_condition_model.py', 'attention_candidate_glp.py')
TEXT_ARMS = ('text_last', 'text_concat')


def history_input(row):
    """Accept exactly ExtensionData.text_inputs output, not a full target record."""
    assert set(row) == {'id', 'text', 'speakers', 'target'}, 'Only history text and speaker identity are inputs'
    assert len(row['text']) == len(row['speakers']) > 0
    assert all(isinstance(text, str) for text in row['text'])
    return row


def local_input(row, tokenizer, device='cpu', sentences=2, arm='direct'):
    row = history_input(row)
    assert sentences > 0
    n = 1 if arm == 'text_last' else len(row['text']) if arm == 'text_concat' else sentences
    parts = [('I: ' if speaker == row['target'] else 'Other: ') + text
             for speaker, text in zip(row['speakers'][-n:], row['text'][-n:])]
    previous = tokenizer.truncation_side
    try:
        if arm in TEXT_ARMS:
            tokenizer.truncation_side = 'left'
        return tokenizer((' ' + tokenizer.sep_token + ' ').join(parts), max_length=512,
                         truncation=True, return_tensors='pt').to(device)
    finally:
        tokenizer.truncation_side = previous


def history_batch(row, vectors, text_index, dim, classes, device='cpu'):
    row = history_input(row)
    utterances = torch.stack([torch.as_tensor(vectors[text_index[text]], device=device) for text in row['text']])
    assert utterances.shape == (len(row['text']), dim)
    empty = torch.empty((1, 0), dtype=torch.long, device=device)
    batch = HistoryBatch(utterances.float().unsqueeze(0),
        torch.zeros((1, len(row['text'])), dtype=torch.long, device=device),
        torch.tensor([row['speakers']], dtype=torch.long, device=device),
        torch.tensor([row['target']], dtype=torch.long, device=device),
        torch.ones((1, len(row['text'])), dtype=torch.bool, device=device),
        empty, empty, empty, torch.zeros((1, 0, 1), device=device),
        torch.zeros((1, 0), device=device), empty.bool())
    batch.validate(classes)
    return batch


class ExtensionGraph(DirectAblationGraph):
    def __init__(self, dim, classes, hidden=16):
        # Same constructor order and forward methods as AttentionGraph, with
        # dimensions propagated to the already-parametric lower-level modules.
        RoleSyntaxEmotionGraph.__init__(self, dim, classes, hidden, readout='direct', syntax_mode='edges')
        self.attn_word = nn.Linear(hidden, hidden, bias=False)
        self.attn_context = nn.Linear(hidden, hidden, bias=False)
        self.attn_score = nn.Linear(hidden, 1, bias=False)
        assert self.role.num_embeddings == 7  # UD categories, not emotion classes.


class ExtensionModel(nn.Module):
    arms = (*DirectAblationGraph.arms, 'backbone_only', *TEXT_ARMS)

    def __init__(self, backbone, dim, classes, hidden=16):
        super().__init__()
        assert (dim, classes) in ((768, 13), (1024, 7))
        self.backbone, self.dim, self.classes = backbone, dim, classes
        self.condition = 'text'
        self.gold = nn.Embedding(classes, dim)
        nn.init.zeros_(self.gold.weight)
        self.candidate = ExtensionGraph(dim, classes, hidden)
        for embedding in (self.gold, self.candidate.graph_emotion, self.candidate.scaffold.history.emotion):
            embedding.__class__ = SoftHistoryEmbedding
        # Retain old initialization/RNG order and state-dict layout at 1024/7.
        self.shift_gate = nn.Linear(3 * hidden, 1)
        nn.init.zeros_(self.shift_gate.weight)
        nn.init.zeros_(self.shift_gate.bias)
        self.text_head = nn.Linear(dim, classes)

    def forward(self, local_inputs, batch, graph, arm='direct'):
        if arm not in self.arms:
            raise ValueError(arm)
        if arm in TEXT_ARMS:
            encoded = self.backbone.ei_roberta(**dict(local_inputs))
            logits = self.text_head(encoded.last_hidden_state[:, 0])
            return dict(logits=logits, base_logits=logits, ungated_logits=logits)
        assert batch.utterances.shape[-1] == self.dim
        for embedding in (self.gold, self.candidate.graph_emotion, self.candidate.scaffold.history.emotion):
            embedding.probabilities = None
            embedding.position = 0
        safe = replace(batch, emotions=torch.zeros_like(batch.emotions))
        safe.validate(self.classes)
        if not safe.valid.all():
            raise ValueError('Use unpadded microbatch1')
        vectors = safe.utterances + self.gold(safe.emotions)
        self.backbone.dialogue_infer.device = vectors.device
        inputs = dict(embs=list(vectors.unbind(1)), speakers=[s.float()[:, None]
                      for s in (safe.speakers == safe.target[:, None]).unbind(1)])
        base = self.backbone(dict(inputs1=inputs, inputs2=dict(local_inputs)))
        assert base.shape == (len(vectors), self.classes)
        mode = 'none' if arm == 'backbone_only' else arm
        logits, _, _ = self.candidate(safe, base, graph, mode)
        if arm == 'backbone_only':
            logits = base
        return dict(logits=logits, base_logits=base, ungated_logits=logits)
