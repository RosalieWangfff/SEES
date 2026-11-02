"""MELD history-label experiment: fixed direct candidate, optional explicit shift gate."""
from dataclasses import replace
import torch
from torch import nn
from formal_model.direct_ablation_glp import DirectAblationGLP


class SoftHistoryEmbedding(nn.Embedding):
    """Read externally supplied historical probabilities, never integer gold placeholders."""
    probabilities = None
    position = None

    def forward(self, indices):
        q = self.probabilities
        if q is None:
            return self.weight.new_zeros((*indices.shape, self.embedding_dim))
        if indices.ndim == 1:
            if self.position is None:
                self.position = 0
            result = q[:, self.position] @ self.weight
            self.position += 1
            return result
        return q @ self.weight


class HistoryConditionGLP(DirectAblationGLP):
    arms = (*DirectAblationGLP.arms, 'gate', 'gate_aux')

    def __init__(self, backbone, condition='gold'):
        super().__init__(backbone)
        assert condition in ('gold', 'predicted', 'text')
        self.condition = condition
        # Class replacement preserves parameters and checkpoint key layout.
        self.gold.__class__ = SoftHistoryEmbedding
        self.candidate.graph_emotion.__class__ = SoftHistoryEmbedding
        self.candidate.scaffold.history.emotion.__class__ = SoftHistoryEmbedding
        self.shift_gate = nn.Linear(48, 1)
        nn.init.zeros_(self.shift_gate.weight)
        nn.init.zeros_(self.shift_gate.bias)

    def forward(self, local_inputs, b, g, arm='direct', history_probs=None):
        if arm not in self.arms:
            raise ValueError(arm)
        if self.condition == 'gold':
            q = torch.nn.functional.one_hot(b.emotions.clamp_min(0), 7).float()
        elif self.condition == 'predicted':
            if history_probs is None:
                raise ValueError('Predicted condition requires aligned probability cache')
            q = history_probs.float()
            assert q.shape == (*b.emotions.shape, 7)
            assert torch.isfinite(q).all() and (q >= 0).all()
            assert torch.allclose(q.sum(-1), torch.ones_like(q[..., 0]), atol=1e-5)
        else:
            q = None
        if arm in ('gate', 'gate_aux') and q is None:
            raise ValueError('Explicit categorical shift requires gold or predicted previous emotion')
        for emb in (self.gold, self.candidate.graph_emotion, self.candidate.scaffold.history.emotion):
            emb.probabilities = q
            emb.position = 0
        # Never pass real historical labels downstream in predicted/text conditions.
        safe = replace(b, emotions=torch.zeros_like(b.emotions)) if self.condition != 'gold' else b
        safe.validate(7)
        if not safe.valid.all():
            raise ValueError('Use unpadded microbatch1')
        vectors = safe.utterances + self.gold(safe.emotions)
        self.backbone.dialogue_infer.device = vectors.device
        inputs = {'embs': list(vectors.unbind(1)),
                  'speakers': [s.float()[:, None] for s in
                               (safe.speakers == safe.target[:, None]).unbind(1)]}
        base = self.backbone({'inputs1': inputs, 'inputs2': dict(local_inputs)})
        candidate_arm = 'direct' if arm in ('gate', 'gate_aux') else arm
        logits, context, _ = self.candidate(safe, base, g, candidate_arm)
        result = {'logits': logits, 'base_logits': base, 'ungated_logits': logits}
        if arm in ('gate', 'gate_aux'):
            own = (safe.speakers == safe.target[:, None]) & safe.valid
            last = torch.where(own, torch.arange(own.shape[1], device=own.device), -1).max(1).values
            active = last >= 0
            previous = q[torch.arange(len(q), device=q.device), last.clamp_min(0)].argmax(-1)
            gate_logit = self.shift_gate(context).squeeze(-1).float()
            # Hierarchical factorization: P(stay), P(change)*P(class | change).
            changed = logits.float().masked_fill(torch.nn.functional.one_hot(previous, 7).bool(), -1e9).softmax(-1)
            shift = gate_logit.sigmoid()
            stay = torch.nn.functional.one_hot(previous, 7).float()
            p = (1-shift[:, None])*stay + shift[:, None]*changed
            final = torch.where(active[:, None], p.clamp_min(1e-8).log(), logits.float())
            result.update(logits=final, gate_logit=gate_logit, gate_active=active, previous_class=previous)
        return result


def training_loss(out, target, arm):
    # Residual penalty remains on ungated logits; log probabilities have a different offset.
    loss = torch.nn.functional.cross_entropy(out['logits'], target)
    loss = loss + .1*(out['ungated_logits']-out['base_logits']).square().mean()
    if arm == 'gate_aux' and out['gate_active'].any():
        active = out['gate_active']
        # In predicted-label mode this means departure from the estimated prior class.
        change = (target != out['previous_class']).float()
        loss = loss + .1*torch.nn.functional.binary_cross_entropy_with_logits(out['gate_logit'][active], change[active])
    return loss
