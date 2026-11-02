"""Fixed P0/G1/G2/M1/M2 experiment definitions; no graph or syntax-type inputs."""
import torch
from torch import nn

ARMS = {'P0': ('gru', 'prior_only'), 'G1': ('gru', 'ordinary'),
        'G2': ('gru', 'conditioned'), 'M1': ('mamba', 'ordinary'),
        'M2': ('mamba', 'conditioned')}
ARMS.update({'G_word': ('gru', 'word'), 'G_utterance': ('gru', 'utterance'),
             'M_word': ('mamba', 'word'), 'M_utterance': ('mamba', 'utterance')})
HIDDEN = 32


class EmotionTrajectory(nn.Module):
    def __init__(self, classes, encoder):
        super().__init__()
        self.embedding = nn.Embedding(classes, HIDDEN)
        if encoder == 'gru':
            self.encoder = nn.GRU(HIDDEN, HIDDEN, batch_first=True)
        else:
            try:
                from mamba_ssm import Mamba
            except ImportError as error:
                raise RuntimeError('Official mamba_ssm is required; no substitute SSM is used.') from error
            self.encoder = Mamba(d_model=HIDDEN, d_state=16, d_conv=4, expand=2)
        self.kind = encoder
        self.norm = nn.LayerNorm(HIDDEN)
        self.head = nn.Linear(HIDDEN, classes)

    def forward(self, own_emotions, own_lengths):
        # Right padding is never selected as the final state. Both encoders are causal.
        x = self.embedding(own_emotions)
        encoded = self.encoder(x)
        if self.kind == 'gru':
            encoded = encoded[0]
        row = torch.arange(len(x), device=x.device)
        state = encoded[row, (own_lengths-1).clamp_min(0)]
        state = torch.where((own_lengths > 0)[:, None], state, torch.zeros_like(state))
        state = self.norm(state)
        return self.head(state), state


class FixedScaffold(nn.Module):
    """Preserve original none parameters, changing only the selected update mode."""
    def __init__(self, scaffold, mode):
        super().__init__()
        self.history, self.readout = scaffold.history, scaffold.readout
        self.classes, self.mode = scaffold.classes, mode

    def forward(self, batch, base, arm='coupled'):
        batch.validate(self.classes)
        self.history.mode = self.mode
        context = torch.cat(self.history(batch), -1)
        correction = self.readout(context)
        correction = correction-correction.mean(-1, keepdim=True)
        known = ((batch.speakers == batch.target[:, None]) & batch.valid).any(1)
        return base+correction, context, known


class EvidenceSelection(nn.Module):
    def __init__(self, dim, classes):
        super().__init__()
        self.word = nn.Linear(dim, HIDDEN)
        self.query = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.default_query = nn.Parameter(torch.zeros(HIDDEN))
        self.word_score = nn.Linear(HIDDEN, 1, bias=False)
        self.role = nn.Embedding(2, HIDDEN)
        self.utterance = nn.Linear(HIDDEN, HIDDEN)
        self.utterance_score = nn.Linear(HIDDEN, 1, bias=False)
        self.output = nn.Linear(2*HIDDEN, classes)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, words, prior_state, own_role, conditioned):
        # Both arms train the same query projection. Only its input differs.
        query_input = self.default_query.expand_as(prior_state)
        word_conditioned = conditioned is True or conditioned in ('conditioned', 'word')
        utterance_conditioned = conditioned is True or conditioned in ('conditioned', 'utterance')
        query = self.query(query_input + prior_state if word_conditioned else query_input)
        query_u = self.query(query_input + prior_state if utterance_conditioned else query_input)
        valid, index, u = words['valid'], words['word_to_utterance'], words['utterances']
        h = self.word(words['values'].float()).tanh().float()
        scores = self.word_score((h[None]+query[:, None]).tanh()).squeeze(-1).float()
        scores = scores.masked_fill(~valid[None], -1e9)
        ix = (index[None]+torch.arange(len(query), device=h.device)[:, None]*u).expand_as(scores).reshape(-1)
        maximum = scores.new_full((len(query)*u,), -1e9).scatter_reduce(
            0, ix, scores.detach().reshape(-1), reduce='amax')
        exp = (scores.reshape(-1)-maximum[ix]).exp()*valid.expand(len(query), -1).reshape(-1)
        denominator = exp.new_zeros(len(query)*u).index_add(0, ix, exp)
        weights = exp/denominator[ix].clamp_min(1e-12)
        pooled = h.new_zeros(len(query)*u, HIDDEN).index_add(
            0, ix, h.expand(len(query), -1, -1).reshape(-1, HIDDEN)*weights[:, None])
        utterances = pooled.reshape(len(query), u, HIDDEN).gather(
            1, words['lookup'][..., None].expand(-1, -1, HIDDEN))
        counts = h.new_zeros(u).index_add(0, index, valid.float())
        available = counts[words['lookup']] > 0
        scores_u = self.utterance_score((self.utterance(utterances)+self.role(own_role.long())+
                                         query_u[:, None]).tanh()).squeeze(-1).float()
        alpha = scores_u.masked_fill(~available, -1e9).softmax(-1)*available
        alpha = alpha/alpha.sum(-1, keepdim=True).clamp_min(1e-12)
        evidence = (utterances*alpha[..., None]).sum(1)
        correction = self.output(torch.cat((evidence, prior_state), -1))
        return correction, alpha


class PriorEvidenceModel(nn.Module):
    def __init__(self, text_model, dim, classes, arm, scaffold, seed):
        super().__init__()
        encoder, self.mode = ARMS[arm]
        self.text = text_model
        if text_model is not None:
            text_model.candidate.scaffold = FixedScaffold(text_model.candidate.scaffold, scaffold)
        # Explicit RNG domains keep text/evidence initialization identical across encoders.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+7100)
            self.evidence = EvidenceSelection(dim, classes) if text_model is not None else None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+8100)
            self.prior = EmotionTrajectory(classes, encoder)

    def forward(self, own_emotions, own_lengths, local=None, batch=None, words=None):
        # Keep the small recurrent prior out of the text encoder's autocast path.
        with torch.autocast(device_type=own_emotions.device.type, enabled=False):
            prior, state = self.prior(own_emotions, own_lengths)
        if self.mode == 'prior_only':
            return dict(logits=prior, prior_logits=prior)
        text = self.text(local, batch, None, 'none')['logits']
        own_role = batch.speakers == batch.target[:, None]
        correction, alpha = self.evidence(words, state, own_role, self.mode)
        # Prior information enters final fusion in BOTH ordinary and conditioned arms.
        return dict(logits=prior+text+correction, prior_logits=prior,
                    text_logits=text, evidence_correction=correction, utterance_weights=alpha)
