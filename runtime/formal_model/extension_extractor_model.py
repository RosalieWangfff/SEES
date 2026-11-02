"""Attention-pooled extension extractor; padding excluded in both languages."""
import torch
from torch import nn


def masked_pool(states, scores, mask):
    assert states.shape[:2] == scores.shape == mask.shape
    assert mask.bool().any(dim=1).all()
    weights = scores.float().masked_fill(~mask.bool(), float('-inf')).softmax(dim=1)
    return (states * weights.unsqueeze(-1).to(states.dtype)).sum(dim=1)


class ExtensionExtractor(nn.Module):
    def __init__(self, spec):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(spec['asset'], local_files_only=True)
        self.model.config.output_hidden_states = False
        assert self.model.config.hidden_size == spec['dim']
        self.attention = nn.Sequential(nn.Linear(spec['dim'], 512), nn.Tanh(), nn.Linear(512, 1))
        self.dropout = nn.Dropout(.1)
        self.fc = nn.Linear(spec['dim'], spec['classes'])
        nn.init.normal_(self.fc.weight, std=self.model.config.initializer_range)
        nn.init.zeros_(self.fc.bias)

    def feature(self, inputs):
        states = self.model(**inputs).last_hidden_state
        return masked_pool(states, self.attention(states).squeeze(-1), inputs['attention_mask'])

    def forward(self, inputs):
        return self.fc(self.dropout(self.feature(inputs)))


if __name__ == '__main__':
    torch.manual_seed(1008)
    states = torch.randn(2, 5, 8, requires_grad=True)
    scores = torch.randn(2, 5, requires_grad=True)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]])
    pooled = masked_pool(states, scores, mask)
    single = masked_pool(states[:1, :2], scores[:1, :2], mask[:1, :2])
    assert torch.allclose(pooled[:1], single)
    pooled.sum().backward()
    assert states.grad[0, 2:].count_nonzero() == 0
    assert scores.grad[0, 2:].count_nonzero() == 0
    print('PASS: padding invariance and zero padded-token gradients; synthetic pooling check only')
