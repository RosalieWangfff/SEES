"""GLP-style English utterance classifier, with padding-aware attention pooling."""
import torch
from torch import nn


class EnglishExtractor(nn.Module):
    def __init__(self,spec):
        super().__init__()
        from transformers import AutoModel
        self.model=AutoModel.from_pretrained(spec['asset'],local_files_only=True)
        self.model.config.output_hidden_states=False
        assert self.model.config.hidden_size==1024
        self.fc_dropout=nn.Dropout(.2)
        self.fc=nn.Linear(1024,spec['classes'])
        self.attention=nn.Sequential(nn.Linear(1024,512),nn.Tanh(),nn.Linear(512,1))
        for module in (self.fc,self.attention[0],self.attention[2]):
            nn.init.normal_(module.weight,mean=0.,std=self.model.config.initializer_range)
            nn.init.zeros_(module.bias)

    def feature(self,inputs):
        states=self.model(**inputs).last_hidden_state
        score=self.attention(states).squeeze(-1).float()
        score=score.masked_fill(~inputs['attention_mask'].bool(),float('-inf'))
        return (states*score.softmax(-1).to(states.dtype).unsqueeze(-1)).sum(1)

    def forward(self,inputs):
        return self.fc(self.fc_dropout(self.feature(inputs)))
