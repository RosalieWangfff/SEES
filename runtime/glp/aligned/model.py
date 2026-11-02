"""Upstream GLP class bodies + explicitly named target-history late-fusion arm."""
import ast
import importlib.util
from types import SimpleNamespace
import torch
from torch import nn
from transformers import AutoModel
from .common import ROOT

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

def make_glp(asset, classes, chinese=False):
    source = ROOT/'dev_only/model'
    di = module('handoff_dialogue_infer', source/'dialogue_infer.py')
    if chinese:
        class Local(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = AutoModel.from_pretrained(str(asset), local_files_only=True)
            def over_write_embedding_forward(self):
                pass
            def forward(self, additional_embedding=None, **kw):
                ids = kw.pop('input_ids')
                emb = self.model.get_input_embeddings()(ids)
                extra = torch.zeros_like(emb)
                extra[:, 0] = additional_embedding
                return self.model(inputs_embeds=emb+extra, **kw)
        dim = 768
    else:
        ei = module('handoff_ei_roberta', source/'ei_roberta.py')
        class LocalAuto:
            @staticmethod
            def from_pretrained(name, **kw):
                assert name == 'roberta-large'
                return AutoModel.from_pretrained(str(asset), local_files_only=True, **kw)
        ei.AutoModel = LocalAuto
        Local, dim = ei.EIRoberta, 1024
    scope = dict(torch=torch, nn=nn, EIRoberta=Local, DialogueInfer=di.DialogueInfer)
    tree = ast.parse((source/'base_models.py').read_text(encoding='utf-8'))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'BaseModel')
    exec(compile(ast.Module(body=[cls], type_ignores=[]), 'upstream_BaseModel', 'exec'), scope)
    cfg = SimpleNamespace(input_size=dim, lstm_hidden_size=dim, hidden_size=dim,
                          fc_dropout=.2, target_size=classes)
    return scope['BaseModel'](cfg)

class TargetHistory(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.embedding = nn.Embedding(classes, 32)
        self.encoder = nn.GRU(32, 32, batch_first=True)
        self.norm = nn.LayerNorm(32)
        self.head = nn.Linear(32, classes)
    def forward(self, labels, lengths):
        encoded = self.encoder(self.embedding(labels))[0]
        state = encoded[torch.arange(len(labels), device=labels.device), (lengths-1).clamp_min(0)]
        state = torch.where((lengths>0)[:, None], state, torch.zeros_like(state))
        return self.head(self.norm(state))

class MatchedGLP(nn.Module):
    def __init__(self, base, classes, history, seed=1008):
        super().__init__()
        self.base = base
        self.history = history
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+8100)
            self.prior = TargetHistory(classes) if history else None
    def forward(self, inputs, labels, lengths):
        output = self.base(inputs).float()
        if self.prior is not None:
            with torch.autocast(device_type=labels.device.type, enabled=False):
                output = output+self.prior(labels, lengths)
        return output
