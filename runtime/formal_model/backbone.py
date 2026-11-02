"""Load the actual upstream DialogueGLP classes without unrelated graph dependencies."""
import ast,importlib.util,hashlib
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from transformers import AutoModel
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/"baselines/DialogueGLP"
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def create_backbone(classes=7,asset=None,device="cpu"):
    asset=Path(asset or ROOT/"baselines/assets/roberta-large")
    if not (asset/"config.json").exists():raise FileNotFoundError(asset)
    ei=load_module("_formal_glp_ei",REPO/"model/ei_roberta.py")
    di=load_module("_formal_glp_di",REPO/"model/dialogue_infer.py")
    class LocalAuto:
        @staticmethod
        def from_pretrained(name,*args,**kwargs):
            if name!="roberta-large":raise ValueError(name)
            return AutoModel.from_pretrained(str(asset),*args,local_files_only=True,**kwargs)
    ei.AutoModel=LocalAuto
    namespace={"torch":torch,"nn":nn,"EIRoberta":ei.EIRoberta,"DialogueInfer":di.DialogueInfer}
    source=REPO/"model/base_models.py"
    node=next(n for n in ast.parse(source.read_text(encoding="utf-8")).body if isinstance(n,ast.ClassDef) and n.name=="BaseModel")
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),"exec"),namespace)
    cfg=SimpleNamespace(input_size=1024,lstm_hidden_size=1024,hidden_size=1024,fc_dropout=.2,target_size=classes)
    model=namespace["BaseModel"](cfg).to(device);model.dialogue_infer.device=torch.device(device)
    provenance={"architecture":"upstream BaseModel + EIRoberta + DialogueInfer class bodies","asset":str(asset),"classes":classes,"source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,REPO/"model/ei_roberta.py",REPO/"model/dialogue_infer.py"]}}
    return model,provenance
