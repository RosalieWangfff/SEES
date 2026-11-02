"""Chinese transfer of DialogueGLP: native BERT positions, same pre-LN CLS injection."""
import ast,json,hashlib,sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from transformers import AutoModel,AutoTokenizer
from formal_model.backbone import ROOT,REPO,load_module
ASSET="hfl/chinese-roberta-wwm-ext"
class ChineseLocal(nn.Module):
 def __init__(self):
  super().__init__();self.model=AutoModel.from_pretrained(ASSET,local_files_only=True);self.model.config.output_hidden_states=False
 def over_write_embedding_forward(self):
  # Native BERT handles positions/types/LN. Do not overwrite it with RoBERTa positions.
  pass
 def forward(self,additional_embedding=None,**kwargs):
  kwargs=dict(kwargs)
  if additional_embedding is None:return self.model(**kwargs)
  if "inputs_embeds" in kwargs:raise ValueError("Explicit token IDs required for audited state injection")
  ids=kwargs.pop("input_ids");emb=self.model.get_input_embeddings()(ids)
  assert additional_embedding.shape==(emb.shape[0],emb.shape[-1])
  extra=torch.zeros_like(emb);extra[:,0,:]=additional_embedding
  return self.model(inputs_embeds=emb+extra,**kwargs)

def create_chinese_backbone(classes=13,device="cpu"):
 di=load_module("_formal_chinese_di",REPO/"model/dialogue_infer.py")
 namespace={"torch":torch,"nn":nn,"EIRoberta":ChineseLocal,"DialogueInfer":di.DialogueInfer}
 source=REPO/"model/base_models.py";node=next(n for n in ast.parse(source.read_text(encoding="utf-8")).body if isinstance(n,ast.ClassDef) and n.name=="BaseModel")
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),"exec"),namespace)
 cfg=SimpleNamespace(input_size=768,lstm_hidden_size=768,hidden_size=768,fc_dropout=.2,target_size=classes)
 model=namespace["BaseModel"](cfg).to(device);model.dialogue_infer.device=torch.device(device)
 return model,dict(asset=ASSET,hidden_size=768,classes=classes,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),adaptation="native BERT word embeddings plus projected global state at CLS before type/position/LN",author_published_cped_model=False)

def verify():
 from formal_model.model import HistoryBatch,HistoryAwareGLP
 from formal_model.data import ATTRIBUTE_DIM
 torch.set_num_threads(2);torch.manual_seed(42);backbone,provenance=create_chinese_backbone();model=HistoryAwareGLP(backbone,768,13,ATTRIBUTE_DIM).eval()
 tok=AutoTokenizer.from_pretrained(ASSET,local_files_only=True);local=tok("我担心考试。 [SEP] 别担心，我们可以一起复习。",return_tensors="pt")
 with torch.no_grad():
  native=backbone.ei_roberta.model(**local).last_hidden_state
  zero=backbone.ei_roberta(additional_embedding=torch.zeros(1,768),**local).last_hidden_state
 assert torch.equal(native,zero),"Zero-injection changes native Chinese model output"
 b=HistoryBatch(utterances=torch.randn(1,2,768),emotions=torch.tensor([[8,10]]),speakers=torch.tensor([[0,1]]),target=torch.tensor([0]),valid=torch.ones(1,2,dtype=torch.bool),response=torch.tensor([[1]]),antecedent=torch.tensor([[0]]),wrong_antecedent=torch.tensor([[0]]),attributes=torch.zeros(1,1,ATTRIBUTE_DIM),confidence=torch.ones(1,1),links=torch.ones(1,1,dtype=torch.bool))
 output=model(local,b);assert output["logits"].shape==(1,13) and torch.isfinite(output["logits"]).all();assert torch.equal(output["logits"],output["base_logits"])
 output["logits"].square().mean().backward()
 for name in ["backbone.project.weight","gold.weight","relation.delta.weight"]:
  g=dict(model.named_parameters())[name].grad;assert g is not None and torch.isfinite(g).all() and g.abs().sum()>0,name
 report=dict(provenance=provenance,native_zero_injection_exact=True,real_pretrained_chinese_encoder_loaded=True,output_shape=[1,13],global_gold_and_relation_gradients=True,history_vectors="synthetic boundary/gradient check only",cped_training_or_efficacy=False)
 out=ROOT/"formal_model/data/cped/chinese_backbone_verification.json";out.write_text(json.dumps(report,indent=2),encoding="utf-8");print(json.dumps(report,indent=2))
if __name__=="__main__":verify()
