"""MELD public-protocol data adapter. Future utterance text is never stored in inputs."""
import sys,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
from formal_model.model import HistoryBatch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"ERC-revision"))
sys.path.insert(0,str(ROOT/"baselines"))
from extract_reply_kimi_v2 import FUNCTIONS,SEMANTICS
from align_glp_kimi_cohort import repair
LABELS=["anger","disgust","fear","joy","neutral","sadness","surprise"]
ATTRIBUTE_DIM=len(FUNCTIONS)+len(SEMANTICS)+5
BASE=ROOT/"baselines/glp_validation/meld_v1"

def load_rows(split):
    if split not in ("train","val"):raise ValueError("Training adapter does not load test")
    df=pd.read_csv(ROOT/"baselines/DialogueGLP/data/meld/processed"/(split+"_full.csv"))
    rows=[]
    for key,g in df.groupby("Dialogue_ID",sort=True):
        values=g.sort_values("Utterance_ID").to_dict("records")
        if len(values)<2:continue
        history=values[:-1];target=values[-1]["Speaker"]
        speakers=list(dict.fromkeys([h["Speaker"] for h in history]+[target]))
        rows.append({"id":str(key),"text":[str(h["Utterance"]) for h in history],"speakers":[speakers.index(h["Speaker"]) for h in history],"target":speakers.index(target),"emotions":[LABELS.index(h["Emotion"]) for h in history],"y":LABELS.index(values[-1]["Emotion"])})
    return rows

class MELDAdapter:
    def __init__(self):
        self.rows={s:load_rows(s) for s in ("train","val")}
        texts=sorted({t for rs in self.rows.values() for r in rs for t in r["text"]})
        z=np.load(BASE/"features.npz");sha=hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest()
        assert str(z["sha"])==sha
        self.vectors=z["values"];self.text_index={t:i for i,t in enumerate(texts)}
        self.mapping=json.loads((BASE/"kimi_sample_alignment.json").read_text(encoding="utf-8"))
        path=ROOT/"ERC-revision/kimi_relation/v2/cohort4800/annotations.jsonl"
        self.records={r["id"]:r for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines())}
    def make(self,r,split,device="cpu"):
        n=len(r["text"]);m=self.mapping[split].get(r["id"]);record=self.records.get(m["kimi_id"]) if m else None
        response=np.full(3,n-1,np.int64);obj=np.zeros(3,np.int64);wrong=obj.copy();attrs=np.zeros((3,ATTRIBUTE_DIM),np.float32);confidence=np.zeros(3,np.float32);valid=np.zeros(3,bool)
        if record and record.get("parsed") is not None:
            start=m["history_window_start"]
            assert [repair(t[:512]) for t in r["text"][start:]]==[repair(h["text"]) for h in record["history"]]
            for j,link in enumerate(record["parsed"]["links"][:3]):
                source=link["antecedent_turn"]
                if source is None:continue
                index=start+source;assert 0<=index<n-1
                obj[j]=index;wrong[j]=index;valid[j]=True
                candidates=[i for i in range(n-1) if i!=index and r["speakers"][i]==r["speakers"][index]]
                if candidates:wrong[j]=candidates[int(hashlib.sha256((r["id"]+":"+str(j)).encode()).hexdigest(),16)%len(candidates)]
                for name in link["functions"]:attrs[j,FUNCTIONS.index(name)]=1
                attrs[j,len(FUNCTIONS)+SEMANTICS.index(link["semantic"])]=1
                c=["low","medium","high"].index(link["confidence"]);attrs[j,len(FUNCTIONS)+len(SEMANTICS)+c]=1;confidence[j]=[.25,.6,1.][c]
                attrs[j,-2]=r["speakers"][index]==r["target"];attrs[j,-1]=r["speakers"][-1]==r["target"]
        tensor=lambda x:torch.as_tensor(x,device=device).unsqueeze(0)
        b=HistoryBatch(utterances=tensor(np.stack([self.vectors[self.text_index[t]] for t in r["text"]])).float(),emotions=tensor(r["emotions"]).long(),speakers=tensor(r["speakers"]).long(),target=torch.tensor([r["target"]],device=device),valid=torch.ones(1,n,dtype=torch.bool,device=device),response=tensor(response),antecedent=tensor(obj),wrong_antecedent=tensor(wrong),attributes=tensor(attrs),confidence=tensor(confidence),links=tensor(valid))
        b.validate(len(LABELS));return b
    @staticmethod
    def local_inputs(r,tokenizer,device="cpu"):
        parts=[("I: " if s==r["target"] else "Other: ")+t for s,t in zip(r["speakers"][-2:],r["text"][-2:])]
        return tokenizer((" "+tokenizer.sep_token+" ").join(parts),max_length=512,truncation=True,return_tensors="pt").to(device)
