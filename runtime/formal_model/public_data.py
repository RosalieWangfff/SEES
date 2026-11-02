"""Dataset-specific public prefix contracts; historical inputs and future labels separated."""
from pathlib import Path
import json,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
LABELS={"meld":["anger","disgust","fear","joy","neutral","sadness","surprise"],"emorynlp":["Neutral","Joyful","Peaceful","Powerful","Scared","Mad","Sad"],"dailydialog":["no emotion","anger","disgust","fear","happiness","sadness","surprise"]}
def load(dataset,split):
 if dataset not in LABELS:raise ValueError("No audited public adapter for "+dataset)
 if split not in ("train","val"):raise ValueError("Development adapter excludes test")
 if dataset=="dailydialog":
  path=ROOT/"baselines/DialogueGLP/data/daily_dialogue"/(split+".csv")
 else:path=ROOT/"baselines/DialogueGLP/data"/dataset/"processed"/(split+"_full.csv")
 df=pd.read_csv(path)
 if dataset=="dailydialog":
  df=df.sort_values(["Dialogue_ID","Utterance_ID"]);df["Speaker"]=df.groupby("Dialogue_ID",sort=False).cumcount()%2
 labels=LABELS[dataset];result=[]
 for key,g in df.groupby("Dialogue_ID",sort=True):
  values=g.sort_values("Utterance_ID").to_dict("records")
  if len(values)<2:continue
  history=values[:-1];target=values[-1]["Speaker"];speakers=list(dict.fromkeys([v["Speaker"] for v in history]+[target]))
  inputs=dict(id=str(key),text=[str(v["Utterance"]) for v in history],speakers=[speakers.index(v["Speaker"]) for v in history],target=speakers.index(target),emotions=[labels.index(v["Emotion"]) for v in history])
  # Target identity is assumed known by this forecasting task; target text never enters inputs.
  result.append(dict(inputs=inputs,label=labels.index(values[-1]["Emotion"])))
 return result,path

def prepare(dataset):
 out=ROOT/"formal_model/data"/dataset;out.mkdir(parents=True,exist_ok=True);report={"dataset":dataset,"labels":LABELS[dataset],"target_identity_known":True,"future_text_in_inputs":False,"test_loaded":False,"splits":{}}
 for split in ("train","val"):
  rows,path=load(dataset,split);counts=[0]*len(LABELS[dataset]);no_history=0
  with (out/(split+".jsonl")).open("w",encoding="utf-8") as f:
   for r in rows:
    x=r["inputs"];assert len(x["text"])==len(x["speakers"])==len(x["emotions"])>0
    assert "label" not in x and "y" not in x
    counts[r["label"]]+=1;no_history+=int(x["target"] not in x["speakers"])
    f.write(json.dumps(r,ensure_ascii=False)+"\n")
  report["splits"][split]=dict(rows=len(rows),label_counts=counts,no_target_history=no_history,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
 (out/"manifest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
 return report
if __name__=="__main__":
 from formal_model.data import load_rows
 for split in ["train","val"]:
  new,_=load("meld",split);old=load_rows(split)
  assert len(new)==len(old)
  for a,b in zip(new,old):assert dict(a["inputs"],y=a["label"])==b
 print(json.dumps(prepare("emorynlp"),indent=2))
