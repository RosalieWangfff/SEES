"""Fixed four-dataset handoff contract; never opens a test split."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('meld', 'dailydialog', 'emorynlp', 'cped')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    import time
    for attempt in range(20):
        try:
            tmp.replace(path)
            break
        except PermissionError:
            if attempt==19: raise
            time.sleep(.25)

def rows(dataset, split):
    assert dataset in DATASETS and split in ('train', 'val')
    with (ROOT/'data'/dataset/f'{split}.jsonl').open(encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def model_path(dataset):
    return ROOT.parent/'baselines/assets/roberta-large'

def metrics(y, logits, classes):
    from sklearn.metrics import f1_score, accuracy_score
    pred = logits.argmax(-1)
    return {**{a: float(100*f1_score(y, pred, labels=list(range(classes)), average=a, zero_division=0))
               for a in ('weighted', 'macro')}, 'accuracy': float(100*accuracy_score(y, pred))}

def feature_path(dataset):
    return ROOT.parent/('baselines/glp_validation/meld_v1/features.npz' if dataset=='meld' else 'payload/external/emorynlp_public_v1/features.npz' if dataset=='emorynlp' else 'payload/external/extension_dailydialog_v1/features.npz')
def initial_path(dataset):
    return ROOT.parent/('baselines/glp_validation/meld_v1/B_s42/best.pt' if dataset=='meld' else 'payload/external/emorynlp_public_v1/B_s42/best.pt' if dataset=='emorynlp' else 'no_daily_warmstart.pt')
