"""Utterance-supervision input for extension extractors, never expanded prefixes."""
import hashlib
import json
from pathlib import Path
from formal_model.extension_data import ExtensionData, ROOT


def utterances(dataset, split):
    if split not in ('train', 'val'):
        raise ValueError('Extractor development excludes test')
    data = ExtensionData(dataset)
    if dataset == 'cped':
        source = Path('D:/multi-Model ERG/data_w2/cped')
        path = source / ('train.json' if split == 'train' else 'valid.json')
        dialogs = json.loads(path.read_text(encoding='utf-8'))
        valid = json.loads((source / 'valid.json').read_text(encoding='utf-8'))
        fingerprint = lambda d: json.dumps([d['utterances'], d['speakers']], ensure_ascii=False)
        excluded = {fingerprint(d) for d in valid} if split == 'train' else set()
        records = []
        for d in dialogs:
            if fingerprint(d) in excluded:
                continue
            assert len(d['utterances']) == len(d['emotions']) == len(d['speakers'])
            for index, (text, emotion) in enumerate(zip(d['utterances'], d['emotions'])):
                records.append((str(d['dialogue_id']), index, text, data.labels.index(emotion)))
    else:
        import pandas as pd
        path = ROOT / 'baselines/DialogueGLP/data/daily_dialogue' / (split + '.csv')
        frame = pd.read_csv(path).sort_values(['Dialogue_ID', 'Utterance_ID'])
        records = [(str(r.Dialogue_ID), int(r.Utterance_ID), str(r.Utterance), data.labels.index(r.Emotion))
                   for r in frame.itertuples()]
    assert records and len({(d, i) for d, i, _, _ in records}) == len(records)
    return records, dict(source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         sampling='one record per raw utterance occurrence; no prefix expansion')


if __name__ == '__main__':
    from collections import Counter
    report = {'training_started': False, 'test_loaded': False, 'splits': {}}
    for dataset in ('cped', 'dailydialog'):
        for split in ('train', 'val'):
            records, source = utterances(dataset, split)
            report['splits'][dataset + '/' + split] = dict(utterances=len(records),
                dialogues=len({d for d, _, _, _ in records}), counts=dict(Counter(y for _, _, _, y in records)),
                **source)
    (ROOT / 'formal_model/extension_extractor_input_audit.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
