"""Prepared CPED/DailyDialog forecasting inputs; no Kimi or feature prerequisites."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    'cped': dict(dim=768, classes=13, language='zh-hans',
                 asset='hfl/chinese-roberta-wwm-ext', unit='every_noninitial_turn'),
    'dailydialog': dict(dim=1024, classes=7, language='en',
                      asset=str(ROOT / 'baselines/assets/roberta-large'), unit='dialogue_last_turn'),
}


class ExtensionData:
    def __init__(self, dataset):
        self.spec = dict(SPECS[dataset])
        self.dataset = dataset
        directory = ROOT / 'formal_model/data' / dataset
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        self.labels = manifest['labels']
        assert len(self.labels) == self.spec['classes']
        self.rows, self.clusters, self.sources = {}, {}, {}
        for split in ('train', 'val'):
            path = directory / (split + '.jsonl')
            self.sources[split] = hashlib.sha256(path.read_bytes()).hexdigest()
            with path.open(encoding='utf-8') as stream:
                original = [json.loads(line) for line in stream]
            self.rows[split] = [dict(r['inputs'], y=r['label']) for r in original]
            self.clusters[split] = [str(r['dialogue']) if dataset == 'cped' else str(r['inputs']['id'])
                                    for r in original]
        self.texts = sorted({t for rows in self.rows.values() for r in rows for t in r['text']})
        self.text_sha = hashlib.sha256(json.dumps(self.texts, ensure_ascii=False).encode()).hexdigest()

    def text_inputs(self, split):
        """Exclude gold histories at the text-model boundary; targets remain separate."""
        for row, cluster in zip(self.rows[split], self.clusters[split]):
            yield dict(id=row['id'], text=row['text'], speakers=row['speakers'],
                       target=row['target']), row['y'], cluster

    def load_sentence_features(self, directory):
        """Fail closed on incomplete, wrong-source or wrong-dimensional caches."""
        import numpy as np
        directory = Path(directory)
        done = json.loads((directory / 'extractor_complete.json').read_text(encoding='utf-8'))
        protocol = json.loads((directory / 'protocol.json').read_text(encoding='utf-8'))
        assert done['complete'] is True
        assert done.get('diagnostic_only') is False
        assert done.get('checkpoint_replay_verified') is True
        assert hashlib.sha256((directory / 'protocol.json').read_bytes()).hexdigest() == done['protocol_sha256']
        replay_path = directory / 'checkpoint_replay.json'
        assert hashlib.sha256(replay_path.read_bytes()).hexdigest() == done['replay_sha256']
        replay = json.loads(replay_path.read_text(encoding='utf-8'))
        assert replay['passed'] is True
        assert replay['protocol_sha256'] == done['protocol_sha256']
        assert replay['checkpoint_sha256'] == done['checkpoint_sha256']
        assert hashlib.sha256((directory / 'features.npz').read_bytes()).hexdigest() == done['cache_sha256']
        assert protocol['dataset'] == self.dataset and protocol['labels'] == self.labels
        assert protocol['prepared_source_sha256'] == self.sources
        checkpoint = directory / done['checkpoint_file']
        assert checkpoint.resolve().is_relative_to(directory.resolve())
        digest = hashlib.sha256()
        with checkpoint.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        assert digest.hexdigest() == done['checkpoint_sha256']
        with np.load(directory / 'features.npz', allow_pickle=False) as cache:
            assert str(cache['sha']) == self.text_sha
            assert str(cache['checkpoint_sha256']) == done['checkpoint_sha256']
            values = cache['values'].copy()
        assert values.shape == (len(self.texts), self.spec['dim'])
        assert np.issubdtype(values.dtype, np.floating) and np.isfinite(values).all()
        self.vectors = values
        self.text_index = {text: i for i, text in enumerate(self.texts)}
        return values


if __name__ == '__main__':
    for dataset in SPECS:
        data = ExtensionData(dataset)
        for split in ('train', 'val'):
            assert all('emotions' not in x and 'y' not in x for x, y, cluster in data.text_inputs(split))
        print(json.dumps(dict(dataset=dataset, spec=data.spec, texts=len(data.texts),
            text_sha=data.text_sha, rows={s: len(r) for s, r in data.rows.items()},
            clusters={s: len(set(c)) for s, c in data.clusters.items()}, test_loaded=False), ensure_ascii=False))
