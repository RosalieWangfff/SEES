"""Project paths and existing cache contracts for the fixed core experiments."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from formal_model.train_extension_extractor import digest

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT/'payload/external'


class CoreData:
    def __init__(self, dataset, need_text):
        self.dataset = dataset
        self.dim = 768 if dataset == 'cped' else 1024
        self.epochs = 5 if dataset in ('meld', 'emorynlp') else 16
        self.freeze = 0 if dataset in ('meld', 'emorynlp') else 2
        self.initial = None
        self.hashes = {}
        self.asset = ROOT/'baselines/assets/roberta-large'
        if dataset in ('meld', 'emorynlp'):
            from formal_model.public_data import load, LABELS
            self.labels = LABELS[dataset]
            self.rows = {}
            for split in ('train', 'val'):
                records, source = load(dataset, split)
                self.rows[split] = [dict(r['inputs'], y=r['label']) for r in records]
                self.hashes[split] = digest(source)
            self.extractor = ROOT/'baselines/glp_validation/meld_v1' if dataset == 'meld' else EXPERIMENTS/'emorynlp_public_v1'
            self.initial = self.extractor/'B_s42/best.pt'
            if need_text:
                import hashlib
                texts = sorted({t for rows in self.rows.values() for r in rows for t in r['text']})
                key = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
                with np.load(self.extractor/'features.npz', allow_pickle=False) as z:
                    assert str(z['sha']) == key
                    self.vectors = z['values'].copy()
                self.index = {t:i for i,t in enumerate(texts)}
        else:
            from formal_model.extension_data import ExtensionData
            self.extension = ExtensionData(dataset)
            self.rows, self.labels = self.extension.rows, self.extension.labels
            self.hashes.update(self.extension.sources)
            self.extractor = EXPERIMENTS/f'extension_{dataset}_v1'
            if need_text:
                self.vectors = self.extension.load_sentence_features(self.extractor)
                self.index = self.extension.text_index
            if need_text and dataset == 'cped':
                from huggingface_hub import snapshot_download
                self.asset = Path(snapshot_download('hfl/chinese-roberta-wwm-ext', local_files_only=True))
        self.classes = len(self.labels)
        if need_text:
            assert self.vectors.shape == (len(self.index), self.dim) and np.isfinite(self.vectors).all()
            from formal_model.dependency_data import DependencyFeatures
            from formal_model.extension_syntax_data import ExtensionSyntaxFeatures
            directory = ROOT/f'formal_model/data/{dataset}_dependency_v1'
            self.word_features = (DependencyFeatures(directory) if dataset in ('meld', 'emorynlp') else
                                  ExtensionSyntaxFeatures(self.extension, directory, self.extractor))
            self.hashes['features'] = digest(self.extractor/'features.npz')
            self.hashes['word_manifest'] = digest(directory/'word_encoding_complete.json')
            self.hashes['parses'] = digest(directory/'parses.jsonl')
            if self.initial is not None:
                self.hashes['initial_checkpoint'] = digest(self.initial)

    def own_history(self, split, ids, device):
        histories = []
        for i in ids:
            row = self.rows[split][i]
            histories.append([e for e,s in zip(row['emotions'], row['speakers']) if s == row['target']])
        width = max(1, max(map(len, histories)))
        tokens = torch.zeros(len(ids), width, dtype=torch.long)
        lengths = torch.tensor([len(h) for h in histories], dtype=torch.long)
        for i, values in enumerate(histories):
            tokens[i, :len(values)] = torch.tensor(values, dtype=torch.long)
        return tokens.to(device), lengths.to(device)

    def words(self, split, ids, device):
        history = [{k:self.rows[split][i][k] for k in ('id', 'text', 'speakers', 'target')} for i in ids]
        original = self.word_features.collate(history)
        # Use all cached valid words, NOT dependency types, edges, or selected syntax categories.
        keys = ('values', 'valid', 'word_to_utterance', 'lookup', 'utterances')
        return {k:original[k].to(device) if torch.is_tensor(original[k]) else original[k] for k in keys}
