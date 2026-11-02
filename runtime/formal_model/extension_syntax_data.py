"""Fail-closed formal extension UD/word caches and unchanged original graph batching.

Word-cache producer contract: word_encoding_complete.json contains complete,
dataset, dim, texts, parse_sha, checkpoint_sha, extractor_protocol_sha256,
manifest_sha256 and files_sha256 (text hash -> NPZ file hash). Each NPZ retains
the original values/valid/truncated/parse_sha/checkpoint_sha fields.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from formal_model.dependency_data import DependencyFeatures
from formal_model.extension_model import history_input
from formal_model.prepare_extension_syntax import ROLES, validate


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def align_contextual_words(record, offsets, token_values):
    """Mean contextual subwords overlapping each parser span; no text normalization.

Requires a right-truncated fast tokenizer's original-text character offsets.
Special tokens have zero-length offsets. Partial words remain invalid, as in
the old encoder; also reject a missing beginning (e.g. accidental left truncation).
"""
    validate(record)
    offsets, token_values = np.asarray(offsets), np.asarray(token_values)
    assert offsets.shape == (len(token_values), 2) and token_values.ndim == 2
    assert np.issubdtype(offsets.dtype, np.integer) and np.isfinite(token_values).all()
    assert ((offsets >= 0) & (offsets <= len(record['text']))).all()
    assert (offsets[:, 1] >= offsets[:, 0]).all()
    content = offsets[:, 1] > offsets[:, 0]
    values = np.zeros((len(record['words']), token_values.shape[1]), dtype=np.float16)
    valid = np.zeros(len(record['words']), dtype=bool)
    for i, word in enumerate(record['words']):
        overlap = content & (offsets[:, 0] < word['end']) & (offsets[:, 1] > word['start'])
        if overlap.any() and offsets[overlap, 0].min() <= word['start'] and offsets[overlap, 1].max() >= word['end']:
            values[i] = token_values[overlap].mean(0)
            valid[i] = True
    assert np.isfinite(values).all()
    truncated = bool(content.any() and offsets[content, 1].max() < len(record['text'].rstrip()))
    return values, valid, truncated


class ExtensionSyntaxFeatures(DependencyFeatures):
    def __init__(self, data, directory, extractor_directory):
        root, extractor = Path(directory), Path(extractor_directory)
        manifest_path = root/'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        complete = json.loads((root/'parse_complete.json').read_text(encoding='utf-8'))
        assert complete['complete'] is True and not complete['validation_only']
        assert not manifest['validation_only'] and manifest['dataset'] == data.dataset
        assert manifest['texts'] == data.texts and manifest['sha256'] == data.text_sha
        assert manifest['roles'] == ROLES
        assert {s: item['sha256'] for s,item in manifest['sources'].items()} == data.sources
        assert complete['sources'] == manifest['sources'] and complete['parser'] == manifest['parser']
        assert manifest['parser']['lang'] == data.spec['language'] and manifest['parser']['use_gpu'] is False
        assert complete['texts'] == len(data.texts) and complete['sha256'] == data.text_sha
        parse = root/'parses.jsonl'
        parse_sha = digest(parse)
        assert complete['parse_sha256'] == parse_sha
        with parse.open(encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream]
        assert [r['text'] for r in rows] == data.texts
        for record in rows:
            validate(record)
        protocol = json.loads((extractor/'protocol.json').read_text(encoding='utf-8'))
        extracted = json.loads((extractor/'extractor_complete.json').read_text(encoding='utf-8'))
        replay = json.loads((extractor/'checkpoint_replay.json').read_text(encoding='utf-8'))
        assert extracted['complete'] is True and not extracted['diagnostic_only'] and not protocol['probe']
        assert protocol['dataset'] == data.dataset and protocol['labels'] == data.labels
        assert protocol['prepared_source_sha256'] == data.sources
        assert extracted['protocol_sha256'] == replay['protocol_sha256'] == digest(extractor/'protocol.json')
        assert extracted['checkpoint_replay_verified'] and replay['passed']
        assert extracted['replay_sha256'] == digest(extractor/'checkpoint_replay.json')
        checkpoint = extractor/extracted['checkpoint_file']
        assert checkpoint.resolve().is_relative_to(extractor.resolve())
        checkpoint_sha = digest(checkpoint)
        assert extracted['checkpoint_sha256'] == replay['checkpoint_sha256'] == checkpoint_sha
        encoded = json.loads((root/'word_encoding_complete.json').read_text(encoding='utf-8'))
        assert encoded['complete'] is True and not encoded.get('diagnostic_only', False)
        assert encoded['dataset'] == data.dataset and encoded['dim'] == data.spec['dim']
        assert encoded['texts'] == len(data.texts) and encoded['parse_sha'] == parse_sha
        assert encoded['checkpoint_sha'] == checkpoint_sha
        assert encoded['extractor_protocol_sha256'] == extracted['protocol_sha256']
        assert encoded['manifest_sha256'] == digest(manifest_path)
        keys = {hashlib.sha256(t.encode()).hexdigest() for t in data.texts}
        assert set(encoded['files_sha256']) == keys
        for record in rows:
            key = hashlib.sha256(record['text'].encode()).hexdigest()
            path = root/'word_features'/(key+'.npz')
            assert digest(path) == encoded['files_sha256'][key]
            with np.load(path, allow_pickle=False) as z:
                assert z['values'].shape == (len(record['words']), data.spec['dim'])
                assert np.issubdtype(z['values'].dtype, np.floating) and np.isfinite(z['values']).all()
                assert z['valid'].shape == (len(record['words']),) and z['valid'].dtype == np.bool_
                assert str(z['parse_sha']) == parse_sha and str(z['checkpoint_sha']) == checkpoint_sha
                assert z['truncated'].shape == ()
        # Exactly preserve deterministic head/type corruption and full graph keys.
        super().__init__(root)
        self.dataset, self.dim = data.dataset, data.spec['dim']
        self.provenance = dict(parse_sha=parse_sha, checkpoint_sha=checkpoint_sha,
                               word_manifest_sha256=digest(root/'word_encoding_complete.json'))

    def collate(self, rows):
        for row in rows:
            history_input(row)
            assert all(text in self.records for text in row['text']), 'Missing historical word cache'
        assert rows, 'Empty graph batch'
        graph = super().collate(rows)
        assert graph['values'].shape[1] == self.dim
        assert ((graph['roles'] >= 0) & (graph['roles'] < 7)).all()
        return graph
