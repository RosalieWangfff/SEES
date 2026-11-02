"""Audit actual relation-label permutations; never train or alter source caches."""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

DATASETS = ('meld', 'dailydialog', 'emorynlp')
SEEDS = (1008, 42, 43)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def perm_stats(words, valid, annotate, punctuation, shuffle, seed):
    relations, _ = annotate(words)
    assert len(relations) == len(valid) == len(words)
    active = []
    for i, word in enumerate(words):
        head = int(word['head'])
        if head >= 0:
            assert head < len(words) and words[head]['sentence'] == word['sentence']
            if valid[i] and valid[head] and int(relations[i]) != punctuation:
                active.append(int(relations[i]))
    source = torch.tensor(active, dtype=torch.int64)
    graph = {'edge_relation': source, 'edge_child': torch.arange(len(active)),
             'word_to_utterance': torch.zeros(len(active), dtype=torch.int64)}
    permuted = shuffle(graph, seed)['edge_relation'].tolist()
    assert Counter(active) == Counter(permuted), 'Permutation changed label counts'
    eligible = len(set(active)) > 1
    return {'edges': len(active), 'changed': sum(x != y for x, y in zip(active, permuted)),
            'eligible': int(eligible), 'different': int(active != permuted),
            'no_edges': int(not active), 'single': int(len(active) == 1),
            'homogeneous': int(len(active) > 1 and not eligible)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--splits', nargs='+', choices=('train', 'val', 'test'), default=('test',))
    a = p.parse_args()
    root, out = a.root.resolve(), a.out.resolve()
    assert root.is_dir() and not out.exists(), 'Missing project or output folder already exists'
    sys.path[:0] = [str(root), str(root / 'glp')]
    from aligned.common import rows as load_train
    from recovery.test_features import rows as load_test
    from formal_model.english_structure_matrix.data import annotations, RELATIONS
    from nc_revision.model import shuffled_relations
    out.mkdir(parents=True)
    records, metadata, missing = [], [], []
    for ds in DATASETS:
        for split in a.splits:
            if split == 'test':
                data, _, _, _ = load_test(ds)
                instances = [(str(x['id']), x['text']) for x in data]
                cache = root / 'test_cache' / ds
            else:
                data = load_train(ds, split)
                instances = [(str(x['inputs']['id']), x['inputs']['text']) for x in data]
                cache = root / 'formal_model' / 'data' / (ds + '_dependency_v1')
            parse, complete = cache / 'parses.jsonl', cache / 'word_encoding_complete.json'
            if not parse.is_file() or not complete.is_file():
                missing.append({'dataset': ds, 'split': split, 'reason': 'Missing alignment cache'})
                continue
            done = json.loads(complete.read_text(encoding='utf-8-sig'))
            assert done['complete'] and sha(parse) == done['parse_sha'], 'Invalid parse cache'
            wanted = {t for _, ts in instances for t in ts}
            result = {}
            with parse.open(encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    text = item['text']
                    if text not in wanted:
                        continue
                    assert text not in result, 'Duplicate parse text'
                    key = hashlib.sha256(text.encode()).hexdigest()
                    with np.load(cache / 'word_features' / (key + '.npz'), allow_pickle=False) as z:
                        assert str(z['parse_sha']) == done['parse_sha'] and str(z['checkpoint_sha']) == done['checkpoint_sha']
                        result[text] = {seed: perm_stats(item['words'], z['valid'], annotations,
                            RELATIONS.index('punct'), shuffled_relations, seed) for seed in SEEDS}
            assert len(result) == len(wanted), f'{ds}/{split}: missing parsed utterances'
            assert len({i for i, _ in instances}) == len(instances), 'Duplicate instance IDs'
            for seed in SEEDS:
                unique = [v[seed] for v in result.values()]
                occurrences = [result[t][seed] for _, ts in instances for t in ts]
                total = sum(x['edges'] for x in occurrences)
                changed = sum(x['changed'] for x in occurrences)
                eligible = sum(x['eligible'] for x in occurrences)
                changed_utterances = sum(x['different'] for x in occurrences)
                changed_instances = sum(any(result[t][seed]['different'] for t in ts) for _, ts in instances)
                eligible_edges = sum(x['edges'] for x in occurrences if x['eligible'])
                records.append(dict(dataset=ds, split=split, seed=seed, instances=len(instances),
                    changed_instances=changed_instances, instance_changed_rate=changed_instances / len(instances) if instances else 0,
                    utterance_occurrences=len(occurrences), active_edges=total, changed_edges=changed,
                    edge_changed_rate=changed / total if total else 0,
                    eligible_utterance_occurrences=eligible, eligible_edges=eligible_edges,
                    changed_eligible_edge_rate=changed / eligible_edges if eligible_edges else 0,
                    changed_utterance_occurrences=changed_utterances,
                    unchanged_eligible_utterance_occurrences=eligible-changed_utterances,
                    no_edge_utterance_occurrences=sum(x['no_edges'] for x in occurrences),
                    single_edge_utterance_occurrences=sum(x['single'] for x in occurrences),
                    homogeneous_utterance_occurrences=sum(x['homogeneous'] for x in occurrences),
                    unique_utterances=len(unique), unique_active_edges=sum(x['edges'] for x in unique),
                    unique_changed_edges=sum(x['changed'] for x in unique)))
                print(f'{ds}/{split}/s{seed}: changed {changed}/{total} active edge occurrences; {changed_instances}/{len(instances)} histories', flush=True)
            metadata.append({'dataset': ds, 'split': split, 'parse_sha': done['parse_sha'],
                'checkpoint_sha': done['checkpoint_sha'], 'shuffle_source_sha': sha(root / 'nc_revision/model.py')})
    fields = ('dataset','split','seed','instances','changed_instances','instance_changed_rate',
        'utterance_occurrences','active_edges','changed_edges','edge_changed_rate',
        'eligible_utterance_occurrences','eligible_edges','changed_eligible_edge_rate',
        'changed_utterance_occurrences','unchanged_eligible_utterance_occurrences',
        'no_edge_utterance_occurrences','single_edge_utterance_occurrences',
        'homogeneous_utterance_occurrences','unique_utterances','unique_active_edges','unique_changed_edges')
    with (out / 'relation_shuffle_coverage.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(records)
    (out / 'metadata.json').write_text(json.dumps({'sources': metadata, 'missing': missing,
        'scope': 'active child-to-head edges before bidirectional expansion; occurrence counts repeat historical utterances across instances; actual seed-specific deterministic within-utterance permutation; unchanged eligible utterances include coincidental identity/repeated labels',
        'audit_script_sha': sha(Path(__file__))}, indent=2), encoding='utf-8')
    print('DONE:', out, 'missing:', len(missing), flush=True)


if __name__ == '__main__':
    main()
