"""CPU-only historical UD parsing for CPED and DailyDialog; execution is opt-in."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ROLES = ['subject', 'predicate_or_head', 'object', 'nominal_modifier',
         'adverbial', 'complement', 'other']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def role(dep):
    d = dep.split(':')[0]
    for index, names in enumerate((('nsubj', 'csubj'), ('root', 'aux', 'cop'),
            ('obj', 'iobj'), ('amod', 'nmod', 'acl', 'appos', 'det', 'clf', 'nummod', 'compound'),
            ('advmod', 'obl', 'advcl'), ('ccomp', 'xcomp'))):
        if d in names:
            return index
    return 6


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.1 * (attempt + 1))


def validate(record):
    text, words = record['text'], record['words']
    for i, word in enumerate(words):
        assert 0 <= word['start'] < word['end'] <= len(text), 'Invalid character offsets'
        assert word['role'] == role(word['dependency']), 'UD role mismatch'
        head = word['head']
        assert head == -1 or (0 <= head < len(words) and head != i and
                            words[head]['sentence'] == word['sentence']), 'Invalid sentence-local head'


def record_from_doc(text, doc):
    words = []
    for sid, sentence in enumerate(doc.sentences):
        offset = len(words)
        assert [w.id for w in sentence.words] == list(range(1, len(sentence.words) + 1))
        tokens = {w.id: token for token in sentence.tokens for w in token.words}
        for w in sentence.words:
            token = tokens[w.id]
            start = w.start_char if w.start_char is not None else token.start_char
            end = w.end_char if w.end_char is not None else token.end_char
            words.append(dict(sentence=sid, local_id=w.id,
                head=-1 if w.head == 0 else offset + w.head - 1,
                dependency=w.deprel, role=role(w.deprel), text=w.text,
                lemma=w.lemma, start=start, end=end, upos=w.upos, features=w.feats))
    record = dict(text=text, words=words)
    validate(record)
    return record


def resume_records(path, texts):
    """Recover only an incomplete final write; any interior corruption fails closed."""
    records = []
    if not path.exists():
        return records
    with path.open('r+b') as stream:
        while True:
            position = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                stream.truncate(position)
                break
            record = json.loads(line)
            assert len(records) < len(texts) and record['text'] == texts[len(records)], 'Resume order mismatch'
            validate(record)
            records.append(record)
    return records


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--dataset', required=True, choices=('cped', 'dailydialog'))
    cli.add_argument('--out', type=Path)
    cli.add_argument('--resources', type=Path, default=ROOT / 'data/stanza_resources')
    cli.add_argument('--validation-samples', type=int, default=0,
                     help='Small real-data validation only; requires explicit --out; no formal completion marker')
    cli.add_argument('--execute', action='store_true')
    args = cli.parse_args()
    assert 0 <= args.validation_samples <= 32
    formal = ROOT / f'formal_model/data/{args.dataset}_dependency_v1'
    if args.validation_samples:
        assert args.out is not None and args.out.resolve() != formal.resolve(), 'Use an isolated validation directory'
    out = args.out or formal
    source = ROOT / 'formal_model/data' / args.dataset
    texts = set()
    sources = {}
    for split in ('train', 'val'):
        path = source / (split + '.jsonl')
        sources[split] = dict(path=str(path), sha256=digest(path))
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)['inputs']
                assert len(row['text']) == len(row['speakers']) == len(row['emotions']) > 0
                assert all(isinstance(t, str) for t in row['text'])
                texts.update(row['text'])
    texts = sorted(texts)
    full_count = len(texts)
    if args.validation_samples:
        texts = [t for t in texts if len(t.strip()) >= 8][:args.validation_samples]
        assert len(texts) == args.validation_samples
    lang = 'en' if args.dataset == 'dailydialog' else 'zh-hans'
    package = 'ewt' if lang == 'en' else 'gsdsimp'
    processors = dict(tokenize=package, pos=package + '_nocharlm',
                      lemma=package + '_nocharlm', depparse=package + '_nocharlm')
    if lang == 'en':
        processors['mwt'] = package
    model_paths = {name: args.resources / lang / name / (value + '.pt') for name, value in processors.items()}
    model_paths['pretrain'] = args.resources / lang / 'pretrain' / ('conll17.pt' if lang == 'en' else 'fasttext157.pt')
    model_paths['resources'] = args.resources / 'resources.json'
    plan = dict(dataset=args.dataset, execute=args.execute, output=str(out), device='cpu',
        texts=len(texts), full_history_texts=full_count, validation_only=bool(args.validation_samples),
        lang=lang, processors=processors, local_files={k: dict(path=str(p), exists=p.is_file()) for k,p in model_paths.items()})
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return
    assert all(p.is_file() for p in model_paths.values()), 'Local parser assets missing; no downloads allowed'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    sys.path.insert(0, str(ROOT / 'research_runtime'))
    import torch
    import stanza
    torch.set_num_threads(2)
    parser = dict(lang=lang, processors=processors, package=None, use_gpu=False,
                  download_method=None, resource_dir=str(args.resources.resolve()),
                  assets_sha256={k: digest(p) for k,p in model_paths.items()}, stanza_version=stanza.__version__)
    manifest = dict(dataset=args.dataset, texts=texts,
        sha256=hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest(),
        sources=sources, parser=parser, source_sha256=digest(__file__), roles=ROLES,
        scope='Historical input texts only; no target texts or labels',
        mapping='UD syntactic approximations, not universal one-to-one Chinese grammatical categories',
        validation_only=bool(args.validation_samples), full_history_texts=full_count)
    out.mkdir(parents=True, exist_ok=True)
    import msvcrt
    lock = (out / 'parse.lock').open('a+b')
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    if (out / 'manifest.json').exists():
        assert json.loads((out / 'manifest.json').read_text(encoding='utf-8')) == manifest, 'Changed input/parser; use a new output root'
    save(out / 'manifest.json', manifest)
    path = out / 'parses.jsonl'
    records = resume_records(path, texts)
    if len(records) < len(texts):
        nlp = stanza.Pipeline(lang=lang, dir=str(args.resources), package=None,
            processors=processors, download_method=None, use_gpu=False, verbose=False)
        for start in range(len(records), len(texts), 8):
            batch = texts[start:start + 8]
            docs = nlp.bulk_process(batch)
            assert len(docs) == len(batch)
            with path.open('a', encoding='utf-8', newline='\n') as stream:
                for text, doc in zip(batch, docs):
                    record = record_from_doc(text, doc)
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                    stream.flush()
                    records.append(record)
            save(out / 'status.json', dict(pid=os.getpid(), done=len(records), total=len(texts), device='cpu'))
    assert len(records) == len(texts)
    result = dict(complete=not bool(args.validation_samples), validation_passed=bool(args.validation_samples),
        validation_only=bool(args.validation_samples), texts=len(records), words=sum(len(r['words']) for r in records),
        sha256=manifest['sha256'], parse_sha256=digest(path), sources=sources, parser=parser,
        sentence_local_heads_resolved=True, all_offsets_and_head_bounds_checked=True,
        neural_features_ready=False, efficacy_verified=False, device='cpu')
    save(out / ('validation_result.json' if args.validation_samples else 'parse_complete.json'), result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources', 'parser')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
