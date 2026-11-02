"""Validate the release candidate without data, PyTorch or GPU access."""
import ast
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=json.loads((ROOT/'provenance/source_manifest.json').read_text())
    for record in manifest:
        p=ROOT/record['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256'],str(p)
    files=list(ROOT.rglob('*.py'))
    for p in files:
        ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    with (ROOT/'results/no_evidence_control_runs.csv').open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    keys={(r['dataset'],r['arm'],int(r['seed'])) for r in rows}
    expected={(d,a,s) for d in ('meld','dailydialog','emorynlp')
              for a in ('G2__dep_typed','G2__no_evidence') for s in (1008,42,43)}
    assert len(rows)==18 and keys==expected,'Incomplete/duplicate matched controls'
    alignment=json.loads((ROOT/'provenance/manuscript_alignment.json').read_text(encoding='utf-8'))
    for table in alignment['tables']:
        p=ROOT/table['file']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==table['sha256'],str(p)
        with p.open(encoding='utf-8',newline='') as f:
            data=list(csv.reader(f))
        assert len(data)-1==table['rows'],str(p)
        assert all(len(row)==len(data[0]) for row in data),str(p)
    forbidden={'.pt','.pth','.safetensors','.npz','.pkl','.env'}
    unexpected=[str(p.relative_to(ROOT)) for p in ROOT.rglob('*')
                if p.is_file() and '.git' not in p.parts and p.suffix in forbidden]
    assert not unexpected,unexpected
    print(json.dumps(dict(source_hashes_verified=len(manifest),python_files_parsed=len(files),
        matched_test_runs=len(rows),manuscript_tables_verified=len(alignment['tables']),
        training_executed=False),indent=2))

if __name__=='__main__':main()
