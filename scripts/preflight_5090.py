"""Read-only 5090 environment and project preflight; never starts training."""
import argparse, json, os, shutil, sys
from pathlib import Path
import torch
from prepare import generate

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    required = ['recovery/train.py', 'glp/aligned/train.py', 'formal_model/core_prior/data.py',
                'baselines/assets/roberta-large/config.json', 'payload']
    missing = [x for x in required if not (root/x).exists()]
    result = {'python':sys.executable, 'platform':sys.platform, 'root':str(root),
              'missing_project_items':missing, 'torch':str(torch.__version__),
              'cuda_build':torch.version.cuda, 'cuda_available':torch.cuda.is_available(),
              'training_started':False}
    if root.exists(): result['free_disk_gib'] = round(shutil.disk_usage(root).free/2**30, 2)
    if torch.cuda.is_available():
        result['gpus'] = []
        for index in range(torch.cuda.device_count()):
            device = torch.cuda.get_device_properties(index)
            free, total = torch.cuda.mem_get_info(index)
            result['gpus'].append({'index':index, 'name':device.name,
                'compute_capability':f'{device.major}.{device.minor}',
                'total_gib':round(total/2**30,2),'currently_free_gib':round(free/2**30,2)})
        result['pytorch_arch_list']=torch.cuda.get_arch_list()
    if not missing:
        try:
            sources=generate(root)
            result['source_compatible']=True
            present = [n for n in sources if (root/'nc_revision'/n).exists()]
            result['installed_revision_files']=present
            result['installed_revision_matches']=all((root/'nc_revision'/n).read_text(encoding='utf8') == sources[n] for n in present)
        except Exception as e:
            result['source_compatible']=False;result['source_error']=str(e)
    print(json.dumps(result,indent=2,ensure_ascii=False))
    if missing or not result.get('source_compatible') or not result['cuda_available']:
        raise SystemExit('Preflight failed; do not launch training.')
    if os.name != 'nt': raise SystemExit('Current evaluator and path layout require Windows; Linux needs a separate tested port.')
    print('Preflight only. Review GPU architecture and disk capacity before a probe.')

if __name__=='__main__': main()
