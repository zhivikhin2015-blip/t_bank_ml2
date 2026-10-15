"""Сохранить исходный эксперимент перед первой подготовкой v2."""
import shutil
from datetime import datetime, timezone

from .core import ORIGINAL, OUTPUT, PROJECT, original_integrity, sha, write_json


def snapshot_original():
    directory=OUTPUT/'original_snapshot'
    manifest=directory/'manifest.json'
    if manifest.exists():
        return original_integrity()
    required=['listens.parquet','data_manifest.json','train.npz','reward_val.npz',
              'policy_val.npz','catalog.npz']
    for name in required:
        if not (ORIGINAL/name).exists():
            raise ValueError('Сначала выполните исходные download.py и prepare.py: '+name)
    for seed in [11,22,33]:
        if not (ORIGINAL/f'baseline_s{seed}'/'best.pt').exists():
            raise ValueError(f'Нужен исходный baseline_s{seed}, 6000 шагов; команды приведены в документации')
    directory.mkdir(parents=True,exist_ok=True)
    files=[PROJECT/'README.md',PROJECT/'protocol.json',PROJECT/'requirements.txt',PROJECT/'.gitignore',
           *PROJECT.glob('*.py')]
    for p in files:
        shutil.copy2(p,directory/p.name)
    for p in ORIGINAL.iterdir():
        if p.is_file() and p.suffix in ['.pt','.npz','.parquet','.json','.jsonl','.png','.zip']:
            files.append(p)
        elif p.is_dir() and p.name.startswith(('baseline_','reward_','rl_','sft_','rerank_','popularity')):
            files.extend(f for f in p.rglob('*') if f.is_file() and f.suffix in ['.pt','.npz','.json','.jsonl','.png'])
    value={'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'status':'original_frozen_read_only_by_new_pipeline',
            'files':{str(p.relative_to(PROJECT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in files},
            'old_test_status':'already_accessed in development history; diagnostic only for revised procedures',
            'preservation':'Source copied; original data, checkpoints and results retained and hashed.'}
    write_json(manifest,value)
    return value
