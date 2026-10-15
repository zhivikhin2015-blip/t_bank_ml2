"""Проверка зафиксированных результатов и изолированный запуск исходного протокола."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LAYOUT = ROOT / 'configs' / 'reproduction_layout.json'


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def layout() -> dict:
    return json.loads(LAYOUT.read_text())


def resolve_record(record: dict) -> Path:
    path = ROOT / record['path']
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError(f'Путь выходит за границы проекта: {record["path"]}')
    return path


def verify() -> None:
    config = layout()
    records = {**config['sources'], **config['artifacts']}
    for logical, record in records.items():
        actual = digest(resolve_record(record))
        if actual != record['sha256']:
            raise ValueError(f'Изменён файл: {logical} -> {record["path"]}')
    original = json.loads((ROOT / config['original_manifest']).read_text())
    for logical, record in original['files'].items():
        if records[logical]['sha256'] != record['sha256']:
            raise ValueError(f'Нарушена исходная контрольная сумма: {logical}')
    frozen = json.loads((ROOT / config['frozen_manifest']).read_text())
    for category in ['sources', 'artifacts']:
        for logical, expected in frozen[category].items():
            if records[logical]['sha256'] != expected:
                raise ValueError(f'Нарушена фиксация эксперимента: {logical}')
    print(f'OK: {len(config["sources"])} исходных файлов, '
          f'{len(config["artifacts"])} научных артефактов; '
          f'{len(original["files"])} исходных и {len(frozen["sources"])} '
          'замороженных исходников проверены.')


def prepare_workspace(destination: Path, *, attach_raw: bool = True) -> Path:
    """Воссоздать структуру запуска, не связывая её с сохранёнными результатами."""
    destination = destination.resolve()
    protected = [ROOT, *(ROOT / name for name in
                 ['src', 'configs', 'data', 'results', 'tests', 'reports', 'docs', '.venv', '.git'])]
    if any(destination == p or (p != ROOT and destination.is_relative_to(p)) for p in protected):
        raise ValueError('Рабочая копия должна находиться вне каталогов исходников и результатов')
    config = layout()
    fingerprint = hashlib.sha256(json.dumps(config['sources'], sort_keys=True).encode()).hexdigest()
    marker = destination / '.reproduction.json'
    if destination.exists() and any(destination.iterdir()):
        if not marker.exists() or json.loads(marker.read_text())['sources_sha256'] != fingerprint:
            raise ValueError('Каталог занят или исходники существующего запуска изменились')
    destination.mkdir(parents=True, exist_ok=True)
    for logical, record in config['sources'].items():
        source = resolve_record(record)
        if digest(source) != record['sha256']:
            raise ValueError(f'Исходник изменён: {record["path"]}')
        target = destination / logical
        if target.exists():
            if logical == 'research_v2/README.md':
                continue  # Generated report, not part of the frozen numerical source set.
            if digest(target) != record['sha256']:
                raise ValueError(f'Изменён исходник рабочей копии: {logical}')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    output = destination / 'output'
    output.mkdir(exist_ok=True)
    raw = ROOT / 'data' / 'raw' / 'listens.parquet'
    target_raw = output / 'listens.parquet'
    if attach_raw and raw.exists():
        record = config['artifacts']['output/listens.parquet']
        if digest(raw) != record['sha256']:
            raise ValueError('Повреждён исходный Parquet')
        if target_raw.exists():
            if digest(target_raw) != record['sha256']:
                raise ValueError('Рабочая копия содержит другой датасет')
        else:
            target_raw.symlink_to(os.path.relpath(raw, output))
    marker.write_text(json.dumps({'sources_sha256': fingerprint}, indent=2))
    return destination


def command(stage: str, extra: list[str]) -> list[list[str]]:
    if stage == 'download':
        return [['download.py']]
    if stage == 'prepare-baseline':
        return [['prepare.py']]
    if stage == 'baseline':
        return [['experiment.py', 'baseline', *extra]]
    if stage == 'holdout':
        return [['-c', 'from research_v2.data import prepare_holdout; prepare_holdout()'],
                ['research_v2_runtime.py', 'holdout']]
    if stage == 'audit':
        return [['research_v2_runtime.py', 'audit']]
    if stage == 'serving':
        return [['research_v2_serving.py']]
    if stage == 'report':
        return [['research_v2_finalize.py']]
    return [['-m', 'research_v2.run', stage]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('verify', help='Проверить все сохранённые контрольные суммы')
    init = sub.add_parser('init', help='Создать изолированную рабочую копию')
    init.add_argument('--workspace', type=Path, default=ROOT / '.cache' / 'reproduction')
    run = sub.add_parser('run', help='Выполнить этап в рабочей копии')
    run.add_argument('stage', choices=['download', 'prepare-baseline', 'baseline', 'prepare',
                     'checks', 'benchmark', 'reward', 'pilot', 'retention', 'training',
                     'freeze', 'holdout', 'audit', 'serving', 'report'])
    run.add_argument('--workspace', type=Path, default=ROOT / '.cache' / 'reproduction')
    run.add_argument('--limit-steps', type=int, default=None)
    args, extra = parser.parse_known_args()
    if extra and not (args.action == 'run' and args.stage == 'baseline' and extra[0] == '--'):
        parser.error('Дополнительные параметры допустимы только после run baseline --')
    if args.action == 'verify':
        verify()
        return
    workspace = prepare_workspace(args.workspace)
    if args.action == 'init':
        print(workspace)
        return
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['MPLCONFIGDIR'] = str(ROOT / '.cache' / 'matplotlib')
    env['HF_HOME'] = str(ROOT / '.cache' / 'huggingface')
    # Старые имена переменных поддерживаются только внутри зафиксированного протокола.
    # Внешняя среда не должна перенаправить выход нового запуска в исходные results/.
    env.pop('ANNAI_RL_OUTPUT', None)
    env.pop('ANNAI_V2_OUTPUT', None)
    if args.limit_steps is not None and args.stage not in ['reward', 'pilot', 'training']:
        parser.error('--limit-steps применяется к reward, pilot или training')
    for invocation in command(args.stage, extra[1:] if extra else []):
        if args.limit_steps is not None:
            invocation += ['--limit-steps', str(args.limit_steps)]
        subprocess.run([sys.executable, '-B', *invocation], cwd=workspace, env=env, check=True)


if __name__ == '__main__':
    main()
