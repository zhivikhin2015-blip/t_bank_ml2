"""Проверяемые входы, атомарные артефакты и совместимые модели v2."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve().parent
OUTPUT = PROJECT / os.environ.get('ANNAI_V2_OUTPUT', 'output/research_v2')
ORIGINAL = PROJECT / 'output'
os.environ.setdefault('HF_HOME', str(PROJECT / '.tmp' / 'research_v2' / 'huggingface'))
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT / '.tmp' / 'research_v2' / 'matplotlib'))

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from experiment import (RewardNet, actor_logits, device, inputs, legal_logits,
                        load_actor, macro, model_config, read_checkpoint, seed_all, sync)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {name: f[name] for name in f.files}


def protocol():
    return read_json(SOURCE / 'protocol.json')


def log_access(resource, reason, **details):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    event = {'time_utc': datetime.now(timezone.utc).isoformat(),
             'resource': resource, 'reason': reason, **details}
    with (OUTPUT / 'access_log.jsonl').open('a') as f:
        f.write(json.dumps(event, ensure_ascii=False) + '\n')


def source_hashes():
    files = [*SOURCE.glob('*.py'), SOURCE / 'protocol.json', PROJECT / 'requirements.txt',
             PROJECT / 'experiment.py']
    return {str(p.relative_to(PROJECT)): sha(p) for p in sorted(files)}


def check_sources(expected):
    for filename, expected_hash in expected.items():
        if sha(PROJECT / filename) != expected_hash:
            raise ValueError(f'Исходный код/протокол изменён: {filename}')


def original_integrity():
    snapshot = read_json(OUTPUT / 'original_snapshot' / 'manifest.json')
    for filename, expected in snapshot['files'].items():
        if sha(PROJECT / filename) != expected['sha256']:
            raise ValueError(f'Исходный эксперимент изменён: {filename}')
    return snapshot


def data_manifest():
    return read_json(OUTPUT / 'data_manifest.json')


def catalog():
    meta = data_manifest()
    path = OUTPUT / 'catalog.npz'
    if sha(path) != meta['files']['catalog.npz']:
        raise ValueError('Изменён каталог')
    return load_npz(path)


def load_data(split, reason):
    meta = data_manifest()
    path = OUTPUT / f'{split}.npz'
    if split == 'holdout':
        frozen = read_json(OUTPUT / 'frozen.json')
        check_frozen(frozen)
        expected = read_json(OUTPUT / 'holdout_manifest.json')['sha256']
    else:
        expected = meta['files'][path.name]
    if sha(path) != expected:
        raise ValueError(f'Изменены данные {split}')
    log_access(split, reason, sha256=expected)
    return load_npz(path)


def check_frozen(frozen=None):
    frozen = frozen or read_json(OUTPUT / 'frozen.json')
    check_sources(frozen['sources'])
    if sha(OUTPUT / 'data_manifest.json') != frozen['data_manifest_sha256']:
        raise ValueError('Изменён манифест подготовленных данных')
    for filename, expected in frozen['artifacts'].items():
        if sha(PROJECT / filename) != expected:
            raise ValueError(f'Зафиксированный артефакт изменён: {filename}')
    return frozen


def user_weights(uid):
    """Сумма весов = 1; полный вклад каждого пользователя одинаков."""
    users, inverse, counts = np.unique(uid, return_inverse=True, return_counts=True)
    return 1. / (len(users) * counts[inverse])


class UserSampler:
    """Сэмплирование с возвращением: равновероятный пользователь, затем событие."""
    def __init__(self, uid):
        users, inverse = np.unique(uid, return_inverse=True)
        self.rows = [np.flatnonzero(inverse == i) for i in range(len(users))]

    def sample(self, rng, batch):
        selected = rng.integers(0, len(self.rows), batch)
        return np.array([self.rows[u][rng.integers(len(self.rows[u]))] for u in selected])


class CalibratedReward(RewardNet):
    def __init__(self, backbone, cat, cap, variant):
        super().__init__(backbone, cat, cap)
        self.variant = variant
        self.register_buffer('calibration_slope', torch.tensor(1.))
        self.register_buffer('calibration_intercept', torch.tensor(0.))

    def encode(self, hist, dev):
        if self.variant == 'item_duration':
            return torch.zeros(len(hist), self.backbone.config.n_embd, device=dev)
        return super().encode(hist, dev)

    def raw_score(self, context, item):
        if self.variant == 'item_duration':
            context = torch.zeros_like(context)
        d = torch.log1p(self.duration[item]) / np.log1p(self.cap)
        if self.variant == 'history_no_duration':
            d = torch.zeros_like(d)
        features = torch.cat([context, self.item(item), d.unsqueeze(-1)], -1)
        return self.head(features).squeeze(-1).sigmoid()

    def score(self, context, item):
        return (self.calibration_intercept + self.calibration_slope *
                self.raw_score(context, item)).clamp(0, 1)


def load_reward(path, dev, cat, meta):
    state = read_checkpoint(path)
    cfg = GPT2Config(**state['config']['gpt2'])
    cfg._attn_implementation = 'eager'
    model = CalibratedReward(GPT2LMHeadModel(cfg), cat, meta['duration_cap_seconds'],
                             state['config']['variant']).to(dev)
    model.load_state_dict(state['model'])
    return model.eval()


def capture_rng(rng):
    value = {'numpy': rng.bit_generator.state, 'torch': torch.get_rng_state()}
    if torch.backends.mps.is_available():
        value['mps'] = torch.mps.get_rng_state()
    return value


def restore_rng(value, rng):
    rng.bit_generator.state = value['numpy']
    torch.set_rng_state(value['torch'])
    if 'mps' in value and torch.backends.mps.is_available():
        torch.mps.set_rng_state(value['mps'])


def save_torch(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(state, tmp)
    tmp.replace(path)


def cache_array(path, fingerprint, compute):
    """Кеш используется только при совпадении всех его входов и контрольной суммы."""
    path = Path(path)
    manifest_path = path.with_suffix('.manifest.json')
    if path.exists() and manifest_path.exists():
        saved = read_json(manifest_path)
        if saved['fingerprint'] != fingerprint or saved['sha256'] != sha(path):
            raise ValueError(f'Кеш не соответствует входам: {path.name}')
        return np.load(path, allow_pickle=False)
    if path.exists() or manifest_path.exists():
        # Артефакт не завершён; точные входы пересчитываются в том же пространстве кеша.
        path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        np.save(f, value, allow_pickle=False)
    tmp.replace(path)
    write_json(manifest_path, {'fingerprint': fingerprint, 'sha256': sha(path),
                               'shape': list(value.shape), 'dtype': str(value.dtype)})
    return value


def affine_calibration(pred, target, uid):
    w = user_weights(uid)
    xbar, ybar = np.sum(w * pred), np.sum(w * target)
    variance = np.sum(w * (pred - xbar) ** 2)
    slope = max(0., np.sum(w * (pred - xbar) * (target - ybar)) / variance) if variance > 1e-12 else 0.
    return {'slope': float(slope), 'intercept': float(ybar - slope * xbar)}


def paired_interval(value, reference, uid, factor=1., resamples=None, one_sided=False):
    cfg = protocol()['evaluation']
    users, inverse, counts = np.unique(uid, return_inverse=True, return_counts=True)
    diff = np.bincount(inverse, weights=value - factor * reference) / counts
    rng = np.random.default_rng(cfg['bootstrap_seed'])
    boot = np.array([diff[rng.integers(0, len(users), len(users))].mean()
                     for _ in range(resamples or cfg['bootstrap_resamples'])])
    result = {'difference': float(diff.mean()), 'ci95': np.quantile(boot, [.025, .975]).tolist(),
              'users': len(users), 'unit': 'user', 'factor': factor,
              'resamples': len(boot), 'scope': 'fixed period and fitted weights'}
    if one_sided:
        result['lower_one_sided95'] = float(np.quantile(boot, .05))
        result['noninferiority_supported'] = result['lower_one_sided95'] > 0
    return result
