"""Подготовка development-данных и отложенный доступ к новой когорте."""
import hashlib
import shutil

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .core import (ORIGINAL, OUTPUT, PROJECT, SOURCE, check_frozen, load_npz,
                   log_access, original_integrity, protocol, read_json, save_npz,
                   sha, source_hashes, user_weights, write_json)


def _rows(data, mask):
    return {name: value[mask] for name, value in data.items()}


def _summary(data):
    return {'examples': len(data['uid']), 'users': len(np.unique(data['uid'])),
            'known_fraction': float(np.mean(data['target'] >= 2)),
            'timestamp_min': int(data['timestamp'].min()),
            'timestamp_max': int(data['timestamp'].max())}


def prepare():
    cfg = protocol()
    locked_path = OUTPUT / 'locked_protocol.json'
    if locked_path.exists() and read_json(locked_path) != cfg:
        raise ValueError('Протокол изменён после подготовки; нужен отдельный новый запуск')
    from .bootstrap import snapshot_original
    snapshot_original()
    if (OUTPUT / 'data_manifest.json').exists():
        meta = read_json(OUTPUT / 'data_manifest.json')
        for filename, expected in meta['files'].items():
            if sha(OUTPUT / filename) != expected:
                raise ValueError(f'Подготовленные данные изменены: {filename}')
        print('Development-данные уже подготовлены и проверены', flush=True)
        return meta
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(locked_path, cfg)
    raw = ORIGINAL / 'listens.parquet'
    if sha(raw) != cfg['dataset']['raw_sha256']:
        raise ValueError('Не совпадает хеш исходного Parquet')
    old_meta = read_json(ORIGINAL / 'data_manifest.json')
    a_end, b_end, c_end = old_meta['temporal_cuts']
    b_mid = (a_end + b_end) // 2
    b = load_npz(ORIGINAL / 'reward_val.npz')
    log_access('original B', 'Prepare B1/B2 without old D outcomes')
    # Точная зрелость B1 проверяется по необрезанному исходному отклику.
    table = ds.dataset(raw).to_table(columns=['uid', 'timestamp', 'played_ratio_pct', 'track_length_seconds'],
              filter=(ds.field('uid').isin(np.unique(b['uid']).tolist()) &
                      (ds.field('timestamp') >= a_end) & (ds.field('timestamp') < b_mid)))
    arr = {name: table[name].to_numpy() for name in table.column_names}
    completed = {(int(u), int(t)): float(t) + max(float(d), float(d) * float(y) / 100)
                 for u, t, d, y in zip(arr['uid'], arr['timestamp'], arr['track_length_seconds'], arr['played_ratio_pct'])}
    is_b1 = b['timestamp'] < b_mid
    mature = np.array([completed.get((int(u), int(t)), float('inf')) < b_mid
                      if early else True for u, t, early in zip(b['uid'], b['timestamp'], is_b1)])
    b1, b2 = _rows(b, is_b1 & mature), _rows(b, ~is_b1)
    save_npz(OUTPUT / 'reward_select.npz', **b1)
    save_npz(OUTPUT / 'reward_calibrate.npz', **b2)
    for name in ['train.npz', 'policy_val.npz', 'catalog.npz']:
        shutil.copy2(ORIGINAL / name, OUTPUT / name)
    # ID-only inventory: no new cohort outcomes or D timestamps are read here.
    uid_inventory = np.unique(pq.read_table(raw, columns=['uid'])['uid'].to_numpy())
    ordered = sorted(uid_inventory.tolist(), key=lambda u: hashlib.sha256(str(u).encode()).digest())
    original_users = ordered[:cfg['dataset']['original_user_count']]
    new_users = ordered[cfg['dataset']['original_user_count']:
                        cfg['dataset']['original_user_count'] + cfg['dataset']['new_user_count']]
    assert set(new_users).isdisjoint(original_users)
    # Eligibility uses only information available at end A, not future activity.
    past = ds.dataset(raw).to_table(columns=['uid'], filter=(ds.field('uid').isin(new_users) &
                                                            (ds.field('timestamp') < a_end)))
    ids, counts = np.unique(past['uid'].to_numpy(), return_counts=True)
    eligible = ids[counts >= cfg['dataset']['minimum_train_events']].tolist()
    reservation = {'reserved_users': new_users, 'eligible_users': eligible,
                   'original_users_sha256': hashlib.sha256(np.asarray(original_users, dtype=np.int64).tobytes()).hexdigest(),
                   'disjoint_from_original': True, 'future_outcomes_read': False,
                   'eligibility_uses': 'A timestamps only', 'new_time_period': False}
    write_json(OUTPUT / 'holdout_reservation.json', reservation)
    log_access('new cohort IDs and A eligibility', 'Reserve users before outcome access',
               reserved_users=len(new_users), eligible_users=len(eligible))
    # Source footer can establish observation extent without reading held-out outcomes.
    parquet = pq.ParquetFile(raw)
    ts_index = parquet.schema_arrow.names.index('timestamp')
    maxima = [parquet.metadata.row_group(i).column(ts_index).statistics.max
              for i in range(parquet.num_row_groups)
              if parquet.metadata.row_group(i).column(ts_index).statistics is not None]
    raw_end = int(max(maxima)) if maxima else None
    train = load_npz(OUTPUT / 'train.npz')
    weights = user_weights(train['uid'])
    vocab = old_meta['catalog_items'] + 2
    counts = np.bincount(train['target'], minlength=vocab)
    weighted_counts = np.bincount(train['target'], weights=weights, minlength=vocab)
    weighted_sums = np.bincount(train['target'], weights=weights * train['ratio'], minlength=vocab)
    global_mean = float(np.sum(weights * train['ratio']))
    # Pseudocount is expressed in the same weighted-observation scale.
    pseudoweight = cfg['reward']['baselines']['item_mean_shrinkage'] * weights.mean()
    item_mean = (weighted_sums + pseudoweight * global_mean) / (weighted_counts + pseudoweight)
    save_npz(OUTPUT / 'reward_baselines.npz', item_mean=item_mean, counts=counts,
             global_mean=np.array(global_mean), pseudoweight=np.array(pseudoweight))
    files = ['train.npz', 'policy_val.npz', 'catalog.npz', 'reward_select.npz',
             'reward_calibrate.npz', 'reward_baselines.npz', 'holdout_reservation.json', 'locked_protocol.json']
    meta = {'raw_sha256': cfg['dataset']['raw_sha256'], 'revision': cfg['dataset']['revision'],
            'files': {name: sha(OUTPUT / name) for name in files},
            'catalog_items': old_meta['catalog_items'], 'history_length': old_meta['history_length'],
            'duration_cap_seconds': old_meta['duration_cap_seconds'],
            'cuts': {'A_end': a_end, 'B1_end': b_mid, 'B_end': b_end, 'C_end': c_end,
                     'D_end_exclusive': old_meta['timestamp_max'] + 1},
            'splits': {'train': _summary(train), 'reward_select': _summary(b1),
                       'reward_calibrate': _summary(b2), 'policy_val': _summary(load_npz(OUTPUT / 'policy_val.npz'))},
            'B1_crossing_outcomes_removed': int(np.sum(is_b1 & ~mature)),
            'new_cohort_reserved': len(new_users), 'new_cohort_eligible': len(eligible),
            'source_footer_timestamp_max': raw_end,
            'unused_later_seconds': max(0, raw_end - old_meta['timestamp_max']) if raw_end is not None else None,
            'new_holdout_kind': 'unseen users, same D time window; no new temporal replication',
            'old_D_accessed_by_v2': False,
            'preparation_sources': source_hashes()}
    write_json(OUTPUT / 'data_manifest.json', meta)
    print({key: meta[key] for key in ['splits', 'new_cohort_reserved', 'new_cohort_eligible', 'unused_later_seconds']}, flush=True)
    return meta


def prepare_holdout():
    frozen = check_frozen()
    meta = read_json(OUTPUT / 'data_manifest.json')
    if (OUTPUT / 'holdout_manifest.json').exists():
        saved = read_json(OUTPUT / 'holdout_manifest.json')
        assert sha(OUTPUT / 'holdout.npz') == saved['sha256']
        assert saved['frozen_sha256'] == sha(OUTPUT / 'frozen.json')
        return saved
    reservation = read_json(OUTPUT / 'holdout_reservation.json')
    log_access('new cohort full histories', 'Final evaluation after freeze',
               frozen_sha256=sha(OUTPUT / 'frozen.json'))
    table = ds.dataset(ORIGINAL / 'listens.parquet').to_table(
        filter=ds.field('uid').isin(reservation['eligible_users']))
    a = {name: table[name].to_numpy() for name in table.column_names}
    order = np.lexsort((a['item_id'], a['timestamp'], a['uid']))
    a = {name: value[order] for name, value in a.items()}
    cat = load_npz(OUTPUT / 'catalog.npz')
    mapping = {int(item): j + 2 for j, item in enumerate(cat['item_ids'])}
    tokens = np.fromiter((mapping.get(int(i), 1) for i in a['item_id']), dtype=np.int32)
    starts = np.r_[0, np.flatnonzero(np.diff(a['uid'])) + 1]
    ends = np.r_[starts[1:], len(tokens)]
    cfg = protocol()['dataset']
    rng = np.random.default_rng(cfg['sampling_seed'])
    histories, indices = [], []
    tied, immature = 0, 0
    for left, right in zip(starts, ends):
        ts = a['timestamp'][left:right]
        ties = np.r_[False, np.diff(ts) == 0] | np.r_[np.diff(ts) == 0, False]
        in_d = (ts >= meta['cuts']['C_end']) & (ts < meta['cuts']['D_end_exclusive'])
        tied += int(np.sum(in_d & ties))
        candidates = np.flatnonzero(in_d & ~ties & (np.arange(len(ts)) >= 5)) + left
        duration = a['track_length_seconds'][candidates].astype(float)
        completion = a['timestamp'][candidates].astype(float) + np.maximum(
            duration, duration * a['played_ratio_pct'][candidates] / 100)
        mature = completion < meta['cuts']['D_end_exclusive']
        immature += int(np.sum(~mature))
        candidates = candidates[mature]
        if len(candidates) > cfg['eval_per_user']:
            candidates = np.sort(rng.choice(candidates, cfg['eval_per_user'], replace=False))
        for j in candidates:
            hist = tokens[max(left, j - cfg['history_length']):j]
            padded = np.zeros(cfg['history_length'], dtype=np.int32)
            padded[-len(hist):] = hist
            assert np.all(a['timestamp'][max(left, j - cfg['history_length']):j] < a['timestamp'][j])
            histories.append(padded)
            indices.append(j)
    idx = np.array(indices)
    if not len(idx):
        raise ValueError('Пустая новая окончательная выборка')
    data = {'history': np.stack(histories), 'target': tokens[idx],
            'ratio': np.clip(a['played_ratio_pct'][idx].astype(np.float32) / 100, 0, 1),
            'uid': a['uid'][idx], 'timestamp': a['timestamp'][idx],
            'organic': a['is_organic'][idx], 'duration': a['track_length_seconds'][idx]}
    save_npz(OUTPUT / 'holdout.npz', **data)
    info = {'sha256': sha(OUTPUT / 'holdout.npz'), 'frozen_sha256': sha(OUTPUT / 'frozen.json'),
            'summary': _summary(data), 'tied_D_events_excluded': tied,
            'immature_D_outcomes_excluded': immature,
            'scope': meta['new_holdout_kind'], 'model_selection_allowed': False}
    write_json(OUTPUT / 'holdout_manifest.json', info)
    return info


if __name__ == '__main__':
    prepare()
