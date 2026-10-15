"""Воспроизводимая подготовка реальных прослушиваний Yambda-50M."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
OUT = ROOT / os.environ.get('ANNAI_RL_OUTPUT', 'output')
REVISION = 'dd6f3a19eef5866e346c3270e098baa641a44948'
EXPECTED_SHA256 = 'eed9cbd094af1e189507d2f8132a0dc9653b90e65480125c7cdccd601e0592d1'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--users', type=int, default=1024)
    p.add_argument('--items', type=int, default=5000)
    p.add_argument('--history', type=int, default=64)
    p.add_argument('--train-per-user', type=int, default=200)
    p.add_argument('--eval-per-user', type=int, default=24)
    args = p.parse_args()
    raw = OUT / 'listens.parquet'
    actual_hash = digest(raw)
    assert actual_hash == EXPECTED_SHA256, 'Хеш исходных данных не совпадает'
    parquet = pq.ParquetFile(raw)
    print(parquet.schema_arrow, flush=True)
    uids = np.unique(pq.read_table(raw, columns=['uid'])['uid'].to_numpy())
    # Отбор по хешу ID не использует будущие отклики или активность.
    users = sorted(uids.tolist(), key=lambda u: hashlib.sha256(str(u).encode()).digest())[:args.users]
    table = ds.dataset(raw).to_table(filter=ds.field('uid').isin(users))
    a = {name: table[name].to_numpy() for name in table.column_names}
    assert all(table[name].null_count == 0 for name in table.column_names)
    order = np.lexsort((a['item_id'], a['timestamp'], a['uid']))
    a = {k: v[order] for k, v in a.items()}
    lo, hi = int(a['timestamp'].min()), int(a['timestamp'].max())
    cuts = np.array([lo + int((hi - lo) * x) for x in (.7, .8, .9)], dtype=np.int64)
    split = np.searchsorted(cuts, a['timestamp'], side='right')
    train = split == 0
    item_ids, counts = np.unique(a['item_id'][train], return_counts=True)
    chosen = np.lexsort((item_ids, -counts))[:args.items]
    item_ids, counts = item_ids[chosen], counts[chosen]
    # PAD=0, UNK=1; объекты получают номера начиная с 2.
    mapping = {int(i): j + 2 for j, i in enumerate(item_ids)}
    tokens = np.fromiter((mapping.get(int(i), 1) for i in a['item_id']), dtype=np.int32)
    vocab = len(item_ids) + 2
    duration = np.zeros(vocab, dtype=np.float32)
    first_seen = np.full(vocab, hi + 1, dtype=np.int64)
    means = np.zeros(vocab, dtype=np.float32)
    clipped = np.clip(a['played_ratio_pct'].astype(np.float32) / 100, 0, 1)
    duration_inconsistent = 0
    # Метаданные кандидатов извлекаются только из train.
    train_loc = np.flatnonzero(train & (tokens >= 2))
    train_loc = train_loc[np.argsort(tokens[train_loc], kind='stable')]
    groups = np.r_[0, np.flatnonzero(np.diff(tokens[train_loc])) + 1, len(train_loc)]
    for begin, end_group in zip(groups[:-1], groups[1:]):
        loc = train_loc[begin:end_group]
        token = int(tokens[loc[0]])
        earliest = loc[np.argmin(a['timestamp'][loc])]
        duration[token] = a['track_length_seconds'][earliest]
        first_seen[token] = a['timestamp'][earliest]
        means[token] = clipped[loc].mean()
        duration_inconsistent += int(np.unique(a['track_length_seconds'][loc]).size > 1)
    dcap = float(np.quantile(duration[2:], .95))
    popularity = np.zeros(vocab, dtype=np.float32)
    popularity[2:] = counts
    global_mean = float(clipped[train].mean())
    storage = [[] for _ in range(4)]
    names = ['train', 'reward_val', 'policy_val', 'test']
    rng = np.random.default_rng(20261014)
    excluded_ties = 0
    excluded_cold_users = 0
    start = np.r_[0, np.flatnonzero(np.diff(a['uid'])) + 1]
    end = np.r_[start[1:], len(tokens)]
    eligibility = []
    # Выбираем ограниченное число целей на пользователя, сохраняя всю историю.
    for left, right in zip(start, end):
        if np.sum(train[left:right]) < 20:
            excluded_cold_users += 1
            continue
        ts = a['timestamp'][left:right]
        tie = np.r_[False, np.diff(ts) == 0] | np.r_[np.diff(ts) == 0, False]
        excluded_ties += int(tie.sum())
        uid = int(a['uid'][left])
        eligibility.append(uid)
        for s in range(4):
            idx = np.flatnonzero((split[left:right] == s) & ~tie & (np.arange(right-left) >= 5)) + left
            # Для обучения и reward evaluation нужны известные объекты.
            # В policy/test оставляем неизвестные цели: они считаются промахами.
            if s in (0, 1):
                idx = idx[tokens[idx] >= 2]
                boundary = cuts[s]
                played_seconds = a['track_length_seconds'][idx].astype(np.float64) * a['played_ratio_pct'][idx] / 100
                idx = idx[a['timestamp'][idx].astype(np.float64) + np.maximum(played_seconds, a['track_length_seconds'][idx]) < boundary]
            cap = args.train_per_user if s == 0 else args.eval_per_user
            if len(idx) > cap:
                idx = np.sort(rng.choice(idx, cap, replace=False))
            for j in idx:
                hist = tokens[max(left, j - args.history):j]
                padded = np.zeros(args.history, dtype=np.int32)
                padded[-len(hist):] = hist
                loss_mask = np.zeros(args.history, dtype=np.int8)
                begin = max(left, j-args.history)
                # Для all-position NIP исключаем также неоднозначные цели
                # ВНУТРИ контекста, а не только последнюю цель окна.
                loss_mask[-len(hist):] = (~tie[begin-left+1:j-left+1]).astype(np.int8)
                assert np.all(a['timestamp'][max(left, j-args.history):j] < a['timestamp'][j])
                storage[s].append((padded, tokens[j], clipped[j], uid, a['timestamp'][j], a['is_organic'][j], a['track_length_seconds'][j], loss_mask))
    stats = {}
    for s, name in enumerate(names):
        rows = storage[s]
        assert rows, f'Пустой период {name}'
        histories = np.stack([r[0] for r in rows])
        target = np.array([r[1] for r in rows], dtype=np.int32)
        np.savez_compressed(OUT / f'{name}.npz', history=histories, target=target,
                            loss_mask=np.stack([r[7] for r in rows]),
                            ratio=np.array([r[2] for r in rows], dtype=np.float32),
                            uid=np.array([r[3] for r in rows], dtype=np.int64),
                            timestamp=np.array([r[4] for r in rows], dtype=np.int64),
                            organic=np.array([r[5] for r in rows], dtype=np.int8),
                            duration=np.array([r[6] for r in rows], dtype=np.float32))
        stats[name] = {'examples': len(rows), 'users': len(set(r[3] for r in rows)),
                       'known_target_fraction': float(np.mean(target >= 2)),
                       'timestamp_min': int(min(r[4] for r in rows)),
                       'timestamp_max': int(max(r[4] for r in rows))}
    np.savez_compressed(OUT / 'catalog.npz', item_ids=item_ids, duration=duration,
                        first_seen=first_seen, popularity=popularity, item_mean=means)
    # Признаки возвращения: только полное 7-дневное окно внутри того же периода.
    # По формуле в статье: секунды, округлённые до кратных пяти.
    # Это НЕ номер 5-секундного бина; календарное начало скрыто.
    seven_days = 7 * 24 * 3600
    one_day = 24 * 3600
    return_stats = {}
    boundaries = [lo, *cuts.tolist(), hi + 1]
    for s, name in enumerate(names[:3]):
        n, returned = 0, 0
        for left, right in zip(start, end):
            if int(a['uid'][left]) not in eligibility:
                continue
            ts = a['timestamp'][left:right].astype(np.int64)
            valid = (ts >= boundaries[s]) & (ts + seven_days < boundaries[s+1])
            idx = np.flatnonzero(valid)[::100]
            future = np.searchsorted(ts, ts[idx] + one_day, side='left')
            good = future < len(ts)
            labels = np.zeros(len(idx), dtype=bool)
            labels[good] = ts[future[good]] <= ts[idx[good]] + seven_days
            n += len(idx)
            returned += int(labels.sum())
        return_stats[name] = {'complete_windows': n, 'active_between_day1_and_day7': returned,
                              'fraction': returned/n if n else None}
    manifest = dict(dataset='yandex/yambda', revision=REVISION, raw_sha256=actual_hash,
                    raw_rows=parquet.metadata.num_rows, selected_rows=len(tokens),
                    requested_users=args.users, eligible_users=len(eligibility),
                    excluded_cold_users=excluded_cold_users, catalog_items=len(item_ids),
                    history_length=args.history, timestamp_min=lo, timestamp_max=hi,
                    timestamp_unit_seconds=1, timestamp_resolution_seconds=5,
                    temporal_cuts=cuts.tolist(), splits=stats,
                    train_catalog_event_coverage=float(np.mean(tokens[train] >= 2)),
                    duration_cap_seconds=dcap,
                    raw_train_global_ratio_mean=global_mean,
                    global_ratio_mean=float(np.mean([r[2] for r in storage[0]])),
                    train_ratio_above100_fraction=float(np.mean(a['played_ratio_pct'][train] > 100)),
                    tied_target_events_excluded=excluded_ties,
                    items_with_variable_train_duration=duration_inconsistent,
                    return_proxy=return_stats, preparation=vars(args))
    manifest['prepared_sha256'] = {name: digest(OUT/name) for name in
                                   ['train.npz', 'reward_val.npz', 'policy_val.npz', 'test.npz', 'catalog.npz']}
    (OUT / 'data_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
