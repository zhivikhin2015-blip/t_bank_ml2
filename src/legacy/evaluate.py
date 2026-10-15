"""Одинаковая оценка релевантности, поведения и модельной награды."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiment import (OUT, actor_logits, device, legal_logits, load_actor,
                        load_data, load_reward, macro, metadata, ranking,
                        seed_all, write_json)


@torch.no_grad()
def rerank(actor, reward, data, cat, dev, weight):
    top = []
    for start in range(0, len(data['target']), 32):
        h = data['history'][start:start+32]
        ts = data['timestamp'][start:start+32]
        logits = legal_logits(actor_logits(actor, h, dev), ts, cat, dev)
        values, candidates = logits.log_softmax(-1).topk(100, -1)
        context = reward.encode(h, dev)
        r = reward.score(context.repeat_interleave(100, 0), candidates.reshape(-1)).reshape(-1, 100)
        order = (values + weight*r).topk(20, -1).indices
        top.append(candidates.gather(1, order).cpu().numpy())
    return np.concatenate(top)


def ranking_from_top(top, data):
    matched = top == data['target'][:, None]
    metrics, per_event = {}, {}
    for k in (10, 20):
        for name, val in [(f'recall@{k}', matched[:, :k].any(-1).astype(float)),
                          (f'ndcg@{k}', (matched[:, :k]/np.log2(np.arange(k)+2)).sum(-1))]:
            metrics[name] = macro(val, data['uid'])
            per_event[name] = val
            known = data['target'] >= 2
            metrics[name+'_known'] = macro(val[known], data['uid'][known])
    metrics['examples'] = len(top)
    metrics['users'] = int(len(np.unique(data['uid'])))
    return metrics, per_event


@torch.no_grad()
def behavior(top, reward, auditor, data, cat, meta, dev):
    ratio, other = [], []
    for start in range(0, len(top), 64):
        h = data['history'][start:start+64]
        item = torch.as_tensor(top[start:start+64, :10], device=dev, dtype=torch.long)
        context = reward.encode(h, dev).repeat_interleave(10, 0)
        ratio.append(reward.score(context, item.reshape(-1)).reshape(-1, 10).cpu().numpy())
        if auditor is not None:
            context2 = auditor.encode(h, dev).repeat_interleave(10, 0)
            other.append(auditor.score(context2, item.reshape(-1)).reshape(-1, 10).cpu().numpy())
    ratio = np.concatenate(ratio)
    ids = top[:, :10]
    uid = data['uid']
    dur = cat['duration'][ids]
    time_score = ratio*np.minimum(dur, meta['duration_cap_seconds'])/meta['duration_cap_seconds']
    counts = np.bincount(ids.reshape(-1), minlength=len(cat['duration']))
    freq = counts[2:]
    p = freq[freq>0]/freq.sum()
    repeated_history = (ids[:, :, None] == data['history'][:, None, :]).any(-1).mean(-1)
    # Межпользовательское сходство: одна история на пользователя, фиксированные пары.
    _, first = np.unique(uid, return_index=True)
    rng = np.random.default_rng(1234)
    pairs = rng.integers(0, len(first), size=(min(2000, len(first)*2), 2))
    overlaps = [len(set(ids[first[a]]) & set(ids[first[b]]))/10 for a,b in pairs if a != b]
    result = dict(predicted_ratio_at10=macro(ratio.mean(-1), uid),
                  predicted_capped_seconds_at10=macro((ratio*np.minimum(dur, meta['duration_cap_seconds'])).mean(-1), uid),
                  normalized_time_reward_at10=macro(time_score.mean(-1), uid),
                  mean_duration_seconds=macro(dur.mean(-1), uid),
                  median_duration_seconds=float(np.median(dur)),
                  train_popularity=macro(cat['popularity'][ids].mean(-1), uid),
                  previously_listened_fraction=macro(repeated_history, uid),
                  catalog_coverage_at10=float(np.mean(freq>0)),
                  top10_items_exposure_share=float(np.sort(freq)[-10:].sum()/freq.sum()),
                  recommendation_entropy=float(-(p*np.log(p)).sum()),
                  between_user_overlap_at10=float(np.mean(overlaps)),
                  top_items=[{'token': int(i), 'item_id': int(cat['item_ids'][i-2]), 'recommendations': int(counts[i])}
                             for i in np.argsort(counts)[-10:][::-1]])
    if other:
        result['auditor_predicted_ratio_at10'] = macro(np.concatenate(other).mean(-1), uid)
    return result


@torch.no_grad()
def rollout(actor, ref, reward, auditor, data, cat, meta, dev, n_users=512):
    _, idx = np.unique(data['uid'], return_index=True)
    if len(idx) > n_users:
        idx = np.sort(np.random.default_rng(20261014).choice(idx, n_users, replace=False))
    records = []
    rng_seed = 20261014
    torch.manual_seed(rng_seed)
    duration = cat['duration']
    cap = meta['duration_cap_seconds']
    for start in range(0, len(idx), 32):
        loc = idx[start:start+32]
        original = data['history'][loc]
        ts = data['timestamp'][loc]
        h = original.copy()
        generated = []
        for t in range(5):
            logits = legal_logits(actor_logits(actor, h, dev), ts, cat, dev)
            lp = logits.log_softmax(-1)
            ref_lp = legal_logits(actor_logits(ref, h, dev), ts, cat, dev).log_softmax(-1)
            action = torch.multinomial(lp.exp(), 1).squeeze(-1)
            items = action.cpu().numpy()
            r = reward(h, action, dev).cpu().numpy()
            # Контроль: оценка того же действия на исходной реальной истории.
            original_score = reward(original, action, dev).cpu().numpy()
            ar = auditor(h, action, dev).cpu().numpy() if auditor else np.full(len(items), np.nan)
            kl = (lp.exp()*(lp-ref_lp)).sum(-1).cpu().numpy()
            repeated = np.zeros(len(items), dtype=bool) if not generated else (np.stack(generated, -1) == items[:, None]).any(-1)
            for j in range(len(items)):
                records.append(dict(uid=int(data['uid'][loc[j]]), step=t+1, token=int(items[j]),
                                    predicted_ratio=float(r[j]), original_context_ratio=float(original_score[j]),
                                    auditor_ratio=float(ar[j]) if auditor else None,
                                    predicted_capped_seconds=float(r[j]*min(duration[items[j]], cap)),
                                    duration_seconds=float(duration[items[j]]), repeat_in_rollout=bool(repeated[j]),
                                    seen_in_real_history=bool(items[j] in original[j]), kl_to_reference=float(kl[j])))
            generated.append(items)
            h = np.concatenate([h[:, 1:], items[:, None]], -1)
    summary = []
    for t in range(1, 6):
        rows = [r for r in records if r['step'] == t]
        row = {'step': t, 'users': len(rows)}
        for key in ['predicted_ratio', 'original_context_ratio', 'predicted_capped_seconds',
                    'duration_seconds', 'repeat_in_rollout', 'seen_in_real_history', 'kl_to_reference']:
            row[key] = float(np.mean([r[key] for r in rows]))
        if auditor:
            row['auditor_ratio'] = float(np.mean([r['auditor_ratio'] for r in rows]))
        summary.append(row)
    return summary, records


def paired_interval(values, reference, uid):
    users, inverse = np.unique(uid, return_inverse=True)
    counts = np.bincount(inverse)
    diff = np.bincount(inverse, weights=values-reference)/counts
    rng = np.random.default_rng(20261014)
    boot = np.array([diff[rng.integers(0, len(users), len(users))].mean() for _ in range(2000)])
    return dict(difference=float(diff.mean()), ci95=[float(x) for x in np.quantile(boot, [.025, .975])],
                unit='user', resamples=2000)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=['policy_val', 'test'], default='policy_val')
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--reference', default='baseline_s11')
    p.add_argument('--reward', default='reward_s11')
    p.add_argument('--auditor', default=None)
    p.add_argument('--rerank-weights', nargs='*', type=float, default=[])
    p.add_argument('--rollouts', action='store_true')
    args = p.parse_args()
    if args.split == 'test':
        frozen = json.loads((OUT / 'frozen_selection.json').read_text())
        assert set(args.models).issubset(set(frozen['models']))
        for name in args.models:
            assert hashlib.sha256((OUT/name/'best.pt').read_bytes()).hexdigest() == frozen['checkpoint_sha256'][name]
        for name, expected in frozen['source_sha256'].items():
            assert hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() == expected
        assert not (OUT / 'test_evaluation.json').exists(), 'Test уже оценён; не перезаписывайте без журнала технической причины'
    seed_all(11)
    meta, cat = metadata()
    data = load_data(args.split)
    dev = device()
    ref = load_actor(OUT / args.reference / 'best.pt', dev)
    reward = load_reward(OUT / args.reward / 'best.pt', dev, cat, meta)
    auditor = load_reward(OUT / args.auditor / 'best.pt', dev, cat, meta) if args.auditor else None
    results = {'split': args.split, 'data_sha256': meta['raw_sha256'], 'models': {},
               'reference_policy':args.reference,
               'reward_model': args.reward, 'auditor': args.auditor,
               'interpretation': 'Предсказанная награда не является измеренным откликом на новую политику.'}
    _, ref_top, ref_values = ranking(ref, data, cat, dev)
    for name in args.models:
        print(f'evaluating {name} on {args.split}', flush=True)
        actor = load_actor(OUT / name / 'best.pt', dev)
        metrics, top, per_event = ranking(actor, data, cat, dev)
        metrics.update(behavior(top, reward, auditor, data, cat, meta, dev))
        metrics['overlap_with_reference_at10'] = float(np.mean([len(set(a[:10]) & set(b[:10]))/10 for a,b in zip(top, ref_top)]))
        metrics['paired_comparison'] = {key: paired_interval(v, ref_values[key], data['uid']) for key,v in per_event.items()}
        if args.rollouts:
            summary, records = rollout(actor, ref, reward, auditor, data, cat, meta, dev)
            metrics['rollout'] = summary
            metrics['discounted_rollout_ratio_return'] = sum(.9**t*r['predicted_ratio'] for t,r in enumerate(summary))
            metrics['discounted_rollout_time_return'] = sum(.9**t*r['predicted_capped_seconds']/meta['duration_cap_seconds'] for t,r in enumerate(summary))
            metrics['step5_context_score_shift'] = summary[-1]['predicted_ratio']-summary[-1]['original_context_ratio']
            write_json(OUT / name / f'{args.split}_rollout_records.json', records)
        results['models'][name] = metrics
        np.savez_compressed(OUT / name / f'{args.split}_predictions.npz', top=top, uid=data['uid'], **per_event)
        write_json(OUT / f'{args.split}_evaluation.json', results)
        print(json.dumps({k:v for k,v in metrics.items() if isinstance(v, (int,float))}), flush=True)
        del actor
    for weight in args.rerank_weights:
        name = f'rerank_weight{weight:g}'
        top = rerank(ref, reward, data, cat, dev, weight)
        metrics, per_event = ranking_from_top(top, data)
        metrics.update(behavior(top, reward, auditor, data, cat, meta, dev))
        metrics['paired_comparison'] = {key: paired_interval(v, ref_values[key], data['uid']) for key,v in per_event.items()}
        results['models'][name] = metrics
        (OUT/name).mkdir(exist_ok=True)
        np.savez_compressed(OUT/name/f'{args.split}_predictions.npz', top=top, uid=data['uid'], **per_event)
        write_json(OUT / f'{args.split}_evaluation.json', results)
    # Все объекты каталога наблюдались до конца A, а оценка начинается в C/D.
    pop_order = np.lexsort((np.arange(len(cat['popularity'])), -cat['popularity']))
    pop_order = pop_order[pop_order >= 2][:20]
    pop_top = np.tile(pop_order, (len(data['target']), 1))
    pop_metrics, pop_values = ranking_from_top(pop_top, data)
    pop_metrics.update(behavior(pop_top, reward, auditor, data, cat, meta, dev))
    pop_metrics['paired_comparison'] = {key: paired_interval(v, ref_values[key], data['uid']) for key,v in pop_values.items()}
    results['models']['popularity'] = pop_metrics
    (OUT/'popularity').mkdir(exist_ok=True)
    np.savez_compressed(OUT/'popularity'/f'{args.split}_predictions.npz', top=pop_top, uid=data['uid'], **pop_values)
    results['completed'] = True
    write_json(OUT / f'{args.split}_evaluation.json', results)
    print(json.dumps({'completed':True,'split':args.split,'models':list(results['models']),
                      'result_file':str(OUT/f'{args.split}_evaluation.json')},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
