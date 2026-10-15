"""Оценка полного распределения действий с проверяемым восстановлением по моделям."""
import time

import numpy as np
import torch
from torch.nn import functional as F

from .core import (OUTPUT, PROJECT, SOURCE, actor_logits, cache_array, canonical_hash,
                   capture_rng, catalog, check_frozen, data_manifest, device, inputs,
                   legal_logits, load_actor, load_data, load_npz, load_reward, log_access,
                   macro, paired_interval, protocol, read_json, restore_rng, save_npz,
                   sha, sync, write_json)
from .metrics import behavior, relevance, reward_quality
from .train import actor_path


def evaluation_dependencies():
    return {str(p.relative_to(PROJECT)): sha(p) for p in
            [SOURCE / 'core.py', SOURCE / 'evaluate.py', SOURCE / 'metrics.py',
             SOURCE / 'protocol.json', PROJECT / 'experiment.py']}


def data_hash(split):
    return (read_json(OUTPUT / 'holdout_manifest.json')['sha256'] if split == 'holdout'
            else data_manifest()['files'][split + '.npz'])


@torch.no_grad()
def all_scores(model, context):
    """Факторизация первого Linear; это тот же MLP для всех item ID."""
    first, second, final = model.head[0], model.head[2], model.head[4]
    d = model.backbone.config.n_embd
    if model.variant == 'item_duration':
        context = torch.zeros_like(context)
    context_term = F.linear(context, first.weight[:, :d], first.bias)
    item_term = F.linear(model.item.weight, first.weight[:, d:2*d])
    duration = torch.log1p(model.duration) / np.log1p(model.cap)
    if model.variant != 'history_no_duration':
        item_term = item_term + duration[:, None] * first.weight[:, -1][None, :]
    hidden = F.relu(context_term[:, None, :] + item_term[None, :, :])
    hidden = F.relu(F.linear(hidden, second.weight, second.bias))
    raw = F.linear(hidden, final.weight, final.bias).squeeze(-1).sigmoid()
    calibrated = (model.calibration_intercept + model.calibration_slope * raw).clamp(0, 1)
    return calibrated, raw


def score_rows(scores, start, end):
    return np.repeat(scores, end-start, axis=0) if len(scores) == 1 else scores[start:end]


@torch.no_grad()
def ensure_reward_scores(split, data, path, meta, cat, dev):
    model = load_reward(path, dev, cat, meta)
    fingerprint = {'checkpoint': sha(path), 'data': data_hash(split),
                   'catalog': meta['files']['catalog.npz'], 'code': evaluation_dependencies()}
    prefix = canonical_hash(fingerprint)
    def compute():
        n = 1 if model.variant == 'item_duration' else len(data['uid'])
        out = np.zeros((n, len(cat['duration'])), dtype=np.float32)
        for start in range(0, n, protocol()['evaluation']['batch']):
            end = min(n, start+protocol()['evaluation']['batch'])
            context = model.encode(data['history'][start:end], dev)
            scored, _ = all_scores(model, context)
            out[start:end] = scored.cpu().numpy()
        return out
    path_cache = OUTPUT / 'cache' / f'scores_{prefix}.npy'
    cold = not path_cache.exists()
    started = time.monotonic()
    values = cache_array(path_cache, fingerprint, compute)
    log_access(f'{split} full reward scores', 'Full-catalog frozen model scoring',
               checkpoint_sha256=fingerprint['checkpoint'], cold_cache=cold,
               seconds=time.monotonic()-started)
    return model, values


@torch.no_grad()
def actual_predictions(model, data, scores, dev):
    raw = np.zeros(len(data['uid']), dtype=np.float32)
    calibrated = np.zeros_like(raw)
    for start in range(0, len(raw), 128):
        end = min(start+128, len(raw))
        context = model.encode(data['history'][start:end], dev)
        items = torch.as_tensor(data['target'][start:end], device=dev, dtype=torch.long)
        raw[start:end] = model.raw_score(context, items).cpu().numpy()
        block = score_rows(scores, start, end)
        calibrated[start:end] = block[np.arange(end-start), data['target'][start:end]]
    return calibrated, raw


def descriptor(name, path=None, seed=11, kind='actor', reward_type='ratio', beta=0., weight=None):
    result = {'name': name, 'kind': kind, 'seed': seed, 'reward_type': reward_type, 'beta': beta}
    if path is not None:
        result['actor'] = str(path.relative_to(PROJECT))
    if weight is not None:
        result['weight'] = weight
    return result


@torch.no_grad()
def evaluate_one(spec, data, cat, meta, scores, auditor_scores, dev):
    ref = load_actor(actor_path(spec['seed']), dev)
    actor = load_actor(PROJECT / spec['actor'], dev) if spec['kind'] == 'actor' else None
    top_chunks, ratio_chunks = [], []
    quantities = {name: [] for name in ['policy_expected_ratio', 'policy_expected_time_reward',
                  'policy_expected_auditor_ratio', 'kl_to_reference', 'regularized_J']}
    cap = meta['duration_cap_seconds']
    duration = torch.as_tensor(np.minimum(cat['duration'], cap)/cap, device=dev)
    batch = protocol()['evaluation']['batch']
    for start in range(0, len(data['uid']), batch):
        end = min(start+batch, len(data['uid']))
        h, ts = data['history'][start:end], data['timestamp'][start:end]
        ref_lp = legal_logits(actor_logits(ref, h, dev), ts, cat, dev).log_softmax(-1)
        ratio = torch.as_tensor(score_rows(scores, start, end), device=dev)
        audit = torch.as_tensor(score_rows(auditor_scores, start, end), device=dev)
        reward = ratio if spec['reward_type'] == 'ratio' else ratio * duration[None, :]
        if spec['kind'] == 'actor':
            logits = legal_logits(actor_logits(actor, h, dev), ts, cat, dev)
        elif spec['kind'] == 'analytic':
            logits = ref_lp + reward / spec['beta']
        elif spec['kind'] == 'rerank':
            candidates = ref_lp.topk(protocol()['controls']['rerank_candidates'], -1).indices
            supported = torch.zeros_like(ref_lp, dtype=torch.bool).scatter_(1, candidates, True)
            logits = (ref_lp + spec['weight']*reward).masked_fill(~supported, -1e9)
        elif spec['kind'] == 'popularity':
            counts = torch.as_tensor(cat['popularity'], device=dev)
            logits = legal_logits(counts.clamp(min=1).log()[None, :].expand(len(h), -1), ts, cat, dev)
        else:
            raise ValueError(spec['kind'])
        lp = logits.log_softmax(-1)
        p = lp.exp()
        # Stable token-order tie handling is shared by every policy.
        top = np.argsort(-logits.cpu().numpy(), axis=1, kind='stable')[:, :20]
        top_chunks.append(top)
        ratio_chunks.append(score_rows(scores, start, end)[np.arange(end-start)[:, None], top])
        expected = (p * reward).sum(-1)
        kl = (p * (lp-ref_lp)).sum(-1)
        values = [(p*ratio).sum(-1), (p*ratio*duration[None,:]).sum(-1),
                  (p*audit).sum(-1), kl, expected-spec['beta']*kl]
        for name, value in zip(quantities, values):
            quantities[name].append(value.cpu().numpy())
    top = np.concatenate(top_chunks)
    predicted_ratio = np.concatenate(ratio_chunks)
    per_event = {name: np.concatenate(chunks) for name, chunks in quantities.items()}
    metrics, ranking_values = relevance(top, data, cat)
    per_event.update(ranking_values)
    metrics.update(behavior(top, predicted_ratio, data, cat, cap))
    metrics.update({name: macro(value, data['uid']) for name, value in per_event.items()
                    if name not in ranking_values})
    metrics['policy_expected_duration_limited_seconds'] = metrics['policy_expected_time_reward']*cap
    metrics['descriptor'] = spec
    metrics['J_definition'] = 'E[selected reward] - descriptor.beta * KL; compare only matching reward and beta'
    return metrics, {'top': top, 'uid': data['uid'], **per_event}


def evaluate(split, specs, reward_path, auditor_path, name):
    if split == 'holdout':
        frozen = check_frozen()
        if specs != frozen['descriptors']:
            raise ValueError('Окончательные политики не совпадают с фиксацией')
        if str(reward_path.relative_to(PROJECT)) != frozen['reward_path'] or str(auditor_path.relative_to(PROJECT)) != frozen['auditor_path']:
            raise ValueError('Изменён окончательный оценщик')
    meta, cat, dev = data_manifest(), catalog(), device()
    data = load_data(split, f'policy evaluation {name}')
    fingerprint = {'split': split, 'data': data_hash(split), 'code': evaluation_dependencies(),
                   'reward': sha(reward_path), 'auditor': sha(auditor_path), 'descriptors': specs,
                   'actor_hashes': {s['actor']: sha(PROJECT/s['actor']) for s in specs if 'actor' in s},
                   'reference_hashes': {str(actor_path(s['seed']).relative_to(PROJECT)): sha(actor_path(s['seed'])) for s in specs}}
    directory = OUTPUT / 'evaluations' / name
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / 'manifest.json'
    if manifest_path.exists():
        transaction = read_json(manifest_path)
        if transaction['fingerprint'] != fingerprint:
            raise ValueError('Изменены входы незавершённой/завершённой оценки; нужен новый development job')
    else:
        transaction = {'fingerprint': fingerprint, 'completed_models': {}, 'completed': False}
        write_json(manifest_path, transaction)
    for model_name, expected in transaction['completed_models'].items():
        for filename, checksum in expected.items():
            if sha(directory / filename) != checksum:
                raise ValueError(f'Изменён результат завершённой модели {model_name}')
    if transaction['completed']:
        if sha(directory/'results.json') != transaction['results_sha256'] or sha(directory/'reward_predictions.npz') != transaction['reward_predictions_sha256']:
            raise ValueError('Изменены итоговые метрики или reward-предсказания')
        return read_json(directory / 'results.json')
    reward, scores = ensure_reward_scores(split, data, reward_path, meta, cat, dev)
    auditor, auditor_scores = ensure_reward_scores(split, data, auditor_path, meta, cat, dev)
    results_path = directory / 'results.json'
    results = read_json(results_path) if results_path.exists() else {'split': split, 'models': {},
                  'scope': meta['new_holdout_kind'] if split=='holdout' else 'development diagnostics',
                  'model_reward_is_not_user_response': True}
    if 'reward_quality' not in results:
        pred, raw = actual_predictions(reward, data, scores, dev)
        audit_pred, audit_raw = actual_predictions(auditor, data, auditor_scores, dev)
        baselines = load_npz(OUTPUT / 'reward_baselines.npz')
        results['reward_quality'] = reward_quality(pred, raw, data, cat, baselines)
        results['auditor_quality'] = reward_quality(audit_pred, audit_raw, data, cat, baselines)
        save_npz(directory / 'reward_predictions.npz', prediction=pred, raw_prediction=raw,
                 auditor_prediction=audit_pred, auditor_raw_prediction=audit_raw, uid=data['uid'])
        write_json(results_path, results)
    for spec in specs:
        model_name = spec['name']
        if model_name in transaction['completed_models']:
            continue
        sync(dev)
        started = time.monotonic()
        metrics, predictions = evaluate_one(spec, data, cat, meta, scores, auditor_scores, dev)
        sync(dev)
        metrics['inference_and_metrics_seconds'] = time.monotonic()-started
        save_npz(directory / f'{model_name}.npz', **predictions)
        write_json(directory / f'{model_name}.json', metrics)
        results['models'][model_name] = metrics
        write_json(results_path, results)
        transaction['completed_models'][model_name] = {
            f'{model_name}.npz': sha(directory/f'{model_name}.npz'),
            f'{model_name}.json': sha(directory/f'{model_name}.json')}
        write_json(manifest_path, transaction)
        print(name, model_name, {key:metrics[key] for key in
              ['ndcg@10', 'policy_expected_ratio', 'policy_expected_time_reward', 'kl_to_reference']}, flush=True)
    for spec in specs:
        model_name = spec['name']
        base_name = f'baseline_s{spec["seed"]}'
        if base_name not in results['models']:
            continue
        p = load_npz(directory/f'{model_name}.npz')
        r = load_npz(directory/f'{base_name}.npz')
        comparisons = {metric: paired_interval(p[metric], r[metric], data['uid']) for metric in
                       ['ndcg@10', 'recall@10', 'policy_expected_ratio', 'policy_expected_time_reward']}
        comparisons['noninferiority_ndcg10'] = paired_interval(p['ndcg@10'], r['ndcg@10'], data['uid'],
                                                               factor=.95, one_sided=True)
        results['models'][model_name]['paired_vs_own_baseline'] = comparisons
    results['completed'] = True
    write_json(results_path, results)
    transaction['completed'] = True
    transaction['results_sha256'] = sha(results_path)
    transaction['reward_predictions_sha256'] = sha(directory/'reward_predictions.npz')
    write_json(manifest_path, transaction)
    return results
