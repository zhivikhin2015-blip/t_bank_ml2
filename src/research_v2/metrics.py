"""Единые знаменатели, оценки отклика и диагностические срезы."""
import numpy as np

from .core import macro, paired_interval, user_weights


def errors(pred, target, uid):
    if not len(uid):
        return None
    residual = pred - target
    mse = macro(residual**2, uid)
    return {'examples': len(uid), 'users': len(np.unique(uid)),
            'mae': macro(np.abs(residual), uid), 'mse': mse, 'rmse': float(np.sqrt(mse)),
            'predicted_mean': macro(pred, uid), 'observed_mean': macro(target, uid)}


def calibration(pred, target, uid):
    weights = user_weights(uid)
    bins = np.minimum((np.clip(pred, 0, 1) * 10).astype(int), 9)
    rows = []
    error = 0.
    for index in range(10):
        mask = bins == index
        if not mask.any():
            continue
        mass = float(weights[mask].sum())
        predicted = float(np.dot(weights[mask], pred[mask]) / mass)
        observed = float(np.dot(weights[mask], target[mask]) / mass)
        error += mass * abs(predicted - observed)
        rows.append({'lower': index / 10, 'upper': (index + 1) / 10,
                     'examples': int(mask.sum()), 'users': len(np.unique(uid[mask])),
                     'weight_mass': mass, 'predicted_mean': predicted, 'observed_mean': observed})
    return {'bins': rows, 'weighted_mean_response_calibration_error': error,
            'weight_definition': 'Full known-target population: 1/(number_users * events_of_user); kept fixed across bins'}


def slice_masks(data, cat):
    known = data['target'] >= 2
    duration = np.zeros(len(known))
    popularity = np.zeros(len(known))
    duration[known] = cat['duration'][data['target'][known]]
    popularity[known] = cat['popularity'][data['target'][known]]
    return {'known': known, 'positive_ratio_gt50': data['ratio'] > .5,
            'nonpositive_ratio_le50': data['ratio'] <= .5,
            'organic': data['organic'] == 1, 'nonorganic': data['organic'] == 0,
            'short': known & (duration <= 180), 'long': known & (duration > 300),
            'popular': known & (popularity >= np.quantile(cat['popularity'][2:], .9)),
            'rare': known & (popularity <= np.quantile(cat['popularity'][2:], .5))}


def reward_quality(pred, raw_pred, data, cat, baselines):
    good = data['target'] >= 2
    y, uid = data['ratio'][good], data['uid'][good]
    p, raw = pred[good], raw_pred[good]
    global_pred = np.full_like(y, float(baselines['global_mean']))
    item_pred = baselines['item_mean'][data['target'][good]]
    result = {'calibrated': errors(p, y, uid), 'raw': errors(raw, y, uid),
              'global_mean': errors(global_pred, y, uid), 'item_mean': errors(item_pred, y, uid),
              'calibration': calibration(p, y, uid), 'raw_calibration': calibration(raw, y, uid),
              'mse_difference_vs_global': paired_interval((p-y)**2, (global_pred-y)**2, uid),
              'slices': {}}
    for name, mask in slice_masks(data, cat).items():
        selected = mask[good]
        result['slices'][name] = errors(p[selected], y[selected], uid[selected])
    for threshold in [.8, .9]:
        selected = p >= threshold
        if selected.any():
            w = user_weights(uid)[selected]
            mass = w.sum()
            result[f'tail_ge{threshold:g}'] = {'examples': int(selected.sum()),
                    'users': len(np.unique(uid[selected])), 'weight_mass': float(mass),
                    'predicted_mean': float(np.dot(w, p[selected])/mass),
                    'observed_mean': float(np.dot(w, y[selected])/mass)}
        else:
            result[f'tail_ge{threshold:g}'] = None
    return result


def relevance(top, data, cat):
    matched = top == data['target'][:, None]
    per_event = {}
    for k in [10, 20]:
        per_event[f'recall@{k}'] = matched[:, :k].any(-1).astype(float)
        per_event[f'ndcg@{k}'] = (matched[:, :k] / np.log2(np.arange(k)+2)).sum(-1)
    metrics = {name: macro(value, data['uid']) for name, value in per_event.items()}
    metrics.update(examples=len(top), users=len(np.unique(data['uid'])), slices={})
    for name, selected in slice_masks(data, cat).items():
        metrics['slices'][name] = ({'examples': int(selected.sum()),
            'users': len(np.unique(data['uid'][selected])),
            **{metric: macro(value[selected], data['uid'][selected]) for metric, value in per_event.items()}}
            if selected.any() else None)
    metrics['oracle_catalog_recall_ceiling'] = macro((data['target'] >= 2).astype(float), data['uid'])
    return metrics, per_event


def behavior(top, predicted_ratio, data, cat, cap):
    uid = data['uid']
    ids = top[:, :10]
    duration = cat['duration'][ids]
    ratio = predicted_ratio[:, :10]
    counts = np.bincount(ids.reshape(-1), minlength=len(cat['duration']))[2:]
    probability = counts[counts > 0] / counts.sum()
    seen = (ids[:, :, None] == data['history'][:, None, :]).any(-1).mean(-1)
    _, first = np.unique(uid, return_index=True)
    rng = np.random.default_rng(1234)
    pairs = rng.integers(0, len(first), size=(min(2000, len(first)*2), 2))
    overlap = [len(set(ids[first[a]]) & set(ids[first[b]])) / 10 for a, b in pairs if a != b]
    return {'predicted_ratio_at10': macro(ratio.mean(-1), uid),
            'duration_limited_seconds_at10': macro((ratio*np.minimum(duration,cap)).mean(-1), uid),
            'mean_duration_seconds_at10': macro(duration.mean(-1), uid),
            'mean_popularity_at10': macro(cat['popularity'][ids].mean(-1), uid),
            'seen_in_history_at10': macro(seen, uid),
            'catalog_coverage_at10': float(np.mean(counts>0)),
            'top10_items_exposure_share': float(np.sort(counts)[-10:].sum()/counts.sum()),
            'recommendation_entropy': float(-(probability*np.log(probability)).sum()),
            'between_user_overlap_at10': float(np.mean(overlap)) if overlap else None,
            'exposure_denominator': 'All event-list positions; activity weighted. Relevance/reward metrics are macro-user.'}
