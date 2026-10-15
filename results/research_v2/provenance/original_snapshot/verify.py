"""Независимые проверки причинной маски, метрик и восстановления обучения."""
import tempfile
from pathlib import Path

import numpy as np
import torch
import pyarrow.dataset as ds
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

from evaluate import ranking_from_top
from experiment import (OUT, device, inputs, legal_logits, load_data, metadata,
                        model_config, restore, save_checkpoint, seed_all, write_json)


def main():
    rng = seed_all(17)
    meta, cat = metadata()
    dev = device()
    data = load_data('train')
    # Независимая сверка сохранённой маски с исходными timestamp-группами.
    chosen = rng.choice(len(data['target']), 8, replace=False)
    raw = ds.dataset(OUT/'listens.parquet').to_table(filter=ds.field('uid').isin(data['uid'][chosen].tolist()))
    raw_arrays = {k:raw[k].to_numpy() for k in ['uid','timestamp','item_id']}
    raw_order = np.lexsort((raw_arrays['item_id'],raw_arrays['timestamp'],raw_arrays['uid']))
    raw_arrays = {k:v[raw_order] for k,v in raw_arrays.items()}
    for row in chosen:
        timestamps = raw_arrays['timestamp'][raw_arrays['uid'] == data['uid'][row]]
        target_pos = np.flatnonzero(timestamps == data['timestamp'][row])
        assert len(target_pos) == 1
        j = int(target_pos[0])
        begin = max(0,j-meta['history_length'])
        times, counts = np.unique(timestamps,return_counts=True)
        lookup = dict(zip(times.tolist(),counts.tolist()))
        expected = np.zeros(meta['history_length'],dtype=np.int8)
        expected[-(j-begin):] = [int(lookup[int(t)] == 1) for t in timestamps[begin+1:j+1]]
        assert np.array_equal(expected,data['loss_mask'][row]), 'Внутренняя маска не соответствует исходному времени'
    model = GPT2LMHeadModel(model_config(meta)).to(dev).eval()
    hist = data['history'][:2].copy()
    changed = hist.copy()
    changed[:, -1] = 2 + (changed[:, -1] % meta['catalog_items'])
    with torch.no_grad():
        original = model(**inputs(hist, dev)).logits[:, :-1]
        perturbed = model(**inputs(changed, dev)).logits[:, :-1]
    causal_error = float((original-perturbed).abs().max().cpu())
    assert causal_error < 1e-5, 'Будущий токен изменил прошлые предсказания'
    # Ручной пример: пользователю 10 принадлежат два примера (попадание на
    # местах 1 и 2), пользователю 20 — один промах. Вклад пользователей равен.
    top = np.tile(np.arange(2, 22), (3, 1))
    manual = {'target': np.array([2, 3, 99]), 'uid': np.array([10, 10, 20])}
    metrics, _ = ranking_from_top(top, manual)
    expected_ndcg = ((1 + 1/np.log2(3))/2)/2
    assert abs(metrics['recall@10']-.5) < 1e-10
    assert abs(metrics['ndcg@10']-expected_ndcg) < 1e-10
    perm = [2, 0, 1]
    permuted, _ = ranking_from_top(top[perm], {k:v[perm] for k,v in manual.items()})
    assert abs(permuted['ndcg@10']-expected_ndcg) < 1e-10
    fake_firstseen = dict(cat)
    fake_firstseen['first_seen'] = cat['first_seen'].copy()
    fake_firstseen['first_seen'][2] = meta['timestamp_max'] + 1
    masked = legal_logits(torch.zeros(2, meta['catalog_items']+2, device=dev),
                          data['timestamp'][-2:], fake_firstseen, dev)
    assert bool((masked[:, 2] < -1e8).all())
    # Проверяем реальное сохранение оптимизатора и обоих RNG на MPS/CPU.
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    def update(m, o, gen):
        idx = gen.integers(0, len(data['target']), 2)
        logits = m(**inputs(data['history'][idx], dev)).logits[:, -1]
        y = torch.as_tensor(data['target'][idx], device=dev, dtype=torch.long)
        loss = F.cross_entropy(logits, y)
        o.zero_grad(set_to_none=True)
        loss.backward()
        o.step()
        return idx
    model.train()
    update(model, opt, rng)
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        path = Path(tmp)/'resume.pt'
        save_checkpoint(path, model, opt, 1, rng, {'gpt2':model.config.to_dict()})
        expected_idx = update(model, opt, rng)
        expected = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        other = GPT2LMHeadModel(model_config(meta)).to(dev).train()
        other_opt = torch.optim.AdamW(other.parameters(), lr=.001)
        other_rng = np.random.default_rng(999)
        step = restore(path, other, other_opt, other_rng)
        actual_idx = update(other, other_opt, other_rng)
        assert step == 1 and np.array_equal(expected_idx, actual_idx)
        resume_error = max(float((expected[k]-v.detach().cpu()).abs().max()) for k,v in other.state_dict().items())
        assert resume_error < 1e-5, f'Ошибка восстановления: {resume_error}'
    cuts = meta['temporal_cuts']
    splits = meta['splits']
    assert splits['train']['timestamp_max'] < cuts[0] <= splits['reward_val']['timestamp_min']
    assert splits['reward_val']['timestamp_max'] < cuts[1] <= splits['policy_val']['timestamp_min']
    assert splits['policy_val']['timestamp_max'] < cuts[2] <= splits['test']['timestamp_min']
    result = {'device': str(dev), 'causal_mask_max_error': causal_error,
              'resume_max_parameter_error': resume_error, 'manual_macro_metrics': metrics,
              'future_item_mask': 'passed', 'temporal_boundaries': 'passed',
              'internal_tie_loss_mask_checked_against_raw': 8,
              'test_outcomes_used': False}
    write_json(OUT/'verification.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
