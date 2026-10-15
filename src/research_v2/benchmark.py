"""Проверка эквивалентности оптимизации и технические замеры на development."""
import copy
import statistics
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .core import (CalibratedReward, OUTPUT, capture_rng, catalog, data_manifest,
                   device, inputs, load_actor, load_data, protocol, restore_rng,
                   seed_all, sync, write_json)
from .train import actor_path, policy_loss


def main():
    rng = seed_all(71)
    dev, meta, cat = device(), data_manifest(), catalog()
    data = load_data('train', 'technical benchmark; no heldout outcomes')
    initial = load_actor(actor_path(11), dev)
    ref = copy.deepcopy(initial).requires_grad_(False)
    reward = CalibratedReward(copy.deepcopy(initial), cat, meta['duration_cap_seconds'],
                              'history_duration').to(dev).eval().requires_grad_(False)
    spec = {key: protocol()['rl'][key] for key in ['samples', 'gamma']}
    spec.update(beta=.1, horizon=1, reward_type='ratio', repeat_penalty=.1)
    idx = rng.integers(0, len(data['uid']), protocol()['rl']['batch'])
    h, ts = data['history'][idx], data['timestamp'][idx]
    saved_rng = capture_rng(rng)
    gradients, losses = [], []
    for optimized in [False, True]:
        actor = copy.deepcopy(initial).eval()
        restore_rng(saved_rng, rng)
        loss, _ = policy_loss(actor, ref, reward, h, ts, cat, meta['duration_cap_seconds'],
                               spec, dev, optimized=optimized)
        loss.backward()
        gradients.append({name: p.grad.detach().cpu().clone() for name, p in actor.named_parameters()})
        losses.append(float(loss.detach().cpu()))
        del actor
    gradient_error = max(float((gradients[0][k]-gradients[1][k]).abs().max()) for k in gradients[0])
    loss_error = abs(losses[0]-losses[1])
    if loss_error > 2e-5 or gradient_error > 2e-5:
        raise ValueError(f'Оптимизация изменила результат: loss {loss_error}, gradient {gradient_error}')
    del gradients
    measured = {}
    for optimized in [False, True]:
        actor = copy.deepcopy(initial).eval()
        opt = torch.optim.AdamW(actor.parameters(), lr=protocol()['rl']['lr'],
                               weight_decay=protocol()['rl']['weight_decay'])
        times = []
        for step in range(20):
            sync(dev)
            started = time.monotonic()
            loss, _ = policy_loss(actor, ref, reward, h, ts, cat, meta['duration_cap_seconds'],
                                   spec, dev, optimized=optimized)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.)
            opt.step()
            sync(dev)
            if step >= 5:
                times.append(time.monotonic()-started)
        measured['shared_history' if optimized else 'repeated_history'] = {
            'median_seconds': statistics.median(times), 'min_seconds': min(times),
            'max_seconds': max(times), 'measured_steps': len(times), 'warmup_steps': 5}
        del actor, opt
    actor = copy.deepcopy(initial).train()
    opt = torch.optim.AdamW(actor.parameters(), lr=protocol()['controls']['sft']['lr'],
                           weight_decay=protocol()['rl']['weight_decay'])
    times = []
    for step in range(25):
        idx = rng.integers(0, len(data['uid']), protocol()['controls']['sft']['batch'])
        x = data['history'][idx]
        labels = torch.as_tensor(np.concatenate([x[:, 1:], data['target'][idx, None]], -1), device=dev, dtype=torch.long)
        labels = labels.masked_fill((labels < 2) | torch.as_tensor(x, device=dev).eq(0) |
                                   torch.as_tensor(data['loss_mask'][idx], device=dev).eq(0), -100)
        sync(dev)
        started = time.monotonic()
        logits = actor(**inputs(x, dev)).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.)
        opt.step()
        sync(dev)
        if step >= 5:
            times.append(time.monotonic()-started)
    measured['sft'] = {'median_seconds': statistics.median(times), 'min_seconds': min(times),
                       'max_seconds': max(times), 'measured_steps': len(times), 'warmup_steps': 5}
    result = {'device': str(dev), 'torch': torch.__version__, 'threads': torch.get_num_threads(),
              'equivalence': {'loss_abs_error': loss_error, 'gradient_max_error': gradient_error,
                              'tolerance': 2e-5, 'status': 'passed'},
              'timings': measured,
              'rl_speedup': measured['repeated_history']['median_seconds']/measured['shared_history']['median_seconds'],
              'interpretation': 'Technical warm-cache measurement on fixed train contexts. Reward head randomly initialized for equivalence and cost checks; not an experimental policy result.',
              'heldout_outcomes_read': False}
    if dev.type == 'mps':
        result['mps_current_allocated_bytes'] = torch.mps.current_allocated_memory()
        result['mps_driver_allocated_bytes'] = torch.mps.driver_allocated_memory()
    write_json(OUTPUT / 'benchmark.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
