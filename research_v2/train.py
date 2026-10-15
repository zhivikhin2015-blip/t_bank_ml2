"""Обучение с сохранением RNG, лучшей модели, истории и накопленных затрат."""
import copy
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .core import (CalibratedReward, ORIGINAL, OUTPUT, PROJECT, SOURCE, UserSampler,
                   actor_logits, affine_calibration, cache_array, canonical_hash,
                   capture_rng, catalog, data_manifest, device, inputs, legal_logits,
                   load_actor, load_data, load_reward, macro, protocol, read_checkpoint,
                   read_json, restore_rng, save_torch, seed_all, sha, sync, write_json)


def dependencies():
    return {str(p.relative_to(PROJECT)): sha(p) for p in
            [SOURCE / 'core.py', SOURCE / 'train.py', SOURCE / 'protocol.json', PROJECT / 'experiment.py']}


def actor_path(seed):
    path = ORIGINAL / f'baseline_s{seed}' / 'best.pt'
    expected = read_json(OUTPUT / 'original_snapshot' / 'manifest.json')['files']
    if sha(path) != expected[str(path.relative_to(PROJECT))]['sha256']:
        raise ValueError('Исходный actor изменён')
    return path


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


class TrainingSession:
    """last.pt — атомарный источник истории и времени, включая лучшую reward model."""
    def __init__(self, name, model, optimizer, rng, config, dev):
        self.path = OUTPUT / 'models' / name
        self.path.mkdir(parents=True, exist_ok=True)
        self.model, self.optimizer, self.rng, self.config, self.dev = model, optimizer, rng, config, dev
        self.identity = canonical_hash(config)
        self.step = 0
        self.training_seconds = self.evaluation_seconds = self.checkpoint_seconds = 0.
        self.best_mse = float('inf')
        self.best_model = None
        self.best_step = None
        self.history = []
        last = self.path / 'last.pt'
        if last.exists():
            state = read_checkpoint(last)
            if state['identity'] != self.identity:
                raise ValueError(f'Нельзя продолжить {name}: изменены данные, код или настройки')
            if state['device'] != str(dev):
                raise ValueError(f'Продолжение {name} требует исходного устройства {state["device"]}')
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            restore_rng(state['rng'], rng)
            for field in ['step', 'training_seconds', 'evaluation_seconds', 'checkpoint_seconds',
                          'best_mse', 'best_model', 'best_step', 'history']:
                setattr(self, field, state[field])
            timing_path = self.path / 'timing.json'
            if timing_path.exists():
                timing = read_json(timing_path)
                if timing['step'] == self.step and timing['identity'] == self.identity:
                    self.checkpoint_seconds = max(self.checkpoint_seconds, timing['checkpoint_seconds'])

    def update(self, loss_fn):
        sync(self.dev)
        started = time.monotonic()
        loss, details = loss_fn()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 1.)
        self.optimizer.step()
        sync(self.dev)
        self.training_seconds += time.monotonic() - started
        self.step += 1
        return float(loss.detach().cpu()), details

    def checkpoint(self, loss, details=None, mse=None, milestone=False):
        if mse is not None and mse < self.best_mse:
            self.best_mse, self.best_step = mse, self.step
            self.best_model = cpu_state(self.model)
        row = {'step': self.step, 'loss': loss, 'training_seconds': self.training_seconds,
               'evaluation_seconds': self.evaluation_seconds, 'checkpoint_seconds': self.checkpoint_seconds,
               **(details or {})}
        if mse is not None:
            row['validation_mse'] = mse
        self.history.append(row)
        payload = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                   'rng': capture_rng(self.rng), 'identity': self.identity, 'config': self.config,
                   'device': str(self.dev), 'step': self.step, 'training_seconds': self.training_seconds,
                   'evaluation_seconds': self.evaluation_seconds, 'checkpoint_seconds': self.checkpoint_seconds,
                   'best_mse': self.best_mse, 'best_model': self.best_model,
                   'best_step': self.best_step, 'history': self.history}
        started = time.monotonic()
        save_torch(self.path / 'last.pt', payload)
        self.checkpoint_seconds += time.monotonic() - started
        # Последнее сохранение также учитывается при продолжении.
        write_json(self.path / 'timing.json', {'training_seconds': self.training_seconds,
                   'evaluation_seconds': self.evaluation_seconds, 'checkpoint_seconds': self.checkpoint_seconds,
                   'step': self.step, 'identity': self.identity})
        write_json(self.path / 'history.json', self.history)
        if milestone:
            save_torch(self.path / 'stages' / f'step{self.step:05d}.pt',
                       {'model': self.model.state_dict(), 'config': self.config, 'step': self.step,
                        'training_seconds': self.training_seconds, 'identity': self.identity})
        print(self.path.name, row, flush=True)

    def finish(self, choose_best=False):
        if self.step != self.config['spec']['steps']:
            return None
        state = self.best_model if choose_best else self.model.state_dict()
        step = self.best_step if choose_best else self.step
        save_torch(self.path / 'best.pt', {'model': state, 'config': self.config, 'step': step,
                   'training_seconds': self.training_seconds, 'identity': self.identity})
        write_json(self.path / 'completed.json', {'steps': self.step, 'chosen_step': step,
                   'checkpoint_sha256': sha(self.path / 'best.pt'), 'identity': self.identity,
                   'training_seconds': self.training_seconds, 'evaluation_seconds': self.evaluation_seconds,
                   'checkpoint_seconds': self.checkpoint_seconds})
        return self.path / 'best.pt'


@torch.no_grad()
def encoded_context(backbone, data, data_hash, backbone_hash, dev):
    fingerprint = {'data': data_hash, 'backbone': backbone_hash,
                   'encoding_code': sha(SOURCE / 'core.py')}
    name = canonical_hash(fingerprint)
    def compute():
        chunks = []
        backbone.eval()
        for start in range(0, len(data['uid']), 128):
            chunks.append(backbone.transformer(**inputs(data['history'][start:start+128], dev))
                          .last_hidden_state[:, -1].cpu().numpy())
        return np.concatenate(chunks)
    started = time.monotonic()
    previously_cached = (OUTPUT / 'cache' / f'context_{name}.npy').exists()
    value = cache_array(OUTPUT / 'cache' / f'context_{name}.npy', fingerprint, compute)
    timing_path = OUTPUT / 'cache' / f'context_{name}.timing.json'
    timing = read_json(timing_path) if timing_path.exists() else {'access_seconds': [], 'compute_seconds': 0.}
    elapsed = time.monotonic() - started
    timing['access_seconds'].append(elapsed)
    if not previously_cached:
        timing['compute_seconds'] += elapsed
    write_json(timing_path, {**timing, 'fingerprint': fingerprint})
    return value


@torch.no_grad()
def reward_predictions(model, context, data, dev, raw=False):
    prediction = []
    for start in range(0, len(data['target']), 512):
        c = torch.as_tensor(context[start:start+512], device=dev)
        item = torch.as_tensor(data['target'][start:start+512], device=dev, dtype=torch.long)
        fn = model.raw_score if raw else model.score
        prediction.append(fn(c, item).cpu().numpy())
    return np.concatenate(prediction)


def _config(stage, model, spec, **details):
    return {'stage': stage, 'gpt2': model.backbone.config.to_dict() if stage == 'reward' else model.config.to_dict(),
            'spec': spec, 'dependencies': dependencies(),
            'data_manifest_sha256': sha(OUTPUT / 'data_manifest.json'), **details}


def train_reward(variant, seed, limit_steps=None):
    cfg = protocol()['reward']
    spec = {key: cfg[key] for key in ['steps', 'batch', 'lr', 'weight_decay', 'save_every', 'eval_every']}
    spec.update(seed=seed, variant=variant)
    rng = seed_all(seed)
    dev, meta, cat = device(), data_manifest(), catalog()
    data = load_data('train', f'reward training {variant} seed {seed}')
    val = load_data('reward_select', f'reward model/checkpoint selection {variant} seed {seed}')
    parent = actor_path(seed)
    backbone = load_actor(parent, dev)
    model = CalibratedReward(backbone, cat, meta['duration_cap_seconds'], variant).to(dev)
    context = encoded_context(backbone, data, meta['files']['train.npz'], sha(parent), dev)
    val_context = encoded_context(backbone, val, meta['files']['reward_select.npz'], sha(parent), dev)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=spec['lr'], weight_decay=spec['weight_decay'])
    name = f'reward_{variant}_s{seed}'
    session = TrainingSession(name, model, optimizer, rng,
                              _config('reward', model, spec, variant=variant, parent_sha256=sha(parent)), dev)
    sampler = UserSampler(data['uid'])
    def loss_fn():
        idx = sampler.sample(rng, spec['batch'])
        c = torch.as_tensor(context[idx], device=dev)
        item = torch.as_tensor(data['target'][idx], device=dev, dtype=torch.long)
        target = torch.as_tensor(data['ratio'][idx], device=dev)
        return F.mse_loss(model.raw_score(c, item), target), {}
    target_steps = spec['steps'] if limit_steps is None else min(spec['steps'], limit_steps)
    while session.step < target_steps:
        loss, details = session.update(loss_fn)
        if session.step % spec['save_every'] == 0 or session.step == target_steps:
            mse = None
            if session.step % spec['eval_every'] == 0 or session.step == spec['steps']:
                started = time.monotonic()
                pred = reward_predictions(model, val_context, val, dev, raw=True)
                mse = macro((pred - val['ratio']) ** 2, val['uid'])
                details['validation_mae'] = macro(np.abs(pred - val['ratio']), val['uid'])
                session.evaluation_seconds += time.monotonic() - started
            session.checkpoint(loss, details, mse=mse)
    return session.finish(choose_best=True)


def calibrate_reward(name):
    dev, meta, cat = device(), data_manifest(), catalog()
    path = OUTPUT / 'models' / name / 'best.pt'
    output = path.with_name('calibrated.pt')
    if output.exists():
        manifest = read_json(path.with_name('calibration.json'))
        if sha(path) != manifest['raw_checkpoint_sha256'] or sha(output) != manifest['calibrated_checkpoint_sha256']:
            raise ValueError('Калибровка не соответствует исходной модели')
        return output
    model = load_reward(path, dev, cat, meta)
    data = load_data('reward_calibrate', f'affine calibration of {name}; no C fitting')
    # Исходный encoder неизменен при обучении reward model.
    parent = actor_path(read_checkpoint(path)['config']['spec']['seed'])
    context = encoded_context(model.backbone, data, meta['files']['reward_calibrate.npz'], sha(parent), dev)
    pred = reward_predictions(model, context, data, dev, raw=True)
    fit = affine_calibration(pred, data['ratio'], data['uid'])
    model.calibration_slope.fill_(fit['slope'])
    model.calibration_intercept.fill_(fit['intercept'])
    state = read_checkpoint(path)
    state['model'] = model.state_dict()
    save_torch(output, state)
    write_json(path.with_name('calibration.json'), {**fit, 'fit_split': 'B2',
               'raw_checkpoint_sha256': sha(path), 'calibrated_checkpoint_sha256': sha(output),
               'fit_examples': len(pred), 'fit_users': len(np.unique(data['uid'])),
               'fit_mse_before': macro((pred-data['ratio'])**2, data['uid']),
               'fit_mse_after': macro((np.clip(fit['intercept']+fit['slope']*pred,0,1)-data['ratio'])**2,data['uid'])})
    return output


def policy_loss(actor, ref, reward, histories, timestamps, cat, cap, spec, dev, optimized=True):
    """Точное вычисление KL; score-function градиент для наград и будущих KL."""
    batch, samples = len(histories), spec['samples']
    h = np.repeat(histories, samples, axis=0)
    ts = np.repeat(timestamps, samples)
    rewards, logps, kls = [], [], []
    duration = torch.as_tensor(np.minimum(cat['duration'], cap)/cap, device=dev)
    for t in range(spec['horizon']):
        shared = optimized and t == 0
        current_h, current_ts = (histories, timestamps) if shared else (h, ts)
        lp = legal_logits(actor_logits(actor, current_h, dev), current_ts, cat, dev).log_softmax(-1)
        with torch.no_grad():
            ref_lp = legal_logits(actor_logits(ref, current_h, dev), current_ts, cat, dev).log_softmax(-1)
            context = reward.encode(current_h, dev)
        kl = (lp.exp() * (lp - ref_lp)).sum(-1)
        if shared:
            lp = lp.repeat_interleave(samples, dim=0)
            context = context.repeat_interleave(samples, dim=0)
            kl = kl.repeat_interleave(samples, dim=0)
        with torch.no_grad():
            action = torch.multinomial(lp.exp(), 1).squeeze(-1)
            r = reward.score(context, action)
            if spec['reward_type'] in ['time', 'balanced']:
                r = r * duration[action]
            if spec['reward_type'] == 'balanced' and t:
                previous = torch.as_tensor(h[:, -t:], device=dev)
                r = r - spec['repeat_penalty'] * previous.eq(action[:, None]).any(-1).float()
        rewards.append(r)
        logps.append(lp.gather(1, action[:, None]).squeeze(-1))
        kls.append(kl)
        h = np.concatenate([h[:, 1:], action.cpu().numpy()[:, None]], -1)
    return_to_go = torch.zeros_like(rewards[-1])
    pg, direct_kl = 0., 0.
    for t in reversed(range(spec['horizon'])):
        return_to_go = rewards[t] - spec['beta']*kls[t].detach() + spec['gamma']*return_to_go
        grouped = return_to_go.reshape(batch, samples)
        baseline = (grouped.sum(-1, keepdim=True)-grouped)/(samples-1)
        advantage = (grouped-baseline).reshape(-1).detach()
        pg = pg - spec['gamma']**t * (advantage*logps[t]).mean()
        direct_kl = direct_kl + spec['gamma']**t * kls[t].mean()
    loss = (pg + spec['beta']*direct_kl)/spec['horizon']
    return loss, {'sample_reward': float(torch.stack(rewards).mean().cpu()),
                  'discounted_kl_per_step': float((direct_kl/spec['horizon']).detach().cpu())}


def train_rl(seed, reward_type, beta, horizon, reward_path, limit_steps=None):
    cfg = protocol()['rl']
    spec = {key: cfg[key] for key in ['steps', 'batch', 'samples', 'lr', 'weight_decay', 'save_every', 'gamma']}
    if horizon == 5:
        spec['steps'] = cfg['h5']['steps_full']
    spec.update(seed=seed, reward_type=reward_type, beta=beta, horizon=horizon,
                repeat_penalty=cfg['h5']['repeat_penalty'])
    rng = seed_all(seed)
    dev, meta, cat = device(), data_manifest(), catalog()
    data = load_data('train', f'RL {reward_type} beta {beta} H {horizon} seed {seed}')
    parent = actor_path(seed)
    actor, ref = load_actor(parent, dev), load_actor(parent, dev)
    ref.requires_grad_(False)
    reward = load_reward(reward_path, dev, cat, meta).requires_grad_(False)
    actor.eval()
    optimizer = torch.optim.AdamW(actor.parameters(), lr=spec['lr'], weight_decay=spec['weight_decay'])
    name = f'rl_{reward_type}_b{beta:g}_h{horizon}_s{seed}'
    session = TrainingSession(name, actor, optimizer, rng,
              _config('rl', actor, spec, parent_sha256=sha(parent), reward_sha256=sha(reward_path)), dev)
    sampler = UserSampler(data['uid'])
    def loss_fn():
        idx = sampler.sample(rng, spec['batch'])
        return policy_loss(actor, ref, reward, data['history'][idx], data['timestamp'][idx],
                           cat, meta['duration_cap_seconds'], spec, dev)
    target_steps = spec['steps'] if limit_steps is None else min(spec['steps'], limit_steps)
    while session.step < target_steps:
        loss, details = session.update(loss_fn)
        if session.step % spec['save_every'] == 0 or session.step == target_steps:
            session.checkpoint(loss, details, milestone=horizon == 5)
    return session.finish()


def train_sft(seed, steps, limit_steps=None):
    cfg = protocol()['controls']['sft']
    spec = {'seed': seed, 'steps': steps, 'batch': cfg['batch'], 'lr': cfg['lr'],
            'save_every': cfg['save_every'], 'weight_decay': protocol()['rl']['weight_decay']}
    rng = seed_all(seed)
    dev = device()
    data = load_data('train', f'SFT time control seed {seed}')
    parent = actor_path(seed)
    model = load_actor(parent, dev)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec['lr'], weight_decay=spec['weight_decay'])
    session = TrainingSession(f'sft_s{seed}', model, optimizer, rng,
                              _config('sft', model, spec, parent_sha256=sha(parent)), dev)
    sampler = UserSampler(data['uid'])
    def loss_fn():
        idx = sampler.sample(rng, spec['batch'])
        x = data['history'][idx]
        labels = np.concatenate([x[:, 1:], data['target'][idx, None]], -1)
        labels = torch.as_tensor(labels, device=dev, dtype=torch.long)
        labels = labels.masked_fill((labels < 2) | torch.as_tensor(x, device=dev).eq(0) |
                                   torch.as_tensor(data['loss_mask'][idx], device=dev).eq(0), -100)
        logits = model(**inputs(x, dev)).logits
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100), {}
    target_steps = steps if limit_steps is None else min(steps, limit_steps)
    while session.step < target_steps:
        loss, details = session.update(loss_fn)
        if session.step % spec['save_every'] == 0 or session.step == target_steps:
            session.checkpoint(loss, details)
    return session.finish()
