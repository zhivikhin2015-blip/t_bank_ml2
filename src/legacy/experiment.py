"""Обучение, возобновление и оценка компактной GPT-2 на Yambda."""
import argparse
import copy
import json
import os
from pathlib import Path
import time

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).parent / '.tmp' / 'matplotlib'))
os.environ.setdefault('HF_HOME', str(Path(__file__).parent / '.tmp' / 'huggingface'))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2Config, GPT2LMHeadModel

ROOT = Path(__file__).resolve().parent
OUT = ROOT / os.environ.get('ANNAI_RL_OUTPUT', 'output')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def sync(dev):
    if dev.type == 'mps':
        torch.mps.synchronize()


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)
    return np.random.default_rng(seed)


def load_data(split):
    with np.load(OUT / f'{split}.npz') as f:
        return {k: f[k] for k in f.files}


def metadata():
    meta = json.loads((OUT / 'data_manifest.json').read_text())
    with np.load(OUT / 'catalog.npz') as f:
        cat = {k: f[k] for k in f.files}
    return meta, cat


def model_config(meta):
    return GPT2Config(vocab_size=meta['catalog_items'] + 2,
                      n_positions=meta['history_length'], n_embd=128, n_layer=2,
                      n_head=4, bos_token_id=1, eos_token_id=None, pad_token_id=0,
                      resid_pdrop=.1, embd_pdrop=.1, attn_pdrop=.1,
                      use_cache=False, attn_implementation='eager')


def inputs(hist, dev):
    x = torch.as_tensor(hist, dtype=torch.long, device=dev)
    mask = x.ne(0).long()
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return {'input_ids': x, 'attention_mask': mask, 'position_ids': pos, 'use_cache': False}


def actor_logits(model, hist, dev):
    return model(**inputs(hist, dev)).logits[:, -1, :]


def legal_logits(logits, timestamp, cat, dev):
    seen = torch.as_tensor(cat['first_seen'], device=dev)
    ts = torch.as_tensor(timestamp, device=dev)
    legal = seen.unsqueeze(0) < ts.unsqueeze(1)
    legal[:, :2] = False
    assert legal.any(-1).all(), 'Пустой каталог кандидатов'
    # Конечная константа предотвращает 0 * NaN в KL.
    return logits.masked_fill(~legal, -1e9)


def save_checkpoint(path, model, opt, step, rng, extra):
    payload = dict(model=model.state_dict(), optimizer=opt.state_dict(), step=step,
                   numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                   config=extra)
    if torch.backends.mps.is_available():
        payload['mps_rng'] = torch.mps.get_rng_state()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def read_checkpoint(path):
    # Только собственные checkpoints из этого эксперимента, без внешнего pickle.
    return torch.load(path, map_location='cpu', weights_only=True)


def restore(path, model, opt, rng):
    state = read_checkpoint(path)
    model.load_state_dict(state['model'])
    opt.load_state_dict(state['optimizer'])
    rng.bit_generator.state = state['numpy_rng']
    torch.set_rng_state(state['torch_rng'])
    if 'mps_rng' in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state['mps_rng'])
    return int(state['step'])


def load_actor(path, dev):
    state = read_checkpoint(path)
    cfg = GPT2Config(**state['config']['gpt2'])
    cfg._attn_implementation = 'eager'
    model = GPT2LMHeadModel(cfg).to(dev)
    model.load_state_dict(state['model'])
    model.eval()
    return model


class RewardNet(nn.Module):
    def __init__(self, backbone, cat, cap):
        super().__init__()
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.item = nn.Embedding.from_pretrained(backbone.transformer.wte.weight.detach().clone(), freeze=False)
        d = backbone.config.n_embd
        self.head = nn.Sequential(nn.Linear(d * 2 + 1, 128), nn.ReLU(),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.register_buffer('duration', torch.as_tensor(cat['duration'], dtype=torch.float32))
        self.cap = cap

    def encode(self, hist, dev):
        self.backbone.eval()
        with torch.no_grad():
            return self.backbone.transformer(**inputs(hist, dev)).last_hidden_state[:, -1, :]

    def score(self, context, item):
        d = torch.log1p(self.duration[item]) / np.log1p(self.cap)
        return self.head(torch.cat([context, self.item(item), d.unsqueeze(-1)], -1)).squeeze(-1).sigmoid()

    def forward(self, hist, item, dev):
        return self.score(self.encode(hist, dev), item)


def load_reward(path, dev, cat, meta):
    state = read_checkpoint(path)
    backbone = GPT2LMHeadModel(GPT2Config(**state['config']['gpt2'])).to(dev)
    backbone.config._attn_implementation = 'eager'
    reward = RewardNet(backbone, cat, meta['duration_cap_seconds']).to(dev)
    reward.load_state_dict(state['model'])
    reward.eval()
    return reward


def macro(values, uids):
    users, inverse = np.unique(uids, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    return float(np.mean(sums/counts))


@torch.no_grad()
def ranking(model, data, cat, dev, limit=None):
    model.eval()
    n = len(data['target']) if limit is None else min(limit, len(data['target']))
    predictions = []
    for start in range(0, n, 64):
        h = data['history'][start:start+64]
        ts = data['timestamp'][start:start+64]
        logits = legal_logits(actor_logits(model, h, dev), ts, cat, dev)
        predictions.append(logits.topk(20, -1).indices.cpu().numpy())
    top = np.concatenate(predictions)
    target = data['target'][:n]
    uid = data['uid'][:n]
    matched = top == target[:, None]
    vals = {}
    per_event = {}
    for k in (10, 20):
        recall = matched[:, :k].any(-1).astype(float)
        ndcg = (matched[:, :k] / np.log2(np.arange(k)+2)).sum(-1)
        for name, v in [(f'recall@{k}', recall), (f'ndcg@{k}', ndcg)]:
            vals[name] = macro(v, uid)
            per_event[name] = v
            known = target >= 2
            vals[name + '_known'] = macro(v[known], uid[known]) if known.any() else None
    vals['examples'] = n
    vals['users'] = int(len(np.unique(uid)))
    return vals, top, per_event


@torch.no_grad()
def reward_quality(reward, data, cat, meta, dev):
    good = data['target'] >= 2
    indices = np.flatnonzero(good)
    pred = []
    for start in range(0, len(indices), 128):
        idx = indices[start:start+128]
        item = torch.as_tensor(data['target'][idx], device=dev, dtype=torch.long)
        pred.append(reward(data['history'][idx], item, dev).cpu().numpy())
    pred = np.concatenate(pred)
    y = data['ratio'][indices]
    uid = data['uid'][indices]
    def errors(q, mask=None):
        m = np.ones(len(y), dtype=bool) if mask is None else mask
        if not m.any():
            return None
        return dict(n=int(m.sum()), users=int(len(np.unique(uid[m]))), mae=macro(np.abs(q[m]-y[m]), uid[m]),
                    rmse=float(np.sqrt(macro((q[m]-y[m])**2, uid[m]))),
                    predicted_mean=macro(q[m], uid[m]), observed_mean=macro(y[m], uid[m]))
    base = np.full_like(y, meta['global_ratio_mean'])
    item_mean = cat['item_mean'][data['target'][indices]]
    durations = cat['duration'][data['target'][indices]]
    freqs = cat['popularity'][data['target'][indices]]
    result = {'reward_model': errors(pred), 'global_mean': errors(base), 'item_mean': errors(item_mean),
              'short_tracks': errors(pred, durations <= 180), 'long_tracks': errors(pred, durations > 300),
              'popular_tracks': errors(pred, freqs >= np.quantile(cat['popularity'][2:], .9)),
              'rare_tracks': errors(pred, freqs <= np.quantile(cat['popularity'][2:], .5)),
              'non_organic': errors(pred, data['organic'][indices] == 0),
              'organic': errors(pred, data['organic'][indices] == 1),
              'calibration': []}
    for lower in np.arange(0, 1, .1):
        m = (pred >= lower) & (pred < lower+.1)
        if m.any():
            result['calibration'].append(dict(lower=float(lower), **errors(pred, m)))
    return result


def train_baseline(args, meta, cat, dev, rng):
    data, val = load_data('train'), load_data('policy_val')
    run = OUT / args.name
    run.mkdir(exist_ok=True)
    model = GPT2LMHeadModel(model_config(meta)).to(dev)
    if args.initial:
        model.load_state_dict(read_checkpoint(OUT / args.initial / 'best.pt')['model'])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    last = run / 'last.pt'
    step = restore(last, model, opt, rng) if last.exists() else 0
    history_path = run / 'history.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    extra = {'gpt2': model.config.to_dict(), 'arguments': vars(args), 'stage': 'supervised'}
    t0 = time.monotonic()
    while step < args.steps:
        model.train()
        idx = rng.integers(0, len(data['target']), args.batch)
        x = data['history'][idx]
        labels = np.concatenate([x[:, 1:], data['target'][idx, None]], axis=1)
        labels = torch.as_tensor(labels, device=dev, dtype=torch.long)
        labels = labels.masked_fill((labels < 2) | torch.as_tensor(x, device=dev).eq(0)
                                   | torch.as_tensor(data['loss_mask'][idx], device=dev).eq(0), -100)
        logits = model(**inputs(x, dev)).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        step += 1
        if step % args.save_every == 0 or step == args.steps:
            sync(dev)
            elapsed = time.monotonic() - t0
            save_checkpoint(last, model, opt, step, rng, extra)
            row = {'step': step, 'loss': float(loss.detach().cpu()), 'active_seconds_this_session': elapsed}
            if step % args.eval_every == 0 or step == args.steps:
                metrics, _, _ = ranking(model, val, cat, dev, limit=2048)
                row.update(metrics)
                if step == args.steps:
                    save_checkpoint(run / 'best.pt', model, opt, step, rng, extra)
            history.append(row)
            write_json(history_path, history)
            print(json.dumps(row), flush=True)
            if args.minutes and elapsed >= args.minutes * 60:
                break


def train_reward(args, meta, cat, dev, rng):
    data, val = load_data('train'), load_data('reward_val')
    run = OUT / args.name
    run.mkdir(exist_ok=True)
    backbone = load_actor(OUT / args.initial / 'best.pt', dev)
    reward = RewardNet(backbone, cat, meta['duration_cap_seconds']).to(dev)
    # Реальные истории кодируются один раз; это существенно ускоряет обучение.
    cache = run / 'train_context.npy'
    if cache.exists():
        context = np.load(cache)
    else:
        chunks = []
        for start in range(0, len(data['target']), 128):
            chunks.append(reward.encode(data['history'][start:start+128], dev).cpu().numpy())
        context = np.concatenate(chunks)
        np.save(cache, context)
    opt = torch.optim.AdamW([p for p in reward.parameters() if p.requires_grad], lr=args.lr, weight_decay=.01)
    last = run / 'last.pt'
    step = restore(last, reward, opt, rng) if last.exists() else 0
    history_path = run / 'history.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    best = min([h['mae'] for h in history if 'mae' in h], default=float('inf'))
    extra = {'gpt2': backbone.config.to_dict(), 'arguments': vars(args), 'stage': 'reward'}
    t0 = time.monotonic()
    while step < args.steps:
        reward.head.train()
        idx = rng.integers(0, len(data['target']), args.batch)
        c = torch.as_tensor(context[idx], device=dev)
        item = torch.as_tensor(data['target'][idx], device=dev, dtype=torch.long)
        y = torch.as_tensor(data['ratio'][idx], device=dev)
        pred = reward.score(c, item)
        loss = F.mse_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_([p for p in reward.parameters() if p.requires_grad], 1.)
        opt.step()
        step += 1
        if step % args.save_every == 0 or step == args.steps:
            elapsed = time.monotonic() - t0
            save_checkpoint(last, reward, opt, step, rng, extra)
            row = {'step': step, 'loss': float(loss.detach().cpu()), 'active_seconds_this_session': elapsed}
            if step % args.eval_every == 0 or step == args.steps:
                metrics = reward_quality(reward, val, cat, meta, dev)
                row['mae'] = metrics['reward_model']['mae']
                row['rmse'] = metrics['reward_model']['rmse']
                if row['mae'] < best:
                    best = row['mae']
                    save_checkpoint(run / 'best.pt', reward, opt, step, rng, extra)
                    write_json(run / 'reward_val.json', metrics)
            history.append(row)
            write_json(history_path, history)
            print(json.dumps(row), flush=True)
            if args.minutes and elapsed >= args.minutes * 60:
                break


def train_rl(args, meta, cat, dev, rng):
    data, val = load_data('train'), load_data('policy_val')
    run = OUT / args.name
    run.mkdir(exist_ok=True)
    actor = load_actor(OUT / args.initial / 'best.pt', dev)
    ref = load_actor(OUT / args.initial / 'best.pt', dev)
    ref.requires_grad_(False)
    reward = load_reward(OUT / args.reward / 'best.pt', dev, cat, meta)
    reward.requires_grad_(False)
    opt = torch.optim.AdamW(actor.parameters(), lr=args.lr, weight_decay=.01)
    last = run / 'last.pt'
    step = restore(last, actor, opt, rng) if last.exists() else 0
    history_path = run / 'history.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    extra = {'gpt2': actor.config.to_dict(), 'arguments': vars(args), 'stage': 'rl'}
    duration = torch.as_tensor(np.minimum(cat['duration'], meta['duration_cap_seconds'])/meta['duration_cap_seconds'], device=dev)
    t0 = time.monotonic()
    while step < args.steps:
        # Dropout отключён и при sampling, и при вычислении вероятностей.
        actor.eval()
        idx = rng.integers(0, len(data['target']), args.batch)
        h = np.repeat(data['history'][idx], args.samples, axis=0)
        ts = np.repeat(data['timestamp'][idx], args.samples)
        rewards, logps, kls = [], [], []
        for t in range(args.horizon):
            logits = legal_logits(actor_logits(actor, h, dev), ts, cat, dev)
            lp = logits.log_softmax(-1)
            with torch.no_grad():
                ref_lp = legal_logits(actor_logits(ref, h, dev), ts, cat, dev).log_softmax(-1)
                action = torch.multinomial(lp.exp(), 1).squeeze(-1)
                r = reward(h, action, dev)
                if args.reward_type in ('time', 'balanced'):
                    r = r * duration[action]
                if args.reward_type == 'balanced' and t:
                    previous = torch.as_tensor(h[:, -t:], device=dev)
                    r = r - args.repeat_penalty * previous.eq(action[:, None]).any(-1).float()
            rewards.append(r)
            logps.append(lp.gather(1, action[:, None]).squeeze(-1))
            kls.append((lp.exp() * (lp-ref_lp)).sum(-1))
            h = np.concatenate([h[:, 1:], action.detach().cpu().numpy()[:, None]], axis=1)
        return_to_go = torch.zeros_like(rewards[-1])
        policy_loss, kl_loss = 0., 0.
        for t in reversed(range(args.horizon)):
            # Будущие KL-штрафы также входят в return: их величина зависит
            # от ранее выбранных действий через сгенерированную историю.
            return_to_go = rewards[t] - args.beta*kls[t].detach() + args.gamma * return_to_go
            grouped = return_to_go.reshape(args.batch, args.samples)
            # Leave-one-out baseline не включает награду данного сэмпла.
            baseline = (grouped.sum(-1, keepdim=True)-grouped)/(args.samples-1)
            advantage = (grouped-baseline).reshape(-1)
            policy_loss = policy_loss - (args.gamma**t) * (advantage.detach() * logps[t]).mean()
            kl_loss = kl_loss + (args.gamma**t) * kls[t].mean()
        loss = (policy_loss + args.beta * kl_loss) / args.horizon
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.)
        opt.step()
        step += 1
        if step % args.save_every == 0 or step == args.steps:
            elapsed = time.monotonic() - t0
            save_checkpoint(last, actor, opt, step, rng, extra)
            row = {'step': step, 'loss': float(loss.detach().cpu()),
                   'sample_reward': float(torch.stack(rewards).mean().cpu()),
                   'kl': float((kl_loss/args.horizon).detach().cpu()),
                   'active_seconds_this_session': elapsed}
            if step % args.eval_every == 0 or step == args.steps:
                metrics, _, _ = ranking(actor, val, cat, dev, limit=2048)
                row.update(metrics)
            history.append(row)
            write_json(history_path, history)
            print(json.dumps(row), flush=True)
            if args.minutes and elapsed >= args.minutes * 60:
                break
    # Окончательный шаг фиксирован заранее; test не выбирает checkpoint.
    save_checkpoint(run / 'best.pt', actor, opt, step, rng, extra)


def benchmark(args, meta, cat, dev, rng):
    data = load_data('train')
    model = GPT2LMHeadModel(model_config(meta)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    times = []
    for step in range(30):
        idx = rng.integers(0, len(data['target']), args.batch)
        x = data['history'][idx]
        labels = np.concatenate([x[:, 1:], data['target'][idx, None]], -1)
        labels = torch.as_tensor(labels, device=dev, dtype=torch.long)
        labels = labels.masked_fill((labels < 2) | torch.as_tensor(x, device=dev).eq(0)
                                   | torch.as_tensor(data['loss_mask'][idx], device=dev).eq(0), -100)
        sync(dev)
        t0 = time.monotonic()
        logits = model(**inputs(x, dev)).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sync(dev)
        times.append(time.monotonic()-t0)
    result = {'device': str(dev), 'torch': torch.__version__, 'parameters': sum(p.numel() for p in model.parameters()),
              'batch': args.batch, 'history': meta['history_length'], 'vocab': meta['catalog_items']+2,
              'median_seconds_per_step_after_warmup': float(np.median(times[5:])),
              'estimated_minutes_2000_steps': float(np.median(times[5:])*2000/60)}
    if dev.type == 'mps':
        result['mps_allocated_bytes'] = torch.mps.current_allocated_memory()
        result['mps_driver_allocated_bytes'] = torch.mps.driver_allocated_memory()
    write_json(OUT / 'benchmark.json', result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['benchmark', 'baseline', 'reward', 'rl', 'reward-eval'])
    p.add_argument('--name', default='pilot')
    p.add_argument('--initial', default=None)
    p.add_argument('--reward', default='reward_s11')
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--minutes', type=float, default=0)
    p.add_argument('--beta', type=float, default=.1)
    p.add_argument('--horizon', type=int, default=1)
    p.add_argument('--samples', type=int, default=4)
    p.add_argument('--gamma', type=float, default=.9)
    p.add_argument('--repeat-penalty', type=float, default=.1)
    p.add_argument('--reward-type', choices=['ratio', 'time', 'balanced'], default='ratio')
    p.add_argument('--split', choices=['reward_val', 'policy_val', 'test'], default='reward_val')
    args = p.parse_args()
    if args.stage in ('reward', 'rl') and args.initial is None:
        args.initial = 'baseline_s11'
    if args.stage == 'rl':
        assert args.samples >= 2
    rng = seed_all(args.seed)
    meta, cat = metadata()
    dev = device()
    print(f'device={dev}, stage={args.stage}, run={args.name}', flush=True)
    if args.stage == 'benchmark':
        benchmark(args, meta, cat, dev, rng)
    elif args.stage == 'baseline':
        train_baseline(args, meta, cat, dev, rng)
    elif args.stage == 'reward':
        train_reward(args, meta, cat, dev, rng)
    elif args.stage == 'rl':
        train_rl(args, meta, cat, dev, rng)
    elif args.stage == 'reward-eval':
        if args.split == 'test':
            assert (OUT / 'frozen_selection.json').exists(), 'Сначала зафиксируйте модели до test'
        reward = load_reward(OUT / args.reward / 'best.pt', dev, cat, meta)
        quality = reward_quality(reward, load_data(args.split), cat, meta, dev)
        write_json(OUT / args.reward / f'{args.split}.json', quality)
        print(json.dumps(quality, indent=2), flush=True)


if __name__ == '__main__':
    main()
