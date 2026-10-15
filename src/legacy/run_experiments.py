"""Последовательный запуск протокола; завершённые стадии не обучаются заново."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from experiment import OUT, ROOT, read_checkpoint, write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(script, arguments):
    cmd = [sys.executable, str(ROOT/script), *map(str, arguments)]
    print('RUN', ' '.join(cmd), flush=True)
    started = time.monotonic()
    subprocess.run(cmd, check=True, cwd=ROOT)
    elapsed = time.monotonic()-started
    with open(OUT/'execution_log.jsonl', 'a') as f:
        f.write(json.dumps({'script':script, 'arguments':arguments, 'seconds':elapsed})+'\n')


def trained(name, steps):
    path = OUT/name/'last.pt'
    if path.exists() and read_checkpoint(path)['step'] >= steps and (OUT/name/'best.pt').exists():
        print('ALREADY COMPLETE', name, flush=True)
        return True
    return False


def baseline(seed):
    name = f'baseline_s{seed}'
    if not trained(name, 6000):
        execute('experiment.py', ['baseline', '--name',name, '--seed',seed, '--steps',6000,
                                 '--batch',32, '--lr',.001, '--save-every',250, '--eval-every',1000])
    return name


def reward(seed):
    name = f'reward_s{seed}'
    if not trained(name, 4000):
        execute('experiment.py', ['reward', '--name',name, '--initial',f'baseline_s{seed}',
                                 '--seed',seed, '--steps',4000, '--batch',256, '--lr',.001,
                                 '--save-every',250, '--eval-every',500])
    return name


def rl(seed, reward_type, beta, horizon):
    name = f'rl_{reward_type}_b{beta:g}_h{horizon}_s{seed}'
    if not trained(name, 1000):
        execute('experiment.py', ['rl', '--name',name, '--initial',f'baseline_s{seed}',
                                 '--reward','reward_s11', '--seed',seed, '--steps',1000,
                                 '--batch',16, '--samples',4, '--lr',.00005,
                                 '--beta',beta, '--horizon',horizon, '--reward-type',reward_type,
                                 '--save-every',100, '--eval-every',500])
    return name


def choose_policy(validation, kind):
    base = validation['models']['baseline_s11']['ndcg@10']
    metric = 'predicted_ratio_at10' if kind=='ratio' else 'normalized_time_reward_at10'
    candidates = [name for name in validation['models'] if name.startswith(f'rl_{kind}_') and '_h1_' in name]
    feasible = [name for name in candidates if validation['models'][name]['ndcg@10'] >= .95*base]
    selected = max(feasible, key=lambda name: validation['models'][name][metric]) if feasible else max(candidates, key=lambda name: validation['models'][name]['ndcg@10'])
    cfg = read_checkpoint(OUT/selected/'best.pt')['config']['arguments']
    return {'name':selected, 'beta':cfg['beta'], 'horizon':cfg['horizon'],
            'constraint_satisfied':bool(feasible), 'metric':metric}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stop-after', choices=['validation', 'training', 'test'], default='validation')
    args = p.parse_args()
    protocol = json.loads((ROOT/'protocol.json').read_text())
    for seed in [11, 22, 33]:
        baseline(seed)
    for seed in [11, 22]:
        reward(seed)
    grid_names = []
    for variant in protocol['seed11_grid']:
        grid_names.append(rl(11, variant['reward'], variant['beta'], variant['horizon']))
    validation_path = OUT/'policy_val_evaluation.json'
    if not validation_path.exists() or not json.loads(validation_path.read_text()).get('completed'):
        execute('evaluate.py', ['--split','policy_val', '--models','baseline_s11',*grid_names,
                               '--auditor','reward_s22', '--rerank-weights',0,1,3,10, '--rollouts'])
    val = json.loads(validation_path.read_text())
    selection = {kind:choose_policy(val, kind) for kind in ['ratio','time']}
    reranks = [n for n in val['models'] if n.startswith('rerank_')]
    base_ndcg = val['models']['baseline_s11']['ndcg@10']
    feasible = [n for n in reranks if val['models'][n]['ndcg@10'] >= .95*base_ndcg]
    rr_name = max(feasible or reranks, key=lambda n: val['models'][n]['predicted_ratio_at10'])
    selection['rerank_weight'] = float(rr_name.removeprefix('rerank_weight'))
    selection['protocol_sha256'] = sha(ROOT/'protocol.json')
    write_json(OUT/'validation_selection.json', selection)
    print('SELECTION', json.dumps(selection,ensure_ascii=False), flush=True)
    if args.stop_after == 'validation':
        return
    repeated = []
    for seed in [22, 33]:
        for kind in ['ratio','time']:
            choice = selection[kind]
            repeated.append(rl(seed, kind, choice['beta'], choice['horizon']))
    # SFT-контроль с близким временем по измеренной скорости baseline.
    # Число шагов фиксируется по техническому времени без test-метрик.
    ratio_history = json.loads((OUT/selection['ratio']['name']/'history.json').read_text())
    seconds = ratio_history[-1]['active_seconds_this_session']
    base_history = json.loads((OUT/'baseline_s11'/'history.json').read_text())
    observed_sft_step = base_history[-1]['active_seconds_this_session']/6000
    sft_steps = max(1000, int(round(seconds/observed_sft_step/250)*250))
    sft_steps = min(sft_steps, 20000)
    sft_name = 'sft_extra_s11'
    if not trained(sft_name, sft_steps):
        execute('experiment.py', ['baseline', '--name',sft_name, '--initial','baseline_s11', '--seed',11,
                                 '--steps',sft_steps, '--batch',32, '--lr',.00005,
                                 '--save-every',250, '--eval-every',1000])
    models = ['baseline_s11','baseline_s22','baseline_s33', *grid_names, *repeated, sft_name]
    frozen = {'models':models, 'selection':selection, 'extra_sft_steps':sft_steps,
              'checkpoint_sha256':{name:sha(OUT/name/'best.pt') for name in models},
              'reward_checkpoint_sha256':sha(OUT/'reward_s11'/'best.pt'),
              'auditor_checkpoint_sha256':sha(OUT/'reward_s22'/'best.pt'),
              'protocol_sha256':sha(ROOT/'protocol.json'),
              'source_sha256':{name:sha(ROOT/name) for name in ['prepare.py','experiment.py','evaluate.py']}}
    if (OUT/'frozen_selection.json').exists():
        assert frozen == json.loads((OUT/'frozen_selection.json').read_text()), 'После фиксации модели изменились'
    else:
        write_json(OUT/'frozen_selection.json', frozen)
    if args.stop_after == 'training':
        return
    test_path = OUT/'test_evaluation.json'
    if test_path.exists() and json.loads(test_path.read_text()).get('completed'):
        print('TEST ALREADY COMPLETE', flush=True)
    else:
        execute('evaluate.py', ['--split','test', '--models',*models, '--auditor','reward_s22',
                               '--rerank-weights',selection['rerank_weight'], '--rollouts'])
    for rm in ['reward_s11','reward_s22']:
        if not (OUT/rm/'test.json').exists():
            execute('experiment.py', ['reward-eval','--reward',rm, '--split','test'])


if __name__ == '__main__':
    main()
