"""Последовательное выполнение нового протокола без доступа к старому D."""
import argparse
import time

from .core import (OUTPUT, PROJECT, ORIGINAL, catalog, data_manifest, device, load_reward,
                   protocol, read_checkpoint, read_json, sha, write_json, source_hashes,
                   original_integrity, check_frozen)


def reward_stage(limit_steps=None):
    from .train import calibrate_reward, train_reward
    cfg = protocol()['reward']
    started = time.monotonic()
    results = {}
    for variant in cfg['variants']:
        name = f'reward_{variant}_s{cfg["seed"]}'
        path = train_reward(variant, cfg['seed'], limit_steps=limit_steps)
        if path is None:
            print('Сохранена незавершённая reward model; продолжение тем же запуском', name, flush=True)
            return False
        last = read_checkpoint(path.with_name('last.pt'))
        results[name] = {'variant': variant, 'B1_mse': last['best_mse'],
                         'chosen_step': last['best_step'], 'checkpoint_sha256': sha(path)}
    chosen = min(results, key=lambda name: results[name]['B1_mse'])
    calibrated = calibrate_reward(chosen)
    variant = results[chosen]['variant']
    auditor_name = f'reward_{variant}_s{cfg["auditor_seed"]}'
    auditor = train_reward(variant, cfg['auditor_seed'], limit_steps=limit_steps)
    if auditor is None:
        print('Сохранён незавершённый auditor; продолжение тем же запуском', flush=True)
        return False
    calibrated_auditor = calibrate_reward(auditor_name)
    result = {'selected': chosen, 'variant': variant,
              'reward_path': str(calibrated.relative_to(OUTPUT)),
              'reward_sha256': sha(calibrated), 'auditor': auditor_name,
              'auditor_path': str(calibrated_auditor.relative_to(OUTPUT)),
              'auditor_sha256': sha(calibrated_auditor), 'candidates': results,
              'selection_split': 'B1', 'calibration_fit_split': 'B2',
              'C_used_for_selection': False, 'holdout_accessed': False}
    path = OUTPUT / 'reward_selection.json'
    if path.exists() and read_json(path) != result:
        raise ValueError('Изменился зафиксированный выбор reward model')
    write_json(path, result)
    write_json(OUTPUT / 'reward_stage.json', {'completed': True, 'seconds_this_session': time.monotonic()-started,
                                             'reward': chosen, 'auditor': auditor_name})
    print('REWARD SELECTED', result, flush=True)
    return True


def reward_info():
    info = read_json(OUTPUT/'reward_selection.json')
    for key in ['reward', 'auditor']:
        if sha(OUTPUT/info[key+'_path']) != info[key+'_sha256']:
            raise ValueError(f'Изменён выбранный {key}')
    return info, OUTPUT/info['reward_path'], OUTPUT/info['auditor_path']


def pilot_stage(limit_steps=None):
    from .evaluate import descriptor, evaluate
    from .train import actor_path, train_rl
    if not (OUTPUT/'reward_selection.json').exists():
        if not reward_stage(limit_steps):
            return False
    info, reward_path, auditor_path = reward_info()
    cfg = protocol()
    seed = cfg['rl']['pilot_seed']
    specs = [descriptor(f'baseline_s{seed}', actor_path(seed), seed=seed)]
    for kind in cfg['rl']['h1_rewards']:
        for beta in cfg['rl']['h1_betas']:
            path = train_rl(seed, kind, beta, 1, reward_path, limit_steps)
            if path is None:
                print('Сохранён незавершённый RL pilot', kind, beta, flush=True)
                return False
            specs.append(descriptor(path.parent.name, path, seed=seed, reward_type=kind, beta=beta))
        for beta in cfg['controls']['analytic_betas']:
            specs.append(descriptor(f'analytic_{kind}_b{beta:g}', kind='analytic',
                                    seed=seed, reward_type=kind, beta=beta))
        for weight in cfg['controls']['rerank_weights']:
            specs.append(descriptor(f'rerank_{kind}_w{weight:g}', kind='rerank',
                                    seed=seed, reward_type=kind, weight=weight))
    specs.append(descriptor('popularity', kind='popularity', seed=seed))
    result = evaluate('policy_val', specs, reward_path, auditor_path, 'validation_grid')
    base_ndcg = result['models'][f'baseline_s{seed}']['ndcg@10']
    selected = {}
    for kind in cfg['rl']['h1_rewards']:
        candidates = [s for s in specs if s['name'].startswith(f'rl_{kind}_')]
        feasible = [s for s in candidates if result['models'][s['name']]['ndcg@10'] >= .95*base_ndcg]
        metric = 'policy_expected_ratio' if kind=='ratio' else 'policy_expected_time_reward'
        selected[kind] = {'descriptor': max(feasible, key=lambda s: result['models'][s['name']][metric])
                         if feasible else specs[0], 'constraint_satisfied': bool(feasible),
                         'selection_metric': metric, 'selection_split': 'C'}
    for kind in cfg['rl']['h1_rewards']:
        reranks = [s for s in specs if s['name'].startswith(f'rerank_{kind}_')]
        feasible = [s for s in reranks if result['models'][s['name']]['ndcg@10'] >= .95*base_ndcg]
        metric = 'policy_expected_ratio' if kind=='ratio' else 'policy_expected_time_reward'
        selected['rerank_'+kind] = {'descriptor':max(feasible or reranks, key=lambda s: result['models'][s['name']][metric]),
                                   'constraint_satisfied':bool(feasible), 'selection_split':'C'}
    interval = result['reward_quality']['mse_difference_vs_global']['ci95']
    selected['reward_gate'] = {'mse_improvement_supported_on_C': interval[1]<0,
               'mse_difference_ci95':interval, 'reward_changed_after_C':False,
               'interpretation':'RL surrogate diagnostic if superiority unsupported'}
    selected['validation_descriptors'] = specs
    path = OUTPUT/'policy_selection.json'
    if path.exists() and read_json(path) != selected:
        raise ValueError('Выбор политики изменился после фиксации на C')
    write_json(path, selected)
    print('PILOT SELECTED', {k:v for k,v in selected.items() if k!='validation_descriptors'}, flush=True)
    return True


def training_stage(limit_steps=None):
    from .evaluate import descriptor
    from .train import actor_path, train_rl, train_sft
    if not (OUTPUT/'policy_selection.json').exists():
        if not pilot_stage(limit_steps):
            return False
    selection=read_json(OUTPUT/'policy_selection.json')
    info,reward_path,auditor_path=reward_info()
    cfg=protocol()
    # Окончательно оценить всю seed11 RL-сетку, аналитические контроли и выбранные rerank.
    specs=[s for s in selection['validation_descriptors'] if s['kind']!='rerank']
    specs += [selection['rerank_'+kind]['descriptor'] for kind in cfg['rl']['h1_rewards']]
    rollout_specs=[s for s in specs if s['name']=='baseline_s11']
    for kind in cfg['rl']['h1_rewards']:
        chosen=selection[kind]['descriptor']
        rollout_specs.append(chosen)
        rollout_specs.append(selection['rerank_'+kind]['descriptor'])
        beta=chosen['beta'] if chosen['name'].startswith('rl_') else 1.
        rollout_specs.append(descriptor(f'analytic_{kind}_b{beta:g}',kind='analytic',
                                        seed=11,reward_type=kind,beta=beta))
    for seed in cfg['rl']['repeat_seeds']:
        specs.append(descriptor(f'baseline_s{seed}',actor_path(seed),seed=seed))
        for kind in cfg['rl']['h1_rewards']:
            chosen=selection[kind]['descriptor']
            if not chosen['name'].startswith('rl_'):
                continue
            path=train_rl(seed,kind,chosen['beta'],1,reward_path,limit_steps)
            if path is None:
                return False
            specs.append(descriptor(path.parent.name,path,seed=seed,reward_type=kind,beta=chosen['beta']))
    benchmark=read_json(OUTPUT/'benchmark.json')
    budgets={}
    for seed in cfg['baseline']['seeds']:
        chosen=selection['ratio']['descriptor']
        beta=chosen['beta'] if chosen['name'].startswith('rl_') else 1.
        target_name=f'rl_ratio_b{beta:g}_h1_s{seed}'
        timing=OUTPUT/'models'/target_name/'completed.json'
        if not timing.exists():
            # При отсутствии допустимого RL основной результат — база. SFT контролирует
            # технический бюджет beta=1 seed11 и явно помечается диагностическим.
            timing=OUTPUT/'models'/'rl_ratio_b1_h1_s11'/'completed.json'
        seconds=read_json(timing)['training_seconds']
        sft_cfg=cfg['controls']['sft']
        steps=round(seconds/benchmark['timings']['sft']['median_seconds']/sft_cfg['save_every'])*sft_cfg['save_every']
        steps=max(sft_cfg['minimum_steps'],min(sft_cfg['maximum_steps'],steps))
        budgets[str(seed)]={'steps':steps,'target_training_seconds':seconds,
                    'target_run':str(timing.relative_to(OUTPUT)),
                    'benchmark_step_seconds':benchmark['timings']['sft']['median_seconds'],
                    'fallback_diagnostic':not chosen['name'].startswith('rl_')}
    budget_path=OUTPUT/'sft_budgets.json'
    if budget_path.exists() and read_json(budget_path)!=budgets:
        raise ValueError('После фиксации изменился бюджет SFT')
    write_json(budget_path,budgets)
    for seed in cfg['baseline']['seeds']:
        path=train_sft(seed,budgets[str(seed)]['steps'],limit_steps)
        if path is None:
            return False
        spec=descriptor(path.parent.name,path,seed=seed)
        specs.append(spec)
        if seed==11:
            rollout_specs.append(spec)
    for kind in cfg['rl']['h5']['variants']:
        corresponding='time' if kind=='balanced' else kind
        chosen=selection[corresponding]['descriptor']
        beta=chosen['beta'] if chosen['name'].startswith('rl_') else 1.
        path=train_rl(11,kind,beta,5,reward_path,limit_steps)
        if path is None:
            return False
        full=descriptor(path.parent.name,path,seed=11,reward_type=kind,beta=beta)
        full['budget_role']=['full_updates']
        specs.append(full);rollout_specs.append(full)
        stages=list((path.parent/'stages').glob('*.pt'))
        target_name=f'rl_{corresponding}_b{beta:g}_h1_s11'
        seconds=read_json(OUTPUT/'models'/target_name/'completed.json')['training_seconds']
        action_step=cfg['rl']['h5']['steps_action_matched']
        action_path=path.parent/'stages'/f'step{action_step:05d}.pt'
        time_path=min(stages,key=lambda p:abs(read_checkpoint(p)['training_seconds']-seconds))
        chosen_stages={}
        for label,p in [('action_matched',action_path),('time_matched',time_path)]:
            chosen_stages.setdefault(p,[]).append(label)
        for p,roles in chosen_stages.items():
            spec=descriptor(path.parent.name+'_budget_'+p.stem,p,seed=11,reward_type=kind,beta=beta)
            spec['budget_role']=roles
            spec['h1_target_training_seconds']=seconds
            spec['h5_training_seconds']=read_checkpoint(p)['training_seconds']
            specs.append(spec);rollout_specs.append(spec)
    specs=list({s['name']:s for s in specs}.values())
    rollout_specs=list({s['name']:s for s in rollout_specs}.values())
    result={'descriptors':specs,'rollout_descriptors':rollout_specs,
            'selection':selection,'reward':info,'sft_budgets':budgets}
    write_json(OUTPUT/'training_selection.json',result)
    print('TRAINING COMPLETE',len(specs),'policies',len(rollout_specs),'rollout policies',flush=True)
    return True


def freeze_stage():
    from .audit import preflight
    preflight()
    original_integrity()
    selection=read_json(OUTPUT/'training_selection.json')
    info,reward_path,auditor_path=reward_info()
    if not read_json(OUTPUT/'scientific_checks.json')['passed']:
        raise ValueError('Научные проверки не пройдены')
    if read_json(OUTPUT/'benchmark.json')['equivalence']['status']!='passed':
        raise ValueError('Эквивалентность оптимизации не подтверждена')
    artifacts={}
    for spec in selection['descriptors']:
        from .train import actor_path
        ref=actor_path(spec['seed'])
        artifacts[str(ref.relative_to(PROJECT))]=sha(ref)
        if 'actor' in spec:
            artifacts[spec['actor']]=sha(PROJECT/spec['actor'])
    for p in [reward_path,auditor_path,OUTPUT/'reward_selection.json',OUTPUT/'policy_selection.json',
              OUTPUT/'training_selection.json',OUTPUT/'sft_budgets.json',OUTPUT/'benchmark.json',
              OUTPUT/'scientific_checks.json',OUTPUT/'preflight.json',OUTPUT/'retention_diagnostics.json',
              ORIGINAL/'listens.parquet']:
        artifacts[str(p.relative_to(PROJECT))]=sha(p)
    meta=data_manifest()
    for filename,expected in meta['files'].items():
        if sha(OUTPUT/filename)!=expected:
            raise ValueError('Изменены development-данные')
        artifacts[str((OUTPUT/filename).relative_to(PROJECT))]=expected
    frozen={'descriptors':selection['descriptors'],'rollout_descriptors':selection['rollout_descriptors'],
            'sources':source_hashes(),'artifacts':artifacts,
            'data_manifest_sha256':sha(OUTPUT/'data_manifest.json'),
            'reward_path':str(reward_path.relative_to(PROJECT)),
            'auditor_path':str(auditor_path.relative_to(PROJECT)),
            'holdout_scope':meta['new_holdout_kind'],'selection':selection['selection']}
    path=OUTPUT/'frozen.json'
    if path.exists() and read_json(path)!=frozen:
        raise ValueError('Изменён зафиксированный эксперимент')
    write_json(path,frozen)
    check_frozen()
    print('FROZEN before new-cohort outcomes',flush=True)
    return frozen


def holdout_stage():
    from .data import prepare_holdout
    from .evaluate import evaluate
    from .rollout import evaluate_rollouts
    frozen=check_frozen()
    info,reward_path,auditor_path=reward_info()
    prepare_holdout()
    evaluate('holdout',frozen['descriptors'],reward_path,auditor_path,'holdout')
    evaluate_rollouts('holdout',frozen['rollout_descriptors'],reward_path,auditor_path,'holdout')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'checks', 'benchmark', 'reward', 'pilot',
                        'training', 'freeze', 'holdout', 'retention', 'audit', 'report'])
    parser.add_argument('--limit-steps', type=int, default=None,
                        help='Техническое ограничение стадии; не меняет итоговый бюджет или выбор модели')
    args = parser.parse_args()
    if args.stage in ['reward','pilot','training','benchmark','checks','prepare','retention'] and (OUTPUT/'frozen.json').exists():
        raise ValueError('Эксперимент зафиксирован; не изменяйте входы после доступа к holdout')
    if args.stage == 'prepare':
        from .data import prepare
        prepare()
    elif args.stage == 'checks':
        from .tests import main as check
        check()
    elif args.stage == 'benchmark':
        from .benchmark import main as benchmark
        benchmark()
    elif args.stage == 'reward':
        reward_stage(args.limit_steps)
    elif args.stage == 'pilot':
        pilot_stage(args.limit_steps)
    elif args.stage == 'training':
        training_stage(args.limit_steps)
    elif args.stage == 'freeze':
        freeze_stage()
    elif args.stage == 'holdout':
        holdout_stage()
    elif args.stage == 'retention':
        from .retention import run
        run()
    elif args.stage == 'audit':
        from .audit import run
        run()
    elif args.stage == 'report':
        from .report import main as report
        report()


if __name__ == '__main__':
    main()
