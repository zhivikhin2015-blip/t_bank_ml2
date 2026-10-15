"""Независимое агрегирование метрик и проверка полноты окончательного эксперимента."""
import numpy as np
import torch

from .core import (OUTPUT, PROJECT, catalog, check_frozen, data_manifest, load_data,
                   load_npz, load_reward, original_integrity, protocol, read_checkpoint,
                   read_json, sha, write_json)
from .evaluate import actual_predictions, evaluate_one, score_rows


def preflight():
    original_integrity()
    selected=read_json(OUTPUT/'training_selection.json')
    checks=read_json(OUTPUT/'scientific_checks.json')
    if not checks['passed']:
        raise ValueError('Научные проверки не пройдены')
    for filename,expected in checks.get('sources',{}).items():
        if sha(PROJECT/filename)!=expected:
            raise ValueError('Код изменён после научных проверок: '+filename)
    actor_checks={}
    for descriptor in selected['descriptors']:
        if 'actor' not in descriptor:
            continue
        p=PROJECT/descriptor['actor']
        state=read_checkpoint(p)
        if descriptor['name'].startswith('baseline_'):
            if state['step']!=protocol()['baseline']['steps']:
                raise ValueError('Неправильный бюджет исходной модели')
        else:
            spec=state['config']['spec']
            expected=(protocol()['rl']['h5']['steps_action_matched'] if 'action_matched' in descriptor.get('budget_role',[])
                      else (state['step'] if 'time_matched' in descriptor.get('budget_role',[]) else spec['steps']))
            if state['step']!=expected:
                raise ValueError('Неправильная стадия модели')
            for filename,value in state['config']['dependencies'].items():
                if sha(PROJECT/filename)!=value:
                    raise ValueError('Изменены зависимости обучения: '+filename)
        actor_checks[descriptor['name']]={'step':state['step'],'sha256':sha(p)}
    write_json(OUTPUT/'preflight.json',{'status':'passed','actors':actor_checks,
                'scientific_checks':checks['checks_run'],'holdout_outcomes_read':False})
    return actor_checks


def independent_relevance(top,target,uid,k):
    recalls,ndcgs=[],[]
    grouped_r,grouped_n={},{}
    for row,y,user in zip(top,target,uid):
        places=[i for i,item in enumerate(row[:k]) if item==y]
        r=float(bool(places))
        n=1/np.log2(places[0]+2) if places else 0.
        recalls.append(r);ndcgs.append(n)
        grouped_r.setdefault(int(user),[]).append(r)
        grouped_n.setdefault(int(user),[]).append(n)
    return (float(np.mean([np.mean(v) for v in grouped_r.values()])),
            float(np.mean([np.mean(v) for v in grouped_n.values()])),np.array(recalls),np.array(ndcgs))


def run():
    torch.set_num_threads(4)
    frozen=check_frozen()
    original_integrity()
    directory=OUTPUT/'evaluations'/'holdout'
    transaction=read_json(directory/'manifest.json')
    results=read_json(directory/'results.json')
    if not transaction['completed'] or not results['completed']:
        raise ValueError('Окончательная оценка не завершена')
    if sha(directory/'results.json')!=transaction['results_sha256']:
        raise ValueError('Изменены итоговые результаты')
    data=load_data('holdout','independent completion audit after final evaluation')
    meta,cat=data_manifest(),catalog()
    reservation=read_json(OUTPUT/'holdout_reservation.json')
    if not set(data['uid']).issubset(reservation['eligible_users']):
        raise ValueError('Пользователи вне зарезервированной когорты')
    original_train=load_npz(OUTPUT/'train.npz')
    if set(data['uid'])&set(original_train['uid']):
        raise ValueError('Пересечение пользователей обучения и holdout')
    # Независимо восстановить весь исходный ID-отбор, включая исключённых из train.
    import hashlib
    import pyarrow.parquet as pq
    from .core import ORIGINAL
    ids=np.unique(pq.read_table(ORIGINAL/'listens.parquet',columns=['uid'])['uid'].to_numpy())
    ordered=sorted(ids.tolist(),key=lambda u:hashlib.sha256(str(u).encode()).digest())
    original_users=ordered[:protocol()['dataset']['original_user_count']]
    if set(data['uid'])&set(original_users):
        raise ValueError('Пересечение с исходными 1024 пользователями')
    if set(results['models'])!={s['name'] for s in frozen['descriptors']}:
        raise ValueError('Неполный набор окончательных моделей')
    checked=[]
    for spec in frozen['descriptors']:
        name=spec['name']
        expected=transaction['completed_models'][name]
        for filename,value in expected.items():
            if sha(directory/filename)!=value:
                raise ValueError('Изменён результат модели')
        pred=load_npz(directory/(name+'.npz'))
        top=pred['top']
        if top.shape!=(len(data['uid']),20) or not np.array_equal(pred['uid'],data['uid']):
            raise ValueError('Несовпадающие строки рекомендаций')
        if not np.all(top>=2) or not np.all(cat['first_seen'][top]<data['timestamp'][:,None]):
            raise ValueError('Недопустимые кандидаты')
        if not all(len(set(row))==20 for row in top):
            raise ValueError('Повторы внутри ранжированного top20')
        for k in [10,20]:
            recall,ndcg,r,n=independent_relevance(top,data['target'],data['uid'],k)
            if abs(recall-results['models'][name][f'recall@{k}'])>1e-10 or abs(ndcg-results['models'][name][f'ndcg@{k}'])>1e-10:
                raise ValueError('Не совпадают независимо пересчитанные метрики')
            np.testing.assert_allclose(r,pred[f'recall@{k}'],atol=0,rtol=0)
            np.testing.assert_allclose(n,pred[f'ndcg@{k}'],atol=1e-12,rtol=0)
        checked.append(name)
    # Независимый CPU-инференс трёх заранее выбранных акторов на 64 holdout-строках.
    idx=np.sort(np.random.default_rng(71).choice(len(data['uid']),min(64,len(data['uid'])),replace=False))
    subset={key:value[idx] for key,value in data.items()}
    reward_path=PROJECT/frozen['reward_path']
    auditor_path=PROJECT/frozen['auditor_path']
    reward=load_reward(reward_path,torch.device('cpu'),cat,meta)
    auditor=load_reward(auditor_path,torch.device('cpu'),cat,meta)
    with torch.no_grad():
        ctx=reward.encode(subset['history'],torch.device('cpu'))
        audit_ctx=auditor.encode(subset['history'],torch.device('cpu'))
        from .evaluate import all_scores
        scores,_=all_scores(reward,ctx)
        audit_scores,_=all_scores(auditor,audit_ctx)
    scores,audit_scores=scores.numpy(),audit_scores.numpy()
    names={'baseline_s11',frozen['selection']['ratio']['descriptor']['name'],
           frozen['selection']['time']['descriptor']['name']}
    inference={}
    for spec in frozen['descriptors']:
        if spec['name'] not in names:
            continue
        metrics,pred=evaluate_one(spec,subset,cat,meta,scores,audit_scores,torch.device('cpu'))
        saved=load_npz(directory/(spec['name']+'.npz'))
        same=float(np.mean(pred['top']==saved['top'][idx]))
        quantities={key:float(np.max(np.abs(pred[key]-saved[key][idx]))) for key in
                    ['policy_expected_ratio','policy_expected_time_reward','kl_to_reference']}
        if same<.999 or max(quantities.values())>2e-5:
            raise ValueError('CPU-воспроизведение существенно расходится с MPS')
        inference[spec['name']]={'rows':len(idx),'top_positions_matching':same,
                               'max_quantity_errors':quantities}
    reward_pred=load_npz(directory/'reward_predictions.npz')
    with torch.no_grad():
        item=torch.as_tensor(subset['target'],dtype=torch.long)
        cpu_prediction=reward(subset['history'],item,torch.device('cpu')).numpy()
    error=float(np.max(np.abs(cpu_prediction-reward_pred['prediction'][idx])))
    if error>2e-6:
        raise ValueError('Reward-предсказания не воспроизводятся из весов')
    rollouts=read_json(OUTPUT/'rollouts'/'holdout'/'results.json')
    roll_manifest=read_json(OUTPUT/'rollouts'/'holdout'/'manifest.json')
    if not rollouts['completed'] or set(rollouts['models'])!={s['name'] for s in frozen['rollout_descriptors']}:
        raise ValueError('Неполный rollout-анализ')
    for name,hashes in roll_manifest['models'].items():
        for filename,value in hashes.items():
            if sha(OUTPUT/'rollouts'/'holdout'/filename)!=value:
                raise ValueError('Изменены rollout-артефакты')
        records=load_npz(OUTPUT/'rollouts'/'holdout'/(name+'.npz'))
        counts=len(np.unique(records['sampling_seed']))*protocol()['rollout']['trajectories_per_user']*protocol()['rollout']['steps']
        _,n=np.unique(records['uid'],return_counts=True)
        if not np.all(n==counts):
            raise ValueError('Неравное/неполное число траекторий по пользователям')
        if not np.all(records['step']<=protocol()['rollout']['steps']):
            raise ValueError('Неожиданная длина rollout')
    result={'status':'passed','original_snapshot':'unchanged','frozen_inputs':'passed',
            'independently_recomputed_relevance_models':checked,'cpu_inference':inference,
            'cpu_reward_prediction_max_error':error,'rollout_models':len(rollouts['models']),
            'new_cohort_disjoint_from_training':True,'scope':meta['new_holdout_kind']}
    write_json(OUTPUT/'completion_audit.json',result)
    print('COMPLETION AUDIT',result,flush=True)
    return result


if __name__=='__main__':
    run()
