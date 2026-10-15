"""Повторные искусственные траектории с отдельной неопределённостью по пользователям."""
import time

import numpy as np
import torch

from .core import (OUTPUT, PROJECT, actor_logits, canonical_hash, catalog, check_frozen,
                   data_manifest, device, legal_logits, load_actor, load_data, load_npz,
                   load_reward, paired_interval, protocol, read_json, save_npz, seed_all,
                   sha, source_hashes, write_json)
from .evaluate import all_scores
from .train import actor_path


@torch.no_grad()
def trajectory(spec, data, indices, reward, auditor, cat, meta, dev):
    cfg = protocol()['rollout']
    ref = load_actor(actor_path(spec['seed']), dev)
    actor = load_actor(PROJECT/spec['actor'], dev) if spec['kind']=='actor' else None
    records = {key: [] for key in ['uid', 'sampling_seed', 'trajectory', 'step', 'token',
                'ratio', 'auditor_ratio', 'time_reward', 'penalized_time_reward', 'regularized_reward', 'original_context_ratio', 'duration',
                'repeat', 'seen_real', 'kl']}
    for sampling_seed in cfg['sampling_seeds']:
        seed_all(sampling_seed)
        for start in range(0, len(indices), 32):
            loc = indices[start:start+32]
            original = np.repeat(data['history'][loc], cfg['trajectories_per_user'], axis=0)
            uid = np.repeat(data['uid'][loc], cfg['trajectories_per_user'])
            trajectory_id = np.tile(np.arange(cfg['trajectories_per_user']), len(loc))
            ts = np.repeat(data['timestamp'][loc], cfg['trajectories_per_user'])
            h = original.copy()
            generated = []
            for step in range(cfg['steps']):
                ref_lp = legal_logits(actor_logits(ref, h, dev), ts, cat, dev).log_softmax(-1)
                context = reward.encode(h, dev)
                if spec['kind']=='actor':
                    logits = legal_logits(actor_logits(actor, h, dev), ts, cat, dev)
                elif spec['kind']=='analytic':
                    ratios, _ = all_scores(reward, context)
                    r = ratios
                    if spec['reward_type']=='time':
                        r = r * torch.as_tensor(np.minimum(cat['duration'],meta['duration_cap_seconds'])/
                                                meta['duration_cap_seconds'],device=dev)[None,:]
                    logits = ref_lp + r/spec['beta']
                elif spec['kind']=='rerank':
                    count = protocol()['controls']['rerank_candidates']
                    candidates = ref_lp.topk(count,-1).indices
                    r = reward.score(context.repeat_interleave(count,0),candidates.reshape(-1)).reshape(-1,count)
                    if spec['reward_type']=='time':
                        duration = torch.as_tensor(np.minimum(cat['duration'],meta['duration_cap_seconds'])/
                                                   meta['duration_cap_seconds'],device=dev)
                        r = r * duration[candidates]
                    supported = torch.zeros_like(ref_lp,dtype=torch.bool).scatter_(1,candidates,True)
                    logits = torch.full_like(ref_lp,-1e9)
                    logits.scatter_(1,candidates,ref_lp.gather(1,candidates)+spec['weight']*r)
                else:
                    raise ValueError(spec['kind'])
                lp = logits.log_softmax(-1)
                action = torch.multinomial(lp.exp(),1).squeeze(-1)
                item = action.cpu().numpy()
                ratio = reward.score(context,action).cpu().numpy()
                original_ratio = reward(original,action,dev).cpu().numpy()
                auditor_ratio = auditor(h,action,dev).cpu().numpy()
                repeated = np.zeros(len(item),dtype=bool) if not generated else (np.stack(generated,-1)==item[:,None]).any(-1)
                time_reward=ratio*np.minimum(cat['duration'][item],meta['duration_cap_seconds'])/meta['duration_cap_seconds']
                penalized=time_reward-protocol()['rl']['h5']['repeat_penalty']*repeated
                kl=(lp.exp()*(lp-ref_lp)).sum(-1).cpu().numpy()
                objective_reward=ratio if spec['reward_type']=='ratio' else (penalized if spec['reward_type']=='balanced' else time_reward)
                values = [uid,np.full(len(uid),sampling_seed),trajectory_id,np.full(len(uid),step+1),item,
                          ratio,auditor_ratio,time_reward,penalized,objective_reward-spec['beta']*kl,
                          original_ratio,cat['duration'][item],repeated,(original==item[:,None]).any(-1),kl]
                for key,value in zip(records,values):
                    records[key].append(value)
                generated.append(item)
                h=np.concatenate([h[:,1:],item[:,None]],-1)
    return {key:np.concatenate(value) for key,value in records.items()}


def summarize(records):
    cfg = protocol()['rollout']
    users,inverse,counts=np.unique(records['uid'],return_inverse=True,return_counts=True)
    result={'users':len(users),'trajectories_per_user_per_seed':cfg['trajectories_per_user'],
            'sampling_seeds':cfg['sampling_seeds'],'steps':[]}
    # Каждый пользователь имеет одинаковое число траекторий, все пять шагов присутствуют.
    per_user={}
    for kind in ['ratio','time_reward','penalized_time_reward','regularized_reward']:
        value=records[kind]*cfg['gamma']**(records['step']-1)
        returns=np.bincount(inverse,weights=value)/counts*cfg['steps']
        per_user[kind]=returns
        ci=paired_interval(returns,np.zeros_like(returns),users)
        result['discounted_'+kind+'_return']={'mean':float(returns.mean()),'ci95':ci['ci95']}
    result['sampling_seed_means']={}
    for seed in cfg['sampling_seeds']:
        m=records['sampling_seed']==seed
        result['sampling_seed_means'][str(seed)]={kind:float(np.mean(records[kind][m]*
                  cfg['gamma']**(records['step'][m]-1))*cfg['steps']) for kind in ['ratio','time_reward']}
    for step in range(1,cfg['steps']+1):
        m=records['step']==step
        uid=records['uid'][m]
        u,inv,n=np.unique(uid,return_inverse=True,return_counts=True)
        row={'step':step,'users':len(u)}
        for key in ['ratio','auditor_ratio','time_reward','original_context_ratio','duration','repeat','seen_real','kl']:
            mean=np.bincount(inv,weights=records[key][m])/n
            row[key]={'mean':float(mean.mean()),'ci95':paired_interval(mean,np.zeros_like(mean),u)['ci95']}
        shifts=np.bincount(inv,weights=records['ratio'][m]-records['original_context_ratio'][m])/n
        row['context_score_shift']={'mean':float(shifts.mean()),'ci95':paired_interval(shifts,np.zeros_like(shifts),u)['ci95']}
        result['steps'].append(row)
    result['scope']='Artificial always-append-ID process, fixed fitted models. Intervals conditional on sampled trajectories and users; no real response labels.'
    return result,{'uid':users,**per_user}


def evaluate_rollouts(split,specs,reward_path,auditor_path,name):
    if split=='holdout':
        frozen=check_frozen()
        if specs!=frozen['rollout_descriptors']:
            raise ValueError('Изменён набор окончательных rollout')
    meta,cat,dev=data_manifest(),catalog(),device()
    data=load_data(split,f'rollout evaluation {name}')
    _,indices=np.unique(data['uid'],return_index=True)
    if len(indices)>protocol()['rollout']['users']:
        indices=np.sort(np.random.default_rng(protocol()['dataset']['sampling_seed']).choice(
                        indices,protocol()['rollout']['users'],replace=False))
    reward=load_reward(reward_path,dev,cat,meta).requires_grad_(False)
    auditor=load_reward(auditor_path,dev,cat,meta).requires_grad_(False)
    directory=OUTPUT/'rollouts'/name
    directory.mkdir(parents=True,exist_ok=True)
    identity={'sources':source_hashes(),'data':sha(OUTPUT/(split+'.npz')),
              'reward':sha(reward_path),'auditor':sha(auditor_path),'specs':specs}
    manifest_path=directory/'manifest.json'
    if manifest_path.exists():
        manifest=read_json(manifest_path)
        if manifest['identity']!=identity:
            raise ValueError('Изменены входы rollout')
    else:
        manifest={'identity':identity,'models':{},'completed':False}
        write_json(manifest_path,manifest)
    for model_name,checksums in manifest['models'].items():
        for filename,checksum in checksums.items():
            if sha(directory/filename)!=checksum:
                raise ValueError('Изменены сохранённые rollout')
    results={}
    for spec in specs:
        model_name=spec['name']
        if model_name not in manifest['models']:
            started=time.monotonic()
            records=trajectory(spec,data,indices,reward,auditor,cat,meta,dev)
            result,returns=summarize(records)
            result['seconds']=time.monotonic()-started
            save_npz(directory/(model_name+'.npz'),**records)
            save_npz(directory/(model_name+'_returns.npz'),**returns)
            write_json(directory/(model_name+'.json'),result)
            manifest['models'][model_name]={filename:sha(directory/filename) for filename in
                      [model_name+'.npz',model_name+'_returns.npz',model_name+'.json']}
            write_json(manifest_path,manifest)
            print('ROLLOUT',name,model_name,result['discounted_time_reward_return'],flush=True)
        results[model_name]=read_json(directory/(model_name+'.json'))
    comparisons={}
    time_names=[s['name'] for s in specs if s['name'].startswith('rl_time_') and '_h5_' in s['name'] and 'budget' not in s['name']]
    balanced_names=[s['name'] for s in specs if s['name'].startswith('rl_balanced_') and '_h5_' in s['name'] and 'budget' not in s['name']]
    if time_names and balanced_names:
        a=load_npz(directory/(balanced_names[0]+'_returns.npz'))
        b=load_npz(directory/(time_names[0]+'_returns.npz'))
        assert np.array_equal(a['uid'],b['uid'])
        comparisons['balanced_vs_time_return']=paired_interval(a['time_reward'],b['time_reward'],a['uid'])
    write_json(directory/'results.json',{'split':split,'models':results,'comparisons':comparisons,'completed':True})
    manifest['completed']=True
    manifest['results_sha256']=sha(directory/'results.json')
    write_json(manifest_path,manifest)
    return read_json(directory/'results.json')
