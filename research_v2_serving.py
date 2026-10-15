"""Эквивалентный инференс последнего токена и измерение затрат без метрик качества."""
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from research_v2.core import (OUTPUT, PROJECT, actor_logits, catalog, check_frozen,
          data_manifest, device, inputs, legal_logits, load_actor, load_data,
          load_reward, read_json, seed_all, sha, sync, write_json)
from research_v2.evaluate import all_scores
from research_v2.train import actor_path


@torch.no_grad()
def last_logits(model,history,dev):
    hidden=model.transformer(**inputs(history,dev)).last_hidden_state[:,-1]
    return model.lm_head(hidden)


def main():
    frozen=check_frozen()
    rng=seed_all(71)
    dev,meta,cat=device(),data_manifest(),catalog()
    data=load_data('policy_val','post-freeze technical inference benchmark; no parameter selection')
    selected=frozen['selection']
    reward=load_reward(PROJECT/frozen['reward_path'],dev,cat,meta).requires_grad_(False)
    reference=load_actor(actor_path(11),dev).requires_grad_(False)
    # Совместное кодирование допустимо только при идентичных Transformer-весах.
    ref_state=reference.transformer.state_dict()
    rm_state=reward.backbone.transformer.state_dict()
    backbone_error=max(float((ref_state[k]-rm_state[k]).abs().max().cpu()) for k in ref_state)
    if backbone_error!=0:
        raise ValueError('Encoder reward model не совпадает с reference')
    actors={'baseline':reference}
    for kind in ['ratio','time']:
        spec=selected[kind]['descriptor']
        actors['rl_'+kind]=load_actor(PROJECT/spec['actor'],dev).requires_grad_(False)
    locations=rng.choice(len(data['uid']),64,replace=False)
    with torch.no_grad():
        errors={}
        for name,model in actors.items():
            h=data['history'][locations]
            old=actor_logits(model,h,dev)
            new=last_logits(model,h,dev)
            error=float((old-new).abs().max().cpu())
            if error>2e-5:
                raise ValueError('Последняя проекция изменила logits')
            ranks_old=old.topk(20,-1).indices
            ranks_new=new.topk(20,-1).indices
            matching=float((ranks_old==ranks_new).float().mean().cpu())
            if matching<.999:
                raise ValueError('Последняя проекция изменила ранжирование')
            errors[name]={'max_logits_error':error,'top20_positions_matching':matching}
    results={}
    cap=meta['duration_cap_seconds']
    duration=torch.as_tensor(np.minimum(cat['duration'],cap)/cap,device=dev)
    for batch in [1,32]:
        idx=locations[:batch]
        h=data['history'][idx];ts=data['timestamp'][idx].astype(np.int64)
        def recommend(name):
            if name.endswith('_legacy'):
                actor=name.removesuffix('_legacy')
                logits=legal_logits(actor_logits(actors[actor],h,dev),ts,cat,dev)
            elif name in actors:
                logits=legal_logits(last_logits(actors[name],h,dev),ts,cat,dev)
            else:
                kind=name.removeprefix('analytic_')
                hidden=reference.transformer(**inputs(h,dev)).last_hidden_state[:,-1]
                ref_lp=legal_logits(reference.lm_head(hidden),ts,cat,dev).log_softmax(-1)
                r,_=all_scores(reward,hidden)
                if kind=='time':r=r*duration[None,:]
                logits=ref_lp+r/selected[kind]['descriptor']['beta']
            return logits.topk(20,-1).indices
        with torch.no_grad():
            for name in [*[k+'_legacy' for k in actors],*actors,'analytic_ratio','analytic_time']:
                times=[]
                for iteration in range(25):
                    sync(dev);started=time.monotonic()
                    top=recommend(name)
                    sync(dev)
                    if iteration>=5:times.append(time.monotonic()-started)
                results[f'{name}_batch{batch}']={'median_seconds':statistics.median(times),
                     'min_seconds':min(times),'max_seconds':max(times),'measured':len(times),'warmup':5}
    payload={'device':str(dev),'torch':str(torch.__version__),'threads':torch.get_num_threads(),
             'source_sha256':sha(Path(__file__)),'frozen_sha256':sha(OUTPUT/'frozen.json'),
             'backbone_max_error':backbone_error,'equivalence':errors,'measurements':results,
             'scope':'Warm-cache model plus top20 inference on fixed C contexts; excludes ranking-quality metrics, loading, networking and live-service overhead.',
             'analytic_encoding':'One shared frozen reference/reward Transformer, justified by exact weight identity',
             'parameters_changed':False}
    write_json(OUTPUT/'serving_benchmark.json',payload)
    print(payload,flush=True)


if __name__=='__main__':
    main()
