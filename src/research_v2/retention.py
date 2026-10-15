"""Описательная проверка возвращения после конца сессии, только A/B/C."""
import numpy as np
import pyarrow.dataset as ds
import torch
from torch.nn import functional as F

from .core import (ORIGINAL, OUTPUT, data_manifest, load_data, log_access, macro,
                   protocol, seed_all, user_weights, write_json)


def weighted_ap(label, score, uid):
    weights=user_weights(uid)
    total=float(np.sum(weights*label))
    if total==0:
        return None
    order=np.argsort(-score,kind='stable')
    score,label,weights=score[order],label[order],weights[order]
    ends=np.r_[np.flatnonzero(np.diff(score))+1,len(score)]
    starts=np.r_[0,ends[:-1]]
    positives=all_weight=area=0.
    for left,right in zip(starts,ends):
        amount=float(np.sum(weights[left:right]*label[left:right]))
        positives+=amount
        all_weight+=float(weights[left:right].sum())
        area+=amount*positives/all_weight
    return area/total


def quality(pred,data):
    y,uid=data['label'],data['uid']
    pred=np.clip(pred,1e-6,1-1e-6)
    return {'examples':len(y),'users':len(np.unique(uid)),
            'macro_return_rate':macro(y,uid),'brier':macro((pred-y)**2,uid),
            'logloss':macro(-(y*np.log(pred)+(1-y)*np.log1p(-pred)),uid),
            'negative_AP_user_weighted':weighted_ap(1-y,1-pred,uid),
            'negative_rate_user_weighted':macro(1-y,uid)}


def run():
    meta=data_manifest()
    cfg=protocol()['retention']
    train=load_data('train','session-retention development cohort IDs')
    uids=np.unique(train['uid'])
    table=ds.dataset(ORIGINAL/'listens.parquet').to_table(columns=['uid','timestamp'],
          filter=ds.field('uid').isin(uids.tolist())&(ds.field('timestamp')<meta['cuts']['C_end']))
    a={name:table[name].to_numpy() for name in table.column_names}
    order=np.lexsort((a['timestamp'],a['uid']))
    uid,timestamp=a['uid'][order],a['timestamp'][order]
    log_access('original A/B/C timestamps','Session ends and mature seven-day labels; no D/new-cohort outcomes')
    starts=np.r_[0,np.flatnonzero(np.diff(uid))+1]
    ends=np.r_[starts[1:],len(uid)]
    boundaries=[0,meta['cuts']['A_end'],meta['cuts']['B_end'],meta['cuts']['C_end']]
    results={}
    for gap in cfg['session_gaps_seconds']:
        storage=[[] for _ in range(3)]
        for left,right in zip(starts,ends):
            ts=timestamp[left:right].astype(np.int64)
            session_end=np.r_[np.flatnonzero(np.diff(ts)>gap),len(ts)-1]
            session_begin=np.r_[0,session_end[:-1]+1]
            for begin,end in zip(session_begin,session_end):
                t=int(ts[end])
                split=int(np.searchsorted(boundaries[1:],t,side='right'))
                if split>=3 or t+cfg['horizon_seconds']>=boundaries[split+1]:
                    continue
                # Следующее событие должно начинать новую сессию; полноценное окно наблюдается.
                returned=end+1<len(ts) and ts[end+1]<=t+cfg['horizon_seconds']
                past_day=end-np.searchsorted(ts,t-86400,side='left')
                past_week=end-np.searchsorted(ts,t-cfg['horizon_seconds'],side='left')
                features=[np.log1p(end-begin+1),np.log1p(past_day),np.log1p(past_week)]
                storage[split].append((int(uid[left]),float(returned),features))
        data=[]
        for rows in storage:
            if not rows:
                raise ValueError('Пустой период проверки возвращаемости')
            data.append({'uid':np.array([r[0] for r in rows]),'label':np.array([r[1] for r in rows]),
                         'features':np.array([r[2] for r in rows],dtype=np.float32)})
        a_data=data[0]
        rate=macro(a_data['label'],a_data['uid'])
        users,inv,counts=np.unique(a_data['uid'],return_inverse=True,return_counts=True)
        user_sum=np.bincount(inv,weights=a_data['label'])
        user_rate={int(u):(y+20*rate)/(n+20) for u,y,n in zip(users,user_sum,counts)}
        negative_users=len(np.unique(a_data['uid'][a_data['label']==0]))
        out={'gap_seconds':gap,'train_negative_users':negative_users,'periods':{},
             'logistic_model_trained':negative_users>=cfg['minimum_negative_users_for_model']}
        logistic=None
        if out['logistic_model_trained']:
            seed_all(71)
            # Крошечная CPU-модель: все преобразования обучаются только на A.
            mean=a_data['features'].mean(0)
            std=a_data['features'].std(0).clip(min=1e-6)
            x=torch.as_tensor((a_data['features']-mean)/std)
            y=torch.as_tensor(a_data['label'],dtype=torch.float32)
            weights=torch.as_tensor(user_weights(a_data['uid']),dtype=torch.float32)
            logistic=torch.nn.Linear(3,1)
            optimizer=torch.optim.AdamW(logistic.parameters(),lr=.05,weight_decay=.01)
            for _ in range(500):
                loss=(F.binary_cross_entropy_with_logits(logistic(x).squeeze(-1),y,reduction='none')*weights).sum()
                optimizer.zero_grad();loss.backward();optimizer.step()
        for name,d in zip(['A','B','C'],data):
            pred=np.full(len(d['label']),rate)
            out['periods'][name]={'constant':quality(pred,d),
                      'user_train_rate':quality(np.array([user_rate.get(int(u),rate) for u in d['uid']]),d)}
            if logistic is not None:
                with torch.no_grad():
                    p=logistic(torch.as_tensor((d['features']-mean)/std)).squeeze(-1).sigmoid().numpy()
                out['periods'][name]['past_activity_logistic']=quality(p,d)
        out['interpretation']='Development description in future-activity-selected cohort; predictive associations do not identify recommendation effects. No retention signal is used as an RL reward.'
        results[str(gap)]=out
    write_json(OUTPUT/'retention_diagnostics.json',results)
    print('RETENTION',results,flush=True)
    return results


if __name__=='__main__':
    run()
