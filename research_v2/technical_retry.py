"""Документированный технический повтор: uint32 -> int64 без изменения значений.

Исходный frozen.json и исходный код v2 не переписываются. Адаптер имеет
собственный фиксируемый хеш; перед каждым повтором проверяются оба манифеста.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import core


def canonical_integers(data):
    result=dict(data)
    for name in ['uid','timestamp']:
        if name not in result:
            continue
        value=result[name]
        if not np.issubdtype(value.dtype,np.integer):
            raise TypeError('Ожидается целочисленное поле '+name)
        if int(value.min())<np.iinfo(np.int64).min or int(value.max())>np.iinfo(np.int64).max:
            raise OverflowError('Значение не помещается в int64: '+name)
        converted=value.astype(np.int64,copy=False)
        if not np.array_equal(value,converted):
            raise ValueError('Приведение изменило численные значения: '+name)
        result[name]=converted
    return result


def self_check():
    data={'uid':np.array([12,19],dtype=np.uint32),
          'timestamp':np.array([2,4],dtype=np.uint32),
          'history':np.array([[2,3],[3,4]],dtype=np.int32)}
    fixed=canonical_integers(data)
    assert fixed['uid'].dtype==np.int64 and fixed['timestamp'].dtype==np.int64
    assert fixed['history'] is data['history']
    logits=torch.zeros(2,4)
    cat={'first_seen':np.array([0,0,1,2],dtype=np.int64)}
    legal=core.legal_logits(logits,fixed['timestamp'],cat,torch.device('cpu'))
    np.testing.assert_array_equal(legal.numpy()>-1e8,[[False,False,True,False],[False,False,True,True]])
    return {'passed':True,'checks':['exact numeric identity','history unchanged','signed integer candidate comparison'],
            'synthetic_check_only':True}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['holdout','audit','report'],nargs='?',default='holdout')
    args=parser.parse_args()
    frozen=core.check_frozen()
    original=core.original_integrity()
    path=core.OUTPUT/'technical_retry.json'
    fingerprint={'original_frozen_sha256':core.sha(core.OUTPUT/'frozen.json'),
                 'adapter_sha256':core.sha(Path(__file__)),
                 'holdout_npz_sha256':core.sha(core.OUTPUT/'holdout.npz'),
                 'reason':'PyTorch/MPS rejected comparison of int64 catalog first_seen with Parquet uint32 timestamp',
                 'repair':'Cast uid and timestamp to int64 in memory, require exact numeric identity; no change to saved data or models',
                 'first_failure_stage':'holdout first policy, before any policy result was completed',
                 'heldout_reward_outcomes_accessed_before_retry':True,
                 'model_selection_unchanged':True,'initial_frozen_manifest_unchanged':True}
    if path.exists():
        previous=core.read_json(path)
        if previous['fingerprint']!=fingerprint:
            raise ValueError('Изменены входы технического повтора')
    else:
        data=core.load_npz(core.OUTPUT/'holdout.npz')
        fixed=canonical_integers(data)
        identity={name:{'original_dtype':str(data[name].dtype),'canonical_dtype':str(fixed[name].dtype),
                        'numeric_values_identical':bool(np.array_equal(data[name],fixed[name]))}
                  for name in ['uid','timestamp']}
        core.write_json(path,{'fingerprint':fingerprint,'self_check':self_check(),
                   'actual_data_checks':identity,'created_at_utc':datetime.now(timezone.utc).isoformat()})
    from . import evaluate, rollout, audit
    original_loader=core.load_data
    def load_canonical(split,reason):
        return canonical_integers(original_loader(split,reason+'; guarded canonical integer adapter'))
    # Только входная репрезентация. Все исходные hash-guards продолжают выполняться.
    evaluate.load_data=load_canonical
    rollout.load_data=load_canonical
    audit.load_data=load_canonical
    if args.stage=='holdout':
        from .run import holdout_stage
        holdout_stage()
    elif args.stage=='audit':
        audit.run()
    else:
        from .report import main as report
        report()
    core.log_access('technical retry','Completed '+args.stage,**fingerprint)
    print('TECHNICAL RETRY COMPLETE',args.stage,flush=True)


if __name__=='__main__':
    main()
