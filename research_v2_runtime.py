"""Завершение замороженного v2 с зарегистрированными техническими адаптерами.

Приведение целочисленной схемы находится в research_v2.technical_retry.
Здесь устраняется только конфликт имени reason при финальном журналировании.
Численные вычисления и первоначальные frozen-источники не изменяются.
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from research_v2 import core, technical_retry


def compatible_logger(original):
    def wrapped(resource, stage_reason, **details):
        details=dict(details)
        if 'reason' in details:
            details['technical_repair_reason']=details.pop('reason')
        return original(resource,stage_reason,**details)
    return wrapped


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['holdout','audit','report'])
    args=parser.parse_args()
    core.check_frozen()
    fingerprint={'driver_sha256':core.sha(Path(__file__)),
                 'frozen_sha256':core.sha(core.OUTPUT/'frozen.json'),
                 'integer_adapter_sha256':core.sha(Path(technical_retry.__file__)),
                 'holdout_sha256':core.sha(core.OUTPUT/'holdout.npz'),
                 'change':'Only move duplicate final-log reason keyword into technical_repair_reason; no numerical model changes',
                 'original_observed_failure':'TypeError in final log_access after all 32 policies and 17 rollout sets had been saved'}
    manifest_path=core.OUTPUT/'runtime_driver.json'
    if manifest_path.exists():
        if core.read_json(manifest_path)['fingerprint']!=fingerprint:
            raise ValueError('Изменён технический драйвер или его входы')
    else:
        captured=[]
        def capture(resource,reason,**kwargs):
            captured.append((resource,reason,kwargs))
        test=compatible_logger(capture)
        test('check','stage',reason='repair',value=3)
        if captured!=[('check','stage',{'technical_repair_reason':'repair','value':3})]:
            raise ValueError('Ошибка проверки логгера')
        core.write_json(manifest_path,{'fingerprint':fingerprint,'self_check_passed':True,
                     'created_at_utc':datetime.now(timezone.utc).isoformat()})
    core.log_access=compatible_logger(core.log_access)
    sys.argv=['research_v2.technical_retry',args.stage]
    technical_retry.main()


if __name__=='__main__':
    main()
