"""Воспроизводимое оформление отчёта, уточнений и архива без изменения моделей."""
import zipfile
from pathlib import Path

import numpy as np

from research_v2.core import (OUTPUT,PROJECT,SOURCE,check_frozen,read_json,load_npz,
                             paired_interval,sha,write_json,read_checkpoint,protocol)


def audit_budgets(frozen):
    cfg=protocol();rows=[]
    for spec in frozen['descriptors']:
        roles=spec.get('budget_role',[])
        if not roles or roles==['full_updates']:
            continue
        path=PROJECT/spec['actor'];state=read_checkpoint(path);settings=state['config']['spec']
        actions=state['step']*settings['batch']*settings['samples']*settings['horizon']
        row={'model':spec['name'],'roles':roles,'steps':state['step'],'sampled_actions':actions}
        if 'action_matched' in roles:
            expected=cfg['rl']['steps']*cfg['rl']['batch']*cfg['rl']['samples']
            if actions!=expected:raise ValueError('Не совпадает бюджет действий')
            row['h1_sampled_actions']=expected
        if 'time_matched' in roles:
            target=spec['h1_target_training_seconds'];actual=state['training_seconds']
            best=min(abs(read_checkpoint(p)['training_seconds']-target) for p in path.parent.glob('*.pt'))
            if abs(abs(actual-target)-best)>1e-10:raise ValueError('Выбрана не ближайшая временная стадия')
            row.update(target_seconds=target,actual_seconds=actual,
                  relative_time_difference_percent=(actual/target-1)*100,nearest_saved_stage=True)
        rows.append(row)
    write_json(OUTPUT/'budget_audit.json',{'status':'passed','action_budget_identity_verified':True,
            'nearest_time_stage_verified':True,'exact_wall_time_equivalence_claimed':False,'rows':rows})


def main():
    frozen=check_frozen()
    audit_budgets(frozen)
    if not (OUTPUT/'serving_benchmark.json').exists():
        from research_v2_serving import main as benchmark
        benchmark()
    results_sha=sha(OUTPUT/'evaluations/holdout/results.json')
    frozen_sha=sha(OUTPUT/'frozen.json')
    from research_v2.report import main as report_generator
    report_generator()
    report=SOURCE/'README.md'
    before=sha(report)
    text=report.read_text()
    directory=OUTPUT/'rollouts'/'holdout'
    balanced=next(s['name'] for s in frozen['rollout_descriptors'] if s['reward_type']=='balanced' and 'budget' not in s['name'])
    time_policy=next(s['name'] for s in frozen['rollout_descriptors'] if s['reward_type']=='time' and '_h5_' in s['name'] and 'budget' not in s['name'])
    a=load_npz(directory/(balanced+'.npz'))
    b=load_npz(directory/(time_policy+'.npz'))
    def per_user_step5(records):
        m=records['step']==5
        u,inv,n=np.unique(records['uid'][m],return_inverse=True,return_counts=True)
        return u,np.bincount(inv,weights=records['repeat'][m])/n
    ua,va=per_user_step5(a);ub,vb=per_user_step5(b)
    if not np.array_equal(ua,ub):
        raise ValueError('Непарные rollout-пользователи')
    repeat=paired_interval(va,vb,ua)
    write_json(OUTPUT/'rollout_repeat_diagnostic.json',{'status':'diagnostic',
                  'comparison':'R3 vs R2, H5, step5 repeat','models':[balanced,time_policy],**repeat})
    serving=read_json(OUTPUT/'serving_benchmark.json')
    if serving['source_sha256']!=sha(PROJECT/'research_v2_serving.py'):
        raise ValueError('Инференс-код изменён после замера')
    measurements=serving['measurements']
    rows=[]
    for name,label in [('baseline','База'),('rl_ratio','RL по доле'),('rl_time','RL по времени'),
                       ('analytic_ratio','Аналитическая: доля'),('analytic_time','Аналитическая: время')]:
        rows.append(f"| {label} | {measurements[name+'_batch1']['median_seconds']*1000:.2f} | "
                    f"{measurements[name+'_batch32']['median_seconds']*1000:.2f} |")
    text=text.replace('.venv/bin/python -m research_v2.run holdout\n.venv/bin/python -m research_v2.audit\n.venv/bin/python -m research_v2.report',
         '.venv/bin/python -c "from research_v2.data import prepare_holdout; prepare_holdout()"\n'
         '.venv/bin/python research_v2_runtime.py holdout\n.venv/bin/python research_v2_runtime.py audit\n'
         '.venv/bin/python research_v2_serving.py\n.venv/bin/python research_v2_finalize.py')
    text=text.replace('## Данные и временной протокол',
         '## Техническое восстановление окончательного прогона\n\n'
         'Первый прогон остановился при сравнении uint32 timestamp с int64 датами каталога. '
         'Reward-исходы к этому моменту были прочитаны, результаты первой политики ещё не рассчитаны. '
         'Приведение типов проверено на точное совпадение численных значений. '
         'После сохранения всех вычислений отдельно исправлен конфликт имени поля финального журналирования. '
         'Первоначальный frozen-манифест, вычислительные источники, данные и веса сохранены; '
         'дополнительные адаптеры имеют собственные хеши и журналы. '
         'Выбор моделей после доступа к этой выборке не менялся. '
         'Подробности: [TECHNICAL_NOTES.md](TECHNICAL_NOTES.md).\n\n## Данные и временной протокол')
    text+='\n## Дополнительные проверки стоимости и интерпретации\n\n'
    text+='Инференс измерен отдельно от метрик качества на фиксированных контекстах C: '
    text+='5 прогревочных и 20 измеряемых вызовов каждого метода, четыре CPU-потока. '
    text+=f"Устройство замера — {serving['device']}. Медианы ниже относятся ко всему batch, без загрузки весов, сети и затрат сервиса.\n\n"
    text+='| Метод | Batch 1, мс | Batch 32, мс |\n|---|---:|---:|\n'+'\n'.join(rows)+'\n\n'
    text+='`research_v2_serving.py` проецирует в item logits только последнее скрытое состояние. '
    text+='На 64 development-контекстах максимальное расхождение logits трёх акторов равно 0, top-20 совпадают полностью. '
    text+='Аналитическая политика разделяет один Transformer-проход с reward model: идентичность encoder-весов проверена. '
    text+='Замер включает вычисление награды полного каталога для аналитической политики; кеш результатов разных запросов не используется. '
    text+='Это технический warm-cache замер, а не подтверждённая задержка рабочего сервиса.\n\n'
    if (OUTPUT/'budget_audit.json').exists():
        budget=read_json(OUTPUT/'budget_audit.json')
        deviations=[r['relative_time_difference_percent'] for r in budget['rows'] if 'time_matched' in r['roles']]
        text+='Бюджет H5 при 200 обновлениях проверен независимо: 64 000 действий, как у H1 при 1000 обновлениях. '
        text+='Сопоставление времени приблизительное. Отклонения ближайших стадий R1/R2/R3: '
        text+=', '.join(f'{v:+.2f}%' for v in deviations)+'. '
        text+='Они не позволяют считать затраты строго равными; более частое сохранение стадий нужно в следующем заранее заданном эксперименте. '
        text+='Полные значения находятся в `budget_audit.json`.\n\n'
    text+=f"Дополнительный диагностический контраст повторений R3−R2 на шаге 5: {repeat['difference']:+.5f} "
    text+=f"({repeat['difference']*100:+.2f} процентного пункта), 95%-й интервал "
    text+=f"[{repeat['ci95'][0]:+.5f}; {repeat['ci95'][1]:+.5f}]. Он относится к искусственным траекториям.\n\n"
    text+='Предиктор возвращения использует прошлую активность, без ID предлагаемого объекта. '
    text+='Добавление p_return(h) к одноступенчатой награде не меняет предпочтение действий: '
    text+='это одинаковый для всех a член, а сумма производных вероятностей действий равна нулю. '
    text+='Для направленной оптимизации возвращения нужен сигнал, зависящий от действия и реакции, '
    text+='например p_return(h,a,feedback), с отдельно проверенными моделями отклика и прекращения сессии.\n\n'
    text+='Награды первоначальной и новой версий напрямую несопоставимы: изменились оценщик, калибровка, '
    text+='популяция и переход от reward@10 к полной ожидаемой награде. '
    text+='Первое наблюдение объекта в A — консервативный прокси его доступности, без известной даты появления в каталоге. '
    text+='Задержки исходного логирования не документированы полностью; timestamp-проверки не заменяют знания эксплуатационного процесса.\n'
    text+='\nБлизкая постановка beyond-accuracy целей рассмотрена в '
    text+='[Petrov и Macdonald, Aligning GPTRec with Beyond-Accuracy Goals with Reinforcement Learning](https://arxiv.org/html/2403.04875v1). '
    text+='В ней применяются PPO и обучение через teacher; текущий эксперимент использует прямое next-item обучение и REINFORCE, без заявления о воспроизведении той статьи.\n'
    report.write_text(text)
    if sha(OUTPUT/'evaluations/holdout/results.json')!=results_sha or sha(OUTPUT/'frozen.json')!=frozen_sha:
        raise ValueError('Оформление изменило эксперимент')
    write_json(OUTPUT/'presentation_revision.json',{'source_sha256':sha(Path(__file__)),
            'reason':'Technical replay disclosure, corrected commands, separate serving benchmark, qualified diagnostics',
            'generated_readme_sha256':before,'final_readme_sha256':sha(report),
            'model_results_sha256_unchanged':results_sha,'frozen_sha256_unchanged':frozen_sha})
    # Не перезаписывать исследовательские результаты; заменить лишь представление архива.
    relative=OUTPUT.relative_to(PROJECT).as_posix()
    package=OUTPUT/'research_v2_submission.zip';tmp=package.with_name(package.name+'.tmp')
    replacements={'research_v2/README.md':text.replace('../'+relative+'/',
                         '../saved_results/research_v2/').encode(),
                  'research_v2/TECHNICAL_NOTES.md':(SOURCE/'TECHNICAL_NOTES.md').read_bytes()}
    for name in ['research_v2_runtime.py','research_v2_serving.py','research_v2_finalize.py']:
        replacements[name]=(PROJECT/name).read_bytes()
    for name in ['serving_benchmark.json','budget_audit.json','rollout_repeat_diagnostic.json',
                 'presentation_revision.json','runtime_driver.json','technical_retry.json','completion_audit.json']:
        if (OUTPUT/name).exists():
            replacements['saved_results/research_v2/'+name]=(OUTPUT/name).read_bytes()
    with zipfile.ZipFile(package) as old,zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as new:
        for item in old.infolist():
            if item.filename not in replacements:
                new.writestr(item,old.read(item.filename))
        for name,value in replacements.items():
            new.writestr(name,value)
    tmp.replace(package)
    print('FINAL REPORT',report,'ARCHIVE',package,flush=True)


if __name__=='__main__':
    main()
