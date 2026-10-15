"""Сформировать отчёт из завершённой оценки и независимого аудита."""
import json
import shutil
import zipfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .core import (OUTPUT, PROJECT, SOURCE, check_frozen, load_npz, macro, paired_interval,
                   protocol, read_json, sha, write_json)


def seed_summary(kind,selection,directory,data,models):
    chosen=selection[kind]['descriptor']
    comparisons=[]
    bases=[];policies=[]
    rows=[]
    for seed in protocol()['baseline']['seeds']:
        base_name=f'baseline_s{seed}'
        policy_name=(f'rl_{kind}_b{chosen["beta"]:g}_h1_s{seed}'
                     if chosen['name'].startswith('rl_') else base_name)
        b=load_npz(directory/(base_name+'.npz'))
        p=load_npz(directory/(policy_name+'.npz'))
        bases.append(b);policies.append(p)
        delta=models[policy_name]['ndcg@10']-models[base_name]['ndcg@10']
        comparisons.append(delta)
        rows.append({'seed':seed,'base':models[base_name]['ndcg@10'],
                    'policy':models[policy_name]['ndcg@10'],'difference':delta})
    result={'seeds':rows,'mean_difference':float(np.mean(comparisons)),
            'difference_std_across_seeds':float(np.std(comparisons,ddof=1)),
            'conditional_on_three_fitted_seed_models':True,'comparisons':{}}
    for metric in ['ndcg@10','recall@10','policy_expected_ratio','policy_expected_time_reward']:
        b=np.stack([a[metric] for a in bases]).mean(0)
        p=np.stack([a[metric] for a in policies]).mean(0)
        result['comparisons'][metric]=paired_interval(p,b,data['uid'])
    b=np.stack([a['ndcg@10'] for a in bases]).mean(0)
    p=np.stack([a['ndcg@10'] for a in policies]).mean(0)
    result['noninferiority']=paired_interval(p,b,data['uid'],factor=.95,one_sided=True)
    return result


def main():
    frozen=check_frozen()
    audit=read_json(OUTPUT/'completion_audit.json')
    if audit['status']!='passed':
        raise ValueError('Нет завершённого независимого аудита')
    directory=OUTPUT/'evaluations'/'holdout'
    evaluation=read_json(directory/'results.json')
    rollouts=read_json(OUTPUT/'rollouts'/'holdout'/'results.json')
    if not evaluation['completed'] or not rollouts['completed']:
        raise ValueError('Незавершённые результаты нельзя публиковать')
    meta=read_json(OUTPUT/'data_manifest.json')
    holdout=read_json(OUTPUT/'holdout_manifest.json')
    data=load_npz(OUTPUT/'holdout.npz')
    benchmark=read_json(OUTPUT/'benchmark.json')
    reward_selection=read_json(OUTPUT/'reward_selection.json')
    selection=frozen['selection']
    retention=read_json(OUTPUT/'retention_diagnostics.json')
    models=evaluation['models']
    seeds={kind:seed_summary(kind,selection,directory,data,models) for kind in ['ratio','time']}
    q=evaluation['reward_quality']
    base=models['baseline_s11']
    ratio_name=selection['ratio']['descriptor']['name']
    time_name=selection['time']['descriptor']['name']
    ratio,time_model=models[ratio_name],models[time_name]
    figures=OUTPUT/'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'figure.dpi':160})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for spec in frozen['descriptors']:
        if spec['seed']!=11:
            continue
        m=models[spec['name']]
        color={'ratio':'#2563eb','time':'#e97621','balanced':'#8b5cf6'}[spec['reward_type']]
        marker={'actor':'o','analytic':'D','rerank':'s','popularity':'x'}[spec['kind']]
        if spec['name'].startswith('baseline_'):marker='*';color='black'
        if spec['name'].startswith('sft_'):marker='^';color='#059669'
        for ax,key in zip(axes,['policy_expected_ratio','policy_expected_time_reward']):
            ax.scatter(m['ndcg@10'],m[key],color=color,marker=marker,s=45)
    for ax,title in zip(axes,['Ожидаемая модельная доля','Ожидаемый суррогат времени']):
        ax.axvline(.95*base['ndcg@10'],ls='--',color='gray',lw=.8)
        ax.set(xlabel='Macro-user NDCG@10',ylabel=title,title=title)
        ax.grid(alpha=.2)
    fig.text(.08,.02,'● обученные политики; ◆ аналитические; ■ пересортировка; ▲ SFT; ★ база. Синий: доля; оранжевый: время.',fontsize=8)
    fig.tight_layout(rect=(0,.07,1,1));fig.savefig(figures/'tradeoff.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4))
    for name,label in [('raw_calibration','До калибровки'),('calibration','После калибровки')]:
        rows=q[name]['bins']
        ax.plot([r['predicted_mean'] for r in rows],[r['observed_mean'] for r in rows],'o-',label=label)
    ax.plot([0,1],[0,1],'--',color='gray')
    ax.set(xlabel='Средний прогноз',ylabel='Средняя записанная доля',title='Калибровка: новые пользователи')
    ax.legend();fig.tight_layout();fig.savefig(figures/'calibration.png');plt.close(fig)
    h5_specs=[s for s in frozen['rollout_descriptors'] if '_h5_' in s['name'] and 'budget' not in s['name']]
    shown=['baseline_s11',ratio_name,time_name]+[s['name'] for s in h5_specs]
    shown=list(dict.fromkeys(shown))
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for name in shown:
        rows=rollouts['models'][name]['steps']
        for ax,key in zip(axes,['ratio','time_reward','repeat']):
            ax.plot([r['step'] for r in rows],[r[key]['mean'] for r in rows],'o-',label=name)
    for ax,title in zip(axes,['Модельная доля','Суррогат времени','Повторы в цепочке']):
        ax.set(xlabel='Шаг',ylabel=title,xticks=range(1,6));ax.grid(alpha=.2)
    axes[0].legend(fontsize=6);fig.tight_layout();fig.savefig(figures/'rollout.png');plt.close(fig)
    table='\n'.join(f"| {name} | {m['recall@10']:.5f} | {m['ndcg@10']:.5f} | {m['policy_expected_ratio']:.5f} | {m['policy_expected_duration_limited_seconds']:.2f} | {m['kl_to_reference']:.4f} | {m['catalog_coverage_at10']:.3f} |"
                     for name,m in models.items())
    quality_table='\n'.join(f"| {label} | {q[key]['mae']:.5f} | {q[key]['rmse']:.5f} |" for key,label in
              [('global_mean','Общее среднее'),('item_mean','Сглаженное среднее по треку'),('raw','MLP до калибровки'),('calibrated','MLP после калибровки')])
    seed_table='\n'.join(f"| {kind} | {row['seed']} | {row['base']:.6f} | {row['policy']:.6f} | {row['difference']:+.6f} |"
                        for kind,result in seeds.items() for row in result['seeds'])
    comparisons=[]
    for kind,result in seeds.items():
        c=result['comparisons']['ndcg@10'];ni=result['noninferiority']
        comparisons.append(f"{kind}: средняя разница NDCG@10 {c['difference']:+.6f}, парный 95%-й интервал [{c['ci95'][0]:+.6f}; {c['ci95'][1]:+.6f}]. "
             f"Контраст с 95% исходного NDCG: нижняя односторонняя 95%-я граница {ni['lower_one_sided95']:+.6f}; "
             f"сохранение порога {'подтверждено в рамках данной условной проверки' if ni['noninferiority_supported'] else 'не подтверждено'}. "
             f"Стандартное отклонение разниц между тремя training seeds: {result['difference_std_across_seeds']:.6f}.")
    comparison='\n\n'.join(comparisons)
    slices='\n'.join(f"| {name} | {base['slices'][name]['examples']} | {base['slices'][name]['users']} | {base['slices'][name]['ndcg@10']:.6f} | {ratio['slices'][name]['ndcg@10']:.6f} |"
                    for name in ['known','positive_ratio_gt50','nonpositive_ratio_le50','organic','nonorganic'] if base['slices'][name])
    rollout_table='\n'.join(f"| {name} | {r['discounted_ratio_return']['mean']:.5f} | {r['discounted_time_reward_return']['mean']:.5f} | {r['steps'][-1]['repeat']['mean']:.4f} | {r['steps'][-1]['context_score_shift']['mean']:+.5f} |"
                       for name,r in rollouts['models'].items())
    budget_rows=[]
    for seed,value in read_json(OUTPUT/'sft_budgets.json').items():
        actual=read_json(OUTPUT/'models'/f'sft_s{seed}'/'completed.json')['training_seconds']
        budget_rows.append(f"| {seed} | {value['steps']} | {value['target_training_seconds']:.2f} | {actual:.2f} |")
    budgets='\n'.join(budget_rows)
    retention_rows='\n'.join(f"| {gap} | {name} | {row['brier']:.5f} | {row['logloss']:.5f} | {row['negative_AP_user_weighted']:.5f} |"
                     for gap,r in retention.items() for name,row in r['periods']['C'].items())
    training=sum(read_json(p)['training_seconds'] for p in (OUTPUT/'models').glob('*/completed.json'))
    inference=sum(m['inference_and_metrics_seconds'] for m in models.values())
    rm_diff=q['mse_difference_vs_global']
    rm_claim=('меньшую MSE, чем общее среднее, с интервалом разницы целиком ниже нуля' if rm_diff['ci95'][1]<0
              else 'неподтверждённое преимущество по MSE над общим средним')
    ratio_gain=ratio['policy_expected_ratio']-base['policy_expected_ratio']
    time_gain=time_model['policy_expected_duration_limited_seconds']-base['policy_expected_duration_limited_seconds']
    behavior_text=[]
    for name in [ratio_name,time_name,'rl_ratio_b0_h1_s11','rl_time_b0_h1_s11']:
        m=models[name]
        behavior_text.append(f"`{name}`: изменение средней длительности @10 {m['mean_duration_seconds_at10']-base['mean_duration_seconds_at10']:+.2f} с; "
             f"доля десяти самых частых объектов {m['top10_items_exposure_share']:.2%} против {base['top10_items_exposure_share']:.2%}; "
             f"межпользовательское пересечение {m['between_user_overlap_at10']:.2%} против {base['between_user_overlap_at10']:.2%}; "
             f"изменение ожидаемой доли основного оценщика {m['policy_expected_ratio']-base['policy_expected_ratio']:+.5f}, "
             f"диагностического {m['policy_expected_auditor_ratio']-base['policy_expected_auditor_ratio']:+.5f}.")
    behavior_text='\n\n'.join(behavior_text)
    summary={'holdout':holdout,'selected':{'ratio':ratio_name,'time':time_name},'seed_results':seeds,
             'ratio_expected_reward_change':ratio_gain,'time_expected_seconds_change':time_gain,
             'reward_mse_difference_vs_global':rm_diff,'new_training_seconds':training,
             'holdout_inference_and_metrics_seconds':inference,
             'analytical_objective_gaps':{}}
    for kind in ['ratio','time']:
        chosen=selection[kind]['descriptor'];beta=chosen['beta']
        analytic=f'analytic_{kind}_b{beta:g}'
        if analytic in models:
            summary['analytical_objective_gaps'][kind]=models[analytic]['regularized_J']-models[chosen['name']]['regularized_J']
    write_json(OUTPUT/'summary.json',summary)
    content=f'''# RL-рекомендации на Yambda: проверка оптимизации и переноса между пользователями

## Результат и границы вывода

Обучены новые модели отклика, шесть RL-пилотов H=1, повторения выбранных настроек, SFT-контроли и политики H=5. Исходные GPT-2 seeds 11/22/33 переиспользованы после проверки данных, исходного кода и весов; их полное повторное обучение в этом запуске не выполнялось.

Окончательная проверка содержит {holdout['summary']['examples']} событий {holdout['summary']['users']} ранее неиспользованных пользователей. Их исходы прочитаны после фиксации моделей и кода. Это проверка переноса между пользователями в том же периоде D. Данные не дают нового независимого временного периода: за прежней границей доступны только {meta['unused_later_seconds']} секунд. Исходный D прежней когорты уже был просмотрен при разработке. Перенос на новый временной режим не установлен.

Reward model показала {rm_claim}: разница MSE {rm_diff['difference']:+.6f}, парный 95%-й интервал [{rm_diff['ci95'][0]:+.6f}; {rm_diff['ci95'][1]:+.6f}].

Для seed 11 выбранный RL по доле изменил ожидаемую модельную долю на {ratio_gain:+.5f} ({ratio_gain*100:+.2f} процентного пункта). RL по времени изменил ожидаемый суррогат времени на {time_gain:+.2f} с. Эти величины рассчитаны замороженной reward model, не измерены на показах новой политики.

{comparison}

## Данные и временной протокол

Используется Yambda-50M, версия `{meta['revision']}`, SHA-256 исходного Parquet `{meta['raw_sha256']}`. Каталог из 5000 объектов рассчитан только по A исходной когорты. История содержит последние 64 ID; неизвестные объекты представлены UNK. Неизвестные цели остаются в оценке и считаются промахами. Ранжируется полный допустимый каталог без sampled negatives; равные оценки упорядочиваются по token ID.

На A обучаются политики и модели отклика. B1 выбирает архитектуру и checkpoint reward model по macro-user MSE; из B1 исключены отклики, пересекающие его правую границу. На B2 обучается неотрицательная линейная калибровка с clip(0,1). C выбирает RL и пересортировку. После фиксации выполняется проверка новой когорты в D. Правила зрелости откликов, неоднозначных timestamp и ограничения числа целей записаны в `protocol.json` и `data_manifest.json`.

Исходные 1024 пользователя исключены из резервирования новой когорты. Зарезервированы следующие 512 ID в SHA-256 порядке; {meta['new_cohort_eligible']} соответствуют требованию 20 событий A. Отбор пригодности использовал только прошлую активность. Однако исходный Yambda сам отбирает пользователей по активности в раннем и последующем периодах, поэтому выводы об удержании ограничены этой популяцией. [Описание данных и отбора](https://arxiv.org/html/2505.22238v1).

## Reward model

MLP использует замороженное представление истории GPT-2, обучаемый embedding объекта и длительность. Проверены варианты без истории и без длительности. Для каждого варианта выполнены 4000 обновлений; выбран checkpoint с минимальной MSE на B1. Лучший вариант — `{reward_selection['selected']}`. Архитектура диагностического оценщика такая же; другой seed и исходная GPT-2 дают отдельную инициализацию. Общие ошибки данных и постановки могут сохраняться.

Обучение MSE и калибровка дают равный полный вклад пользователям. Baselines рассчитаны по тому же train-набору. Ошибки ниже относятся только к известным логированным объектам: {q['calibrated']['examples']} событий, {q['calibrated']['users']} пользователей.

| Предиктор | MAE | RMSE |
|---|---:|---:|
{quality_table}

![Калибровка](../output/research_v2/figures/calibration.png)

В калибровочных группах сохраняются веса полной популяции известных целей; пользователи не перевзвешиваются отдельно в каждом bin. Размеры групп, масса весов и верхний хвост прогнозов доступны в окончательном JSON. Модель оценивает условный записанный отклик. Без показов и вероятностей логирующей политики нельзя считать его причинным эффектом нового рекомендованного действия. [Ограничения модельной оценки политик](https://arxiv.org/abs/1503.02834).

## Политики, цели и метрики

GPT-2 имеет два блока, четыре головы, скрытую размерность 128, случайную исходную инициализацию и словарь item ID. REINFORCE сэмплирует четыре независимые цепочки на исходную историю. Используются leave-one-out baseline, замороженный оценщик, отключённый dropout, точный KL к исходной GPT-2 и его прямая производная; будущие KL входят в return. Корректность градиента проверена точным перечислением двухшагового примера. [REINFORCE](https://doi.org/10.1007/BF00992696).

Награды: R1 — откалиброванная доля; R2 — min(duration,cap)/cap × R1; R3 — R2 минус 0.1 за повтор внутри цепочки. R2 ограничивает длительность объекта, а не непосредственно фактическое время прослушивания. Повторы реальной истории разрешены. H=5 безусловно добавляет предложенный ID в контекст; отклик и прекращение сессии не моделируются.

Для H=1 проверена точная политика pi_star(a|h) proportional pi_ref(a|h) × exp(reward/beta), beta>0. Для фиксированного контекста J(pi)=beta log Z - beta KL(pi||pi_star); поэтому она максимизирует модельную цель среди произвольных распределений действий. Она не гарантирует сохранение NDCG, требует оценки полного каталога и не доказывает улучшение реальных откликов.

Пересортировка меняет оценки top-100 исходных объектов. При оценке ожидаемой награды её sampling-политика нормируется по этим 100 объектам; поэтому вес 0 сохраняет ранжирование, но не полное распределение исходной GPT-2. Аналитическая политика использует весь каталог. Выбранные на C веса: ratio={selection['rerank_ratio']['descriptor']['weight']}, time={selection['rerank_time']['descriptor']['weight']}.

Recall@K при одной цели равен индикатору попадания; NDCG@K — 1/log2(rank+1) при попадании и 0 иначе. Сначала метрики усредняются по событиям пользователя, затем по пользователям. Expected reward — полная сумма вероятности действия × его модельной награды. Reward@10 — отдельное равномерное среднее списка. J сравнивается только при одинаковых награде и beta. Для H5 его одноступенчатая диагностика не заменяет полный rollout return.

## Окончательная оценка

| Политика | Recall@10 | NDCG@10 | E[доля] | E[суррогат времени], с | KL | Покрытие @10 |
|---|---:|---:|---:|---:|---:|---:|
{table}

![Компромисс](../output/research_v2/figures/tradeoff.png)

| Награда | Seed | Базовый NDCG | RL NDCG | Разница |
|---|---:|---:|---:|---:|
{seed_table}

Основное сравнение — выбранный RL по доле против собственной базы. Парные интервалы основной усреднённой разницы вычислены по пользователям после усреднения трёх заранее заданных обученных seed-моделей. Они условны на этих весах и данном периоде. Разброс training seeds приведён отдельно; три seeds не считаются независимыми исследованиями. Остальные сравнения диагностические, интервал с нулём не доказывает эквивалентность.

| Срез | События | Пользователи | База NDCG@10 | RL по доле NDCG@10 |
|---|---:|---:|---:|---:|
{slices}

Популяции срезов различаются. Все прослушивания включают неудачные взаимодействия; срез доли более 50% даёт отдельную проверку воспроизведения положительного поведения. Ни одна из этих метрик сама по себе не измеряет удовлетворённость.

{behavior_text}

Изменения длительности и концентрации описывают поведение политики. Согласованный рост двух оценщиков не доказывает рост истинного отклика; расхождение между ними также не является само по себе доказательством эксплуатации ошибки. Для анализа персонализации используются концентрация, пересечение списков, популярность и релевантность совместно.

## Несколько шагов вперёд

На {protocol()['rollout']['users']} пользователях выполняются четыре траектории на каждого при трёх sampling seeds; длина 5 и gamma=0.9. Интервалы относятся к пользователям после усреднения траекторий. По каждому шагу доступны награда, KL, повторения и оценка того же действия на исходной истории. Сдвиг оценки — чувствительность к искусственному контексту, не измеренная ошибка на новых пользовательских откликах.

| Политика | Return доли | Return времени | Повтор на шаге 5 | Сдвиг оценки на шаге 5 |
|---|---:|---:|---:|---:|
{rollout_table}

![Пять шагов](../output/research_v2/figures/rollout.png)

Кроме полного H5-обучения сохранены стадия 200 обновлений с тем же числом сгенерированных действий, что H1, и ближайшая по накопленному времени стадия. Число действий и вычислительное время — разные бюджеты. Эти стадии выбирались по техническим затратам, без метрик качества. Исходный time-return, штрафованный time-return и регуляризованный return сохранены отдельно. Сравнение R3 с R2 и интервалы приведены в rollout JSON.

## Возвращение после конца сессии

Сессии определены разрывом более 30 или 60 минут. Момент определения конца сессии — последнее событие плюс этот разрыв. Цель — следующее взаимодействие до конца полного семидневного окна от последнего события. Незрелые окна исключены. Модель прошлой активности использует длину завершившейся сессии и число прошлых действий за день и неделю; преобразования и 500 обновлений логистической модели обучаются на A. Это описательный development-эксперимент; сигнал не использовался как RL-награда.

| Разрыв, с | Предиктор | Brier на C | LogLoss на C | AP отсутствия возвращения |
|---|---|---:|---:|---:|
{retention_rows}

AP рассчитан с macro-user весами для отрицательного исхода; равные оценки обрабатываются группами. Постоянный предиктор имеет AP, равный доле отрицательных исходов. Предсказуемость будущей активности не устанавливает влияние рекомендации. Для оптимизации удержания потребуются модель реакции и прекращения взаимодействия, валидные данные показов либо онлайн-проверка. [Связанный сессионный подход](https://arxiv.org/html/2607.02818v2).

## Вычисления и воспроизведение

Устройство новых запусков: {benchmark['device']}, PyTorch {benchmark['torch']}, четыре CPU-потока. Технический warm-cache замер: медиана RL-шага с повторным кодированием {benchmark['timings']['repeated_history']['median_seconds']:.5f} с, с общим вычислением исходной истории {benchmark['timings']['shared_history']['median_seconds']:.5f} с; ускорение {benchmark['rl_speedup']:.2f} раза. Проверка эквивалентности: максимальное расхождение градиентов {benchmark['equivalence']['gradient_max_error']:.3g}. Для технического замера использована случайно инициализированная reward-head; это не результат обучения политики.

Сумма новых обучающих обновлений: {training/60:.2f} мин; повторное исходное GPT-2-обучение, подготовка признаков, checkpoint I/O и оценка не включены. Время окончательного инференса и вычисления метрик: {inference/60:.2f} мин; первоначальная подготовка полных reward-score кешей учитывается в access log отдельно. Время записывается накопительно и восстанавливается из checkpoints. Вариации нагрузки устройства ограничивают точность сопоставления wall time.

| Seed | Шаги SFT | Целевое время RL, с | Фактическое время SFT, с |
|---|---:|---:|---:|
{budgets}

Основные команды из корня проекта после установки requirements:

```bash
.venv/bin/python -m research_v2.run prepare
.venv/bin/python -m research_v2.run checks
.venv/bin/python -m research_v2.run benchmark
.venv/bin/python -m research_v2.run reward
.venv/bin/python -m research_v2.run pilot
.venv/bin/python -m research_v2.run retention
.venv/bin/python -m research_v2.run training
.venv/bin/python -m research_v2.run freeze
.venv/bin/python -m research_v2.run holdout
.venv/bin/python -m research_v2.audit
.venv/bin/python -m research_v2.report
```

Для свежего окружения подготовьте исходные данные и GPT-2:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python download.py
.venv/bin/python prepare.py
for seed in 11 22 33; do
  .venv/bin/python experiment.py baseline --name "baseline_s$seed" --seed "$seed" --steps 6000 --batch 32 --lr 0.001 --save-every 250 --eval-every 1000
done
```

Не запускайте исходную test-оценку для нового подбора. Новый самостоятельный запуск выполняется в чистой копии проекта; завершённый эксперимент не перезаписывается. `--limit-steps` позволяет технически прервать обучение и продолжить с теми же настройками. Прерванная оценка возобновляется по завершённым моделям только при совпадении всех входных хешей.

Независимый audit пересчитывает метрики всех политик, проверяет каталог и хеши, воспроизводит три выбранные политики на CPU на 64 строках и проверяет численные reward-предсказания. Полный повтор обучения и перенос на другое оборудование могут давать численные различия. Никакой рост предсказанной награды не приравнивается к пользовательскому или причинному эффекту.

## Ограничения и следующий эксперимент

Исследование использует ограниченный популярный каталог, маленькую GPT-2 и отобранную активную популяцию. Точный порядок внутри пятисекундных групп неизвестен. Reward model проверена на известных логированных объектах, а многократное продолжение создаёт состояния без наблюдаемых истинных откликов. Диагностический оценщик разделяет данные и архитектуру с основным. Проверка новых пользователей не заменяет новый временной период. Для дальнейшего подтверждения нужны независимый будущий период, более широкий каталог с проверкой доступности объектов и онлайн-оценка реальных показов.
'''
    # Отчёт сохраняется отдельно; исходный README входит в неизменяемый снимок.
    relative_output=str(OUTPUT.relative_to(PROJECT))
    content=content.replace('../output/research_v2/','../'+relative_output+'/')
    (SOURCE/'README.md').write_text(content)
    package=OUTPUT/'research_v2_submission.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in SOURCE.glob('*'):
            if path.is_file() and path.suffix in ['.py','.json','.md']:
                if path.name=='README.md':
                    archive.writestr('research_v2/README.md',content.replace('../'+relative_output+'/',
                                      '../saved_results/research_v2/'))
                else:
                    archive.write(path,'research_v2/'+path.name)
        for path in PROJECT.glob('*.py'):
            archive.write(path,path.name)
        for filename in ['README.md','protocol.json','requirements.txt','.gitignore']:
            archive.write(PROJECT/filename,filename)
        archive.writestr('START_HERE.md','# RL-рекомендации: новая версия исследования\n\n'
            'Основной отчёт: [research_v2/README.md](research_v2/README.md). '
            'Сохранённые результаты находятся в saved_results/research_v2. '
            'Команды воспроизведения создают отдельный output/research_v2, поэтому архивные результаты не блокируют свежий запуск.\n')
        for path in (PROJECT/'output'/'figures').glob('*.png'):
            archive.write(path,'output/figures/'+path.name)
        for path in OUTPUT.rglob('*'):
            if not path.is_file() or 'cache' in path.parts or 'original_snapshot' in path.parts:
                continue
            if path.suffix in ['.json','.jsonl','.png'] or (path.suffix=='.npz' and
                    ('evaluations' in path.parts or 'rollouts' in path.parts)):
                archive.write(path,'saved_results/research_v2/'+str(path.relative_to(OUTPUT)))
    print('REPORT',SOURCE/'README.md','PACKAGE',package,flush=True)
    return summary


if __name__=='__main__':
    main()
