"""Создать отчёт и графики только из завершённой окончательной оценки."""
import json
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).parent/'.tmp'/'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiment import OUT, ROOT, write_json


def read(path):
    return json.loads(Path(path).read_text())


def fmt(x, digits=4):
    return f'{x:.{digits}f}'


def main():
    result = read(OUT/'test_evaluation.json')
    assert result.get('completed'), 'Итоговая оценка не завершена'
    assert read(OUT/'completion_audit.json')['status'] == 'passed'
    meta = read(OUT/'data_manifest.json')
    frozen = read(OUT/'frozen_selection.json')
    benchmark = read(OUT/'benchmark.json')
    quality = {name:read(OUT/name/'test.json') for name in ['reward_s11','reward_s22']}
    models = result['models']
    choice = frozen['selection']
    base = models['baseline_s11']
    ratio_name, time_name = choice['ratio']['name'], choice['time']['name']
    ratio, time_model = models[ratio_name], models[time_name]
    figures = OUT/'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.dpi':160})

    fig, axes = plt.subplots(1,2,figsize=(12,5.2))
    for ax, metric, title in zip(axes,['predicted_ratio_at10','normalized_time_reward_at10'],
                                ['Доля прослушивания','Ограниченное время прослушивания']):
        number = 0
        for name, m in models.items():
            if name.endswith('_s11') and name.startswith('rl_'):
                number += 1
                marker = 's' if '_h5_' in name else 'o'
                kind = name.split('_')[1]
                color = {'ratio':'#2563eb','time':'#e97621','balanced':'#8b5cf6'}[kind]
                ax.scatter(m['ndcg@10'],m[metric],marker=marker,color=color,s=45)
                offset = {8:(-14,-14),9:(5,7),2:(-14,2),3:(7,9),6:(-14,3)}.get(number,(4,4))
                ax.annotate(str(number),(m['ndcg@10'],m[metric]),fontsize=8,xytext=offset,
                            textcoords='offset points',arrowprops={'arrowstyle':'-','lw':.4,'color':'gray'})
        ax.scatter(base['ndcg@10'],base[metric],marker='*',color='black',s=130,label='Исходная GPT-2')
        sft = models['sft_extra_s11']
        ax.scatter(sft['ndcg@10'],sft[metric],marker='^',color='#64748b',s=70,label='Дополнительное SFT')
        rerank_name = f"rerank_weight{choice['rerank_weight']:g}"
        rr = models[rerank_name]
        ax.scatter(rr['ndcg@10'],rr[metric],marker='D',color='#16a34a',s=55,label='Пересортировка')
        ax.axvline(.95*base['ndcg@10'],color='gray',ls='--',lw=.8,label='95% исходного NDCG')
        ax.set(xlabel='NDCG@10, среднее по пользователям',ylabel='Модельная награда @10',title=title)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.tight_layout(rect=(0,.17,1,1))
    fig.text(.04,.115,'1: R1, β=0, H=1      2: R1, β=0.1, H=1      3: R1, β=1, H=1',fontsize=8)
    fig.text(.04,.075,'4: R2, β=0, H=1      5: R2, β=0.1, H=1      6: R2, β=1, H=1',fontsize=8)
    fig.text(.04,.035,'7: R1, β=0.1, H=5   8: R2, β=0.1, H=5   9: R3, β=0.1, H=5',fontsize=8)
    fig.savefig(figures/'tradeoff.png')
    plt.close(fig)

    q = quality['reward_s11']
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    cal = q['calibration']
    axes[0].plot([v['predicted_mean'] for v in cal],[v['observed_mean'] for v in cal],'o-')
    axes[0].plot([0,1],[0,1],'--',color='gray')
    axes[0].set(xlabel='Средний прогноз',ylabel='Средняя фактическая доля',title='Калибровка на известных test-объектах')
    bars = [q[k]['mae'] for k in ['global_mean','item_mean','reward_model']]
    axes[1].bar(['Общее среднее','Среднее по треку','Reward model'],bars,color=['#94a3b8','#64748b','#2563eb'])
    axes[1].set(ylabel='MAE',title='Ошибка прогноза доли прослушивания')
    fig.tight_layout()
    fig.savefig(figures/'reward_quality.png')
    plt.close(fig)

    fig, axes = plt.subplots(1,4,figsize=(14,4))
    shown = ['baseline_s11','rl_ratio_b0.1_h5_s11','rl_time_b0.1_h5_s11','rl_balanced_b0.1_h5_s11']
    labels = ['Исходная','RL: доля, H=5','RL: время, H=5','RL: время и повторы, H=5']
    for name,label in zip(shown,labels):
        rows = models[name]['rollout']
        for ax,key in zip(axes,['predicted_ratio','predicted_capped_seconds','repeat_in_rollout','kl_to_reference']):
            ax.plot([r['step'] for r in rows],[r[key] for r in rows],'o-',label=label)
    for ax,title,ylabel in zip(axes,['Доля прослушивания','Время прослушивания','Повторы внутри цепочки','Отклонение от исходной'],
                               ['Предсказанная доля','Предсказанное время, с','Доля повторов','KL']):
        ax.set(xlabel='Шаг генерации',ylabel=ylabel,title=title,xticks=range(1,6))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures/'rollout.png')
    plt.close(fig)

    fig, axes = plt.subplots(1,3,figsize=(11,4))
    names = ['baseline_s11',ratio_name,time_name]
    labels = ['Исходная','RL: доля','RL: время']
    for ax,key,title in zip(axes,['mean_duration_seconds','catalog_coverage_at10','top10_items_exposure_share'],
                            ['Средняя длительность, с','Покрытие каталога @10','Доля 10 самых частых объектов']):
        ax.bar(labels,[models[n][key] for n in names],color=['#64748b','#2563eb','#e97621'])
        ax.set_title(title,fontsize=10)
    fig.tight_layout()
    fig.savefig(figures/'behavior.png')
    plt.close(fig)

    rows = []
    for name,m in models.items():
        if name.endswith('_s11') or name.startswith('rerank_') or name=='popularity':
            rows.append(f"| {name} | {fmt(m['recall@10'])} | {fmt(m['ndcg@10'])} | {fmt(m['ndcg@10_known'])} | {fmt(m['predicted_ratio_at10'])} | {fmt(m['predicted_capped_seconds_at10'],1)} | {fmt(m['catalog_coverage_at10'])} |")
    table = '\n'.join(rows)
    seed_rows = []
    seed_results = {}
    for kind in ['ratio','time']:
        beta = choice[kind]['beta']
        values = []
        diffs = []
        for seed in [11,22,33]:
            name = f'rl_{kind}_b{beta:g}_h1_s{seed}'
            b = models[f'baseline_s{seed}']['ndcg@10']
            v = models[name]['ndcg@10']
            values.append(v)
            diffs.append(v-b)
            seed_rows.append(f'| {kind} | {seed} | {fmt(b)} | {fmt(v)} | {fmt(v-b)} |')
        seed_results[kind] = {'mean_ndcg':float(np.mean(values)), 'std_across_seeds':float(np.std(values,ddof=1)),
                              'mean_paired_difference':float(np.mean(diffs))}
    seed_table = '\n'.join(seed_rows)
    rm_rows = '\n'.join(f"| {label} | {fmt(q[key]['mae'])} | {fmt(q[key]['rmse'])} |" for key,label in
                        [('global_mean','Общее среднее train'),('item_mean','Среднее по треку из train'),('reward_model','Reward model')])
    selected_rows = []
    for label,m in [('RL по доле',ratio),('RL по времени',time_model)]:
        ci = m['paired_comparison']['ndcg@10']
        change = m['ndcg@10']-base['ndcg@10']
        relative = 100*change/base['ndcg@10']
        selected_rows.append(f"{label}: изменение NDCG@10 {change:+.6f} ({relative:+.1f}%), парный 95%-й интервал [{ci['ci95'][0]:+.6f}; {ci['ci95'][1]:+.6f}].")
    comparison = ' '.join(selected_rows)
    rr = models[f"rerank_weight{choice['rerank_weight']:g}"]
    sft = models['sft_extra_s11']
    control_text = (f"Дополнительное обычное обучение: NDCG@10 {sft['ndcg@10']:.4f}, модельная доля @10 {sft['predicted_ratio_at10']:.4f}. "
                    f"Пересортировка: NDCG@10 {rr['ndcg@10']:.4f}, модельная доля @10 {rr['predicted_ratio_at10']:.4f}. "
                    f"Для выбранного RL по доле соответствующие значения {ratio['ndcg@10']:.4f} и {ratio['predicted_ratio_at10']:.4f}.")
    if rr['ndcg@10'] >= ratio['ndcg@10'] and rr['predicted_ratio_at10'] >= ratio['predicted_ratio_at10']:
        control_text += ' Пересортировка не хуже выбранного RL по обеим точечным метрикам. В этом сравнении необходимость RL для такого улучшения не подтверждена; статистическая и вычислительная эквивалентность не заявляется.'
    rollout_table = '\n'.join(f"| {name} | {fmt(models[name]['discounted_rollout_ratio_return'])} | {fmt(models[name]['discounted_rollout_time_return'])} | {fmt(models[name]['rollout'][-1]['repeat_in_rollout'])} | {fmt(models[name]['step5_context_score_shift'])} |" for name in shown)
    ratio_gain = ratio['predicted_ratio_at10']-base['predicted_ratio_at10']
    time_gain = time_model['predicted_capped_seconds_at10']-base['predicted_capped_seconds_at10']
    duration_change_ratio = ratio['mean_duration_seconds']-base['mean_duration_seconds']
    duration_change_time = time_model['mean_duration_seconds']-base['mean_duration_seconds']
    rm_better = q['reward_model']['rmse'] < q['item_mean']['rmse']
    rm_conclusion = ('Reward model имеет меньшую RMSE, чем среднее по треку.' if rm_better else
                     'Reward model не превосходит среднее по треку по RMSE; это ограничивает основания доверять оптимизируемой награде.')
    behavior_text = []
    for label,m in [('RL по доле',ratio),('RL по времени',time_model)]:
        reward_delta = m['predicted_ratio_at10']-base['predicted_ratio_at10']
        auditor_delta = m['auditor_predicted_ratio_at10']-base['auditor_predicted_ratio_at10']
        behavior_text.append(f"{label}: покрытие {m['catalog_coverage_at10']:.1%} против {base['catalog_coverage_at10']:.1%}; доля десяти наиболее частых объектов {m['top10_items_exposure_share']:.1%} против {base['top10_items_exposure_share']:.1%}; среднее пересечение списков разных пользователей {m['between_user_overlap_at10']:.1%} против {base['between_user_overlap_at10']:.1%}. Изменение предсказанной доли основного оценщика {reward_delta:+.4f}, диагностического {auditor_delta:+.4f}.")
        if reward_delta > 0 and auditor_delta <= 0:
            behavior_text.append('Рост оценки основной reward model не повторяется у диагностической. Это согласуется с использованием особенностей конкретного оценщика, но без реальных откликов на новые рекомендации не доказывает эксплуатацию ошибки.')
    behavior_text = '\n\n'.join(behavior_text)
    no_kl_ratio, no_kl_time = models['rl_ratio_b0_h1_s11'], models['rl_time_b0_h1_s11']
    no_kl_text = (f"Без KL при R1 средняя длительность снизилась с {base['mean_duration_seconds']:.1f} до {no_kl_ratio['mean_duration_seconds']:.1f} с, "
                 f"а предсказанная доля выросла с {base['predicted_ratio_at10']:.4f} до {no_kl_ratio['predicted_ratio_at10']:.4f}; "
                 f"NDCG@10 изменился на {(no_kl_ratio['ndcg@10']/base['ndcg@10']-1)*100:+.1f}%. "
                 f"Без KL при R2 длительность выросла до {no_kl_time['mean_duration_seconds']:.1f} с, "
                 f"а NDCG@10 изменился на {(no_kl_time['ndcg@10']/base['ndcg@10']-1)*100:+.1f}%. "
                 f"Десять наиболее часто рекомендуемых объектов заняли {no_kl_time['top10_items_exposure_share']:.1%} всех позиций @10 против {base['top10_items_exposure_share']:.1%} у исходной модели; "
                 f"среднее межпользовательское пересечение списков выросло с {base['between_user_overlap_at10']:.1%} до {no_kl_time['between_user_overlap_at10']:.1%}.")
    penalty_time = models['rl_time_b0.1_h5_s11']
    penalty_balanced = models['rl_balanced_b0.1_h5_s11']
    repeat_text = (f"Добавление штрафа R3 уменьшило долю повторов на пятом шаге с {penalty_time['rollout'][-1]['repeat_in_rollout']:.1%} до {penalty_balanced['rollout'][-1]['repeat_in_rollout']:.1%}. "
                   f"При этом дисконтированная сумма награды времени снизилась с {penalty_time['discounted_rollout_time_return']:.4f} до {penalty_balanced['discounted_rollout_time_return']:.4f}. "
                   'Штраф уменьшает зацикливание в модельном процессе ценой части исходной награды, но не устраняет повторы полностью.')
    relevance_change = (ratio['ndcg@10']/base['ndcg@10']-1)*100
    if ratio_gain > 0 and relevance_change >= -5:
        main_conclusion = 'Для выбранного варианта по доле найден рост модельной награды при точечном снижении NDCG@10 не более 5% относительно исходной модели на D. Это ограниченное офлайн-наблюдение, а не доказательство улучшения сервиса.'
    elif ratio_gain > 0:
        main_conclusion = 'Выбранный вариант по доле повысил модельную награду, но не сохранил заранее заданный порог 95% исходного NDCG@10 на D. Рост оптимизируемого суррогата сопровождается потерей релевантности.'
    else:
        main_conclusion = 'Выбранный вариант по доле не показал роста основной модельной награды на D. Оптимизация на обучающих контекстах не гарантировала перенос на будущий период.'
    stages = {}
    for path in OUT.glob('*/history.json'):
        history = read(path)
        if history:
            stages[path.parent.name] = history[-1]['active_seconds_this_session']
    total = sum(stages.values())
    protocol_table = '\n'.join(f"| {name} | {v['examples']} | {v['users']} | {v['known_target_fraction']:.1%} |" for name,v in meta['splits'].items())
    content = f'''# RL-дообучение последовательной рекомендательной модели на Yambda

## Основные результаты

Обучена компактная рекомендательная GPT-2, проверена модель вознаграждения и проведено RL-дообучение с несколькими функциями награды. Итоговая проверка использует будущий временной период: число целей — {meta['splits']['test']['examples']}, число пользователей — {meta['splits']['test']['users']}. Экспериментальные результаты ниже получены в локальных запусках на Apple M5 с 16 ГБ памяти.

Для выбранного по validation варианта RL по доле прослушивания предсказанная доля в первых десяти рекомендациях изменилась на {ratio_gain:+.4f}; для варианта по времени предсказанное ограниченное время изменилось на {time_gain:+.1f} секунды на объект. Эти значения получены от reward model, а не от пользователей новой политики. {comparison}

{rm_conclusion} Независимо от роста модельной награды, эксперимент не доказывает улучшение удовлетворённости или возвращаемости пользователей.

{main_conclusion}

## 1. Постановка и данные

Использованы реальные прослушивания Yambda-50M. Для воспроизводимости зафиксированы версия `{meta['revision']}` и SHA-256 исходного Parquet; файл содержит {meta['raw_rows']:,} событий. По хешу анонимного ID выбраны 1024 пользователя без использования их будущих откликов. В этих исходных историях {meta['selected_rows']:,} событий. После требования не менее 20 событий в train к эксперименту допущены {meta['eligible_users']} пользователей.

Каталог состоит из 5000 наиболее частых треков **только обучающего периода**. Он покрывает {meta['train_catalog_event_coverage']:.1%} событий train. Для вычислительной доступности отобрано до 200 обучающих и до 24 оценочных целей на пользователя в каждом периоде. Это исследование ограниченного популярного каталога; результаты нельзя переносить на весь музыкальный сервис.

Объекты вне каталога представлены UNK в истории. В policy validation и test такие цели сохранены и считаются промахами. Метрики «known» вычислены отдельно, только для известных целей, с усреднением по пользователям, имеющим такие цели. Поэтому их популяция и знаменатель отличаются от основных метрик.

| Период | Примеры | Пользователи | Цели в каталоге |
|---|---:|---:|---:|
{protocol_table}

## 2. Временной протокол и защита от утечек

Общий интервал разделён по времени в отношении 70/10/10/10: A — обучение, B — выбор reward model, C — выбор настроек RL, D — окончательная проверка. Границы в анонимной шкале секунд: {meta['temporal_cuts']}. Время округлено до кратных пяти секунд; календарные даты скрыты.

Вход содержит последние 64 предшествующих ID объектов. Текущий и будущие отклики в признаки не входят. Длительность, популярность и словарь рассчитаны по A. При генерации разрешены объекты, впервые наблюдавшиеся в A раньше времени контекста; это консервативная замена отсутствующим датам доступности каталога. Из A/B исключены цели, чьё оценённое прослушивание могло пересечь правую границу периода.

Цели с неоднозначным порядком при одинаковом timestamp исключены как в конце окна, так и внутри all-position обучения. Таких событий в отобранных историях {meta['tied_target_events_excluded']:,}. В прошлой истории объекты одной временной группы упорядочены детерминированно по ID; точный порядок внутри группы неизвестен. История сохраняет исходные события, включая UNK и повторные прослушивания.

Базовые модели обучаются фиксированное число шагов. Reward model выбирается по MAE на B. Функция выбора RL и сетка параметров заданы в `protocol.json`; перед D сохранены выбранные настройки, хеши checkpoints и исходного кода. На D веса не обновляются, гиперпараметры не выбираются. При последовательной оценке контекст может содержать уже произошедшие реальные события D.

## 3. Архитектуры и обучение

Использована готовая `GPT2LMHeadModel` из Hugging Face Transformers с новым словарём ID объектов и случайной инициализацией. Архитектура: 2 блока, 4 головы внимания, скрытая размерность 128, контекст 64, всего {benchmark['parameters']:,} параметров. Текстовые предобученные веса не использованы. Базовое обучение: next item prediction на всех допустимых позициях, cross-entropy, AdamW, learning rate 0.001, batch 32, 6000 шагов, dropout 0.1. PAD, UNK и неоднозначные временные цели не участвуют в потере. Seeds: 11, 22, 33.

Reward model получает замороженное представление истории от обученной GPT-2, обучаемое представление трека и логарифм его известной длительности. MLP имеет скрытые слои 128 и 64 и sigmoid-выход. Цель — средняя ограниченная доля прослушивания, MSE, AdamW, learning rate 0.001, batch 256, 4000 шагов. Выход обучен на **записанных прослушиваниях известных объектов**, а не на полном наборе показов. Неизвестный или непрослушанный объект не получает искусственную отрицательную метку.

Доля ограничена диапазоном [0,1]; значения выше 100% сохранены в исходном файле, но для этой цели обрезаны. В train их доля {meta['train_ratio_above100_fraction']:.2%}. Длительность каталоговых объектов от 40 до 505 секунд; ограничение времени D = {meta['duration_cap_seconds']:.0f} секунд, 95-й процентиль длительности каталога из A.

### Качество reward model на D

MAE — средняя абсолютная ошибка доли; RMSE — корень из средней квадратичной ошибки. Сначала ошибки усредняются внутри пользователя, затем по пользователям. Проверяются только известные логированные цели.

Объём этой проверки: число взаимодействий — {q['reward_model']['n']}, число пользователей — {q['reward_model']['users']}. При трактовке доли как процентов MAE 0.01 соответствует ошибке в один процентный пункт.

| Предиктор | MAE | RMSE |
|---|---:|---:|
{rm_rows}

![Качество модели вознаграждения](output/figures/reward_quality.png)

Срезы по длительности, популярности и источнику взаимодействия, а также размеры калибровочных групп находятся в `output/reward_s11/test.json`. Второй предиктор `reward_s22` использует другую базовую инициализацию и служит диагностическим оценщиком. Он не используется для обучения или выбора RL и также может иметь общие с основной моделью ошибки.

## 4. RL и функции вознаграждения

Метод — on-policy REINFORCE на сгенерированных действиях с замороженной reward model. Из одного реального контекста сэмплируются четыре независимые цепочки. Leave-one-out baseline для каждой цепочки не включает её собственную награду. Использованы learning rate 0.00005, 1000 шагов, 16 исходных историй на batch, градиентное ограничение нормы 1, температура 1 и полное распределение доступных объектов. Dropout отключён при sampling и вычислении вероятностей.

Сравниваются три награды: R1 = предсказанная доля; R2 = min(длительность,D)/D × предсказанная доля; R3 = R2 − 0.1 при повторе объекта внутри генерируемой цепочки. R2 — ограниченный суррогат времени, который сознательно ограничивает выгоду от очень длинного контента. Повтор ранее прослушанного трека не запрещён: в музыке он может быть релевантным.

RL использует KL-штраф к исходной GPT-2: β = 0, 0.1, 1 для горизонта 1; β = 0.1 для горизонта 5. Будущие KL-штрафы входят в return, а прямая производная штрафа также учитывается. Для пяти шагов коэффициент дисконтирования γ = 0.9 уменьшает вклад отдалённых наград. Здесь «on-policy» относится к sampling в искусственном модельном процессе; реальные пользователи с новой политикой не взаимодействовали.

## 5. Окончательное сравнение

При одном правильном следующем объекте Recall@K равен индикатору попадания. NDCG@K равен 1/log₂(rank+1) при попадании и 0 при промахе. Основные результаты усреднены сначала по событиям пользователя, затем по пользователям. Кандидаты одинаковы для всех моделей; ранжируется весь закрытый каталог, без подмешивания случайных отрицательных объектов.

«Доля @10» — средняя оценка десяти первых объектов reward model; это не средняя фактическая доля на показах. «Время @10» — аналогичная модельная оценка ограниченного времени в секундах. Покрытие — число различных рекомендованных треков, делённое на 5000.

| Модель | Recall@10 | NDCG@10 | NDCG@10 known | Доля @10 | Время @10, с | Покрытие @10 |
|---|---:|---:|---:|---:|---:|---:|
{table}

В именах моделей `ratio` означает долю, `time` — время, `balanced` — время со штрафом за повторы; `b` задаёт β, `h` — число шагов генерации, `s` — seed. Seed фиксирует последовательность случайных чисел и позволяет повторить инициализацию и sampling.

Контроль `sft_extra_s11` продолжает обычное обучение исходной модели с меньшим learning rate; число дополнительных шагов ({frozen['extra_sft_steps']}) выбрано по измеренному времени RL и полному базовому обучению, без использования D. Это приблизительное сопоставление времени, а не точное равенство FLOPs. Пересортировка применяется к 100 лучшим кандидатам исходной модели по log-probability + вес × reward; вес выбран на C. `popularity` использует частоты A.

{control_text}

![Компромисс релевантности и награды](output/figures/tradeoff.png)

Для каждой из R1/R2 на C выбирается максимальная награда среди вариантов, сохраняющих не менее 95% базового NDCG@10. Если допустимых вариантов нет, выбирается максимальный NDCG и фиксируется нарушение ограничения. Выбранные варианты: `{ratio_name}` и `{time_name}`. Наблюдаемое соблюдение порога на C не гарантирует его соблюдение на D.

{comparison} Интервалы получены 2000 парными пересэмплированиями пользователей на одном фиксированном периоде. Они не охватывают неопределённость новых периодов или обучения. Интервал, включающий ноль, не доказывает эквивалентность моделей. Остальные варианты и срезы — диагностические сравнения; универсальные заявления по множеству проверок не делаются.

Для выбранного R1 точечное снижение укладывается в 5%, но нижняя граница интервала допускает более сильное ухудшение. Поэтому соблюдение этого порога по точечной оценке не является доказанной гарантией сохранения релевантности.

### Повторения выбранных настроек

| Награда | Seed | Базовый NDCG@10 | RL NDCG@10 | Разница с собственной базой |
|---|---:|---:|---:|---:|
{seed_table}

Три seeds показывают чувствительность к обучению. Это повторения на одном разбиении и одной reward model, а не три независимых исследования. Средние и стандартные отклонения приведены в `output/summary.json`.

## 6. Ответы на исследовательские вопросы

**Поведение после RL.** Изменение среднего времени самих рекомендуемых треков: R1 {duration_change_ratio:+.1f} с, R2 {duration_change_time:+.1f} с относительно исходной модели. Изменение предсказанной доли при R1 {ratio_gain:+.4f}; ограниченного времени при R2 {time_gain:+.1f} с. Эти измерения позволяют проверить гипотезу предпочтения коротких/длинных объектов; такое предпочтение нельзя объявлять заранее установленным механизмом.

**Релевантность и персонализация.** Изменение точности описано выше. Помимо покрытия измерены доля десяти наиболее часто рекомендуемых объектов, энтропия распределения рекомендаций, средняя обучающая популярность и пересечение списков разных пользователей. Высокая концентрация или сходство списков согласуются с ослаблением персонализации. Низкое покрытие само по себе не доказывает коллапс: оно оценивается вместе с исходной моделью, распределением популярности и релевантностью.

{behavior_text}

{no_kl_text} Наблюдения согласуются с оптимизацией особенностей выбранной награды: R1 увеличивает предпочтение коротких объектов с высокой оценкой доли, а R2 явно умножает её на длительность. Это объяснение модельного поведения; причинный эффект длительности на реальные предпочтения не установлен. Рост повторных рекомендаций знакомого контента не является обязательным следствием RL: доля ранее прослушанных объектов также измерена и не должна подменяться предположением.

![Изменения поведения](output/figures/behavior.png)

**Долгосрочные цели.** Построен описательный сигнал наличия взаимодействия между первым и седьмым днём после события. Используются только окна, полностью заканчивающиеся внутри соответствующего периода. В A доля таких окон {meta['return_proxy']['train']['fraction']:.1%}, в B {meta['return_proxy']['reward_val']['fraction']:.1%}, в C {meta['return_proxy']['policy_val']['fraction']:.1%}. Сигнал сильно несбалансирован и означает будущую активность, а не доказанное возвращение после ухода.

Опорные события для этой описательной проверки выбирались с шагом 100 внутри истории пользователя. Доли усреднены по окнам; более активные пользователи дают больше окон. Этот знаменатель отличается от усреднения Recall/NDCG по пользователям. Проверка не обучает и не оценивает новую политику удержания.

Исходный Yambda отбирает пользователей с активностью в раннем периоде и хотя бы одним действием в последующем. Поэтому эти доли смещены относительно произвольных пользователей сервиса. Возможная награда — вероятность следующего взаимодействия или активности в полном будущем окне; для проверки потребуются сравнение с простым предиктором, калибровка, анализ редких отрицательных исходов и соблюдение зрелости меток. Чтобы оптимизировать фактическое удержание, нужна модель реакции и прекращения взаимодействия, а не только дослушивания. Причинный эффект рекомендации на возвращение потребует онлайн-проверки.

Для дальнейшего исследования можно задать `r_long = a·c_hat + b·p_return7 + c·p_continue − d·repeat`. Здесь `c_hat` — ожидаемая доля прослушивания; `p_return7` — прогноз возвращения в полном семидневном окне после окончания сессии; `p_continue` — прогноз следующего взаимодействия в пределах выбранного сессионного интервала; `repeat` — повтор внутри предложенной цепочки. Веса a, b, c, d неотрицательны и выбираются на validation. Границу сессии нужно отдельно определить по разрывам времени и проверить чувствительность результата к этому определению. Это предложенная функция: модели возвращения и продолжения в текущем эксперименте не обучались. Офлайн можно проверить их прогнозы на реально записанных событиях; их причинную реакцию на другой рекомендуемый трек из этих данных установить нельзя.

**Пять шагов вперёд.** В одинаковых условиях сгенерированы цепочки длины 5 для 512 пользователей. После каждого шага предложенный ID добавляется в контекст; он не считается подтверждённым прослушиванием. По шагам измерены награда, повторения, KL и оценка того же объекта на исходной реальной истории. Последнее сравнение позволяет увидеть чувствительность reward model к искусственным контекстам.

![Генерация пяти шагов](output/figures/rollout.png)

| Модель | Суммарная дисконтированная доля | Суммарная дисконтированная награда времени | Повтор на шаге 5 | Сдвиг оценки контекста на шаге 5 |
|---|---:|---:|---:|---:|
{rollout_table}

Суммы вычислены как Σ γᵗ rₜ при γ = 0.9. Для времени rₜ — ограниченное время, делённое на D; шкалы двух сумм отличаются. Сдвиг контекста — разница оценки одного и того же объекта на сгенерированной и исходной истории. Эти суммы относятся только к описанному искусственному процессу.

{repeat_text}

Различие оценок на исходном и сгенерированном контексте — диагностический признак сдвига, не измеренная ошибка: истинных откликов на сгенерированные цепочки нет. Оценщик обучался на реальных историях, поэтому доверие к многократному продолжению ограничено. Обучение на H=5 и последующее свободное продолжение анализируются отдельно от ранжирования следующего реального объекта.

**Компромисс награды и точности.** График сопоставляет исходную модель, разные β, награды и горизонты при одинаковом D. Сильный KL ограничивает изменение распределения, но может уменьшить доступный рост награды. Слабый KL допускает большее отклонение, включая эксплуатацию ошибок оценщика. Изменение основного и диагностического оценщиков находится в `output/test_evaluation.json`; даже их согласованный рост не заменяет реальный эксперимент с пользователями.

Функции награды имеют разный масштаб и разброс, даже когда обе ограничены [0,1]. Поэтому один и тот же β не создаёт одинакового компромисса для R1 и R2. Его действие оценивается по измеренному KL, точности и изменениям рекомендаций; коэффициент сам по себе не является показателем силы улучшения.

## 7. Ограничения и дальнейшая работа

Эксперимент ограничен одним временным разбиением, малой GPT-2, популярным каталогом и выборкой активных пользователей. Большая доля целей вне каталога снижает достижимые основные метрики. Отбор пользователей в исходном датасете использует будущую активность. Точная последовательность событий внутри пятисекундной группы неизвестна; доступность отклика в исходных логах не документирована полностью. Отложенная проверка reward model охватывает логированные известные объекты, а не произвольные рекомендации и искусственные состояния.

Нельзя вычислить надёжную причинную оценку через IPS только из вероятностей обученной next-item модели: вероятность прослушивания не равна известной вероятности показа логирующей политики. Следующие направления: увеличить каталог без утечки, проверить другой временной период, улучшить модель отклика и неопределённости, отдельно моделировать пропуск и прекращение сессии, затем провести онлайн A/B-проверку. Отрицательный эффект RL на точность является содержательным результатом, а не основанием скрывать запуск.

## 8. Вычисления, проверки и воспроизведение

Окружение: Apple M5, 16 ГБ общей памяти, PyTorch {benchmark['torch']}, Transformers 4.57.6, MPS, четыре CPU-потока. Модель содержит {benchmark['parameters']:,} параметров. На пробном запуске после пяти прогревочных шагов медиана шага составила {benchmark['median_seconds_per_step_after_warmup']:.4f} с. Это технический замер при данном размере batch, а не гарантия скорости всех этапов. Выделенная MPS-память и память драйвера записаны в `output/benchmark.json`; они не являются полным пиковым RSS.

Сумма времени циклов основных обучений по их журналам: {total/60:.1f} минуты; она включает промежуточные оценки и сохранения. Предварительный расчёт reward-признаков, окончательная оценка, установка и скачивание в эту сумму не входят. Время отдельных процессов записано в `output/execution_log.jsonl`; ошибочные предварительные запуски отделены и не входят в результаты.

Проверены причинная маска, запрет будущих кандидатов, временные границы и внутренние маски целей по исходным timestamp. Recall/NDCG сверены с ручным примером и перестановкой строк. Сохранение и продолжение AdamW с восстановлением случайных состояний проверено численно; результаты находятся в `output/verification.json`.

Используется Python 3.12. Команды выполняются из папки этого проекта:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export ANNAI_RL_OUTPUT=output/reproduction
.venv/bin/python download.py
.venv/bin/python prepare.py
.venv/bin/python verify.py
.venv/bin/python experiment.py benchmark --batch 32
.venv/bin/python run_experiments.py --stop-after validation
.venv/bin/python run_experiments.py --stop-after training
.venv/bin/python run_experiments.py --stop-after test
.venv/bin/python audit.py
.venv/bin/python report.py
```

GPU выбирается автоматически при доступности MPS; иначе используется CPU. Скрипт пропускает завершённые обучения, а незавершённые продолжает из `last.pt`. Для ограниченного сеанса можно повторить команду соответствующей модели с теми же параметрами и добавить `--minutes 30`; при повторном запуске состояние восстановится. При внезапном выключении теряются только шаги после последнего checkpoint. Итоговую проверку нельзя использовать повторно для настройки. Новый независимый эксперимент запускается в отдельной папке с чистым `output`.

Версии всех зависимостей перечислены в `requirements.txt`. Хеши данных, исходного кода и выбранных весов приложены. Воспроизведение на другом устройстве может давать небольшие численные различия. Исходные данные скачиваются отдельно; синтетические результаты не использованы.

## Источники

1. [Yambda: официальные данные и документация](https://huggingface.co/datasets/yandex/yambda).
2. [Ploshkin et al. Yambda-5B: статья, преобразование времени и отбор пользователей](https://arxiv.org/html/2505.22238v1).
3. [Petrov, Macdonald. Aligning GPTRec with Beyond-Accuracy Goals with Reinforcement Learning](https://arxiv.org/pdf/2403.04875); [код авторов](https://github.com/asash/gptrec_rl). В статье используются PPO и обучение через teacher; текущий эксперимент использует прямое NIP-обучение и REINFORCE.
4. [Matveev et al. Session-Level Optimization for Large-Scale Retrieval using REINFORCE with Multi-Step Off-Policy Correction, v2](https://arxiv.org/html/2607.02818v2). Взяты постановка сессионной цели и ограничения модельной оценки; полное воспроизведение статьи не заявляется.
5. [Williams. REINFORCE](https://doi.org/10.1007/BF00992696).
6. [Hugging Face: GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2); [PyTorch: MPS](https://docs.pytorch.org/docs/main/notes/mps.html).
'''
    relative_output = OUT.relative_to(ROOT).as_posix()
    content = content.replace('output/', relative_output+'/')
    (ROOT/'README.md').write_text(content)
    summary = {'selected_ratio':ratio_name, 'selected_time':time_name,
               'ratio_reward_change':ratio_gain, 'capped_seconds_change':time_gain,
               'seed_results':seed_results, 'training_loop_seconds':stages,
               'total_training_loop_seconds':total, 'reward_quality':quality}
    write_json(OUT/'summary.json',summary)
    # Комплект для чтения и воспроизведения, без окружения и исходного Parquet.
    package = OUT/'rl_recommendations_submission.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
        for path in ROOT.glob('*.py'):
            z.write(path,path.name)
        z.writestr('README.md', content.replace(relative_output+'/', 'output/'))
        for name in ['requirements.txt','protocol.json']:
            z.write(ROOT/name,name)
        for name in ['data_manifest.json','frozen_selection.json','test_evaluation.json','policy_val_evaluation.json',
                     'summary.json','benchmark.json','verification.json','execution_log.jsonl','validation_selection.json',
                     'completion_audit.json']:
            z.write(OUT/name,f'output/{name}')
        for path in figures.glob('*.png'):
            z.write(path,f'output/figures/{path.name}')
        for name in frozen['models'] + ['reward_s11','reward_s22']:
            for filename in ['history.json','test.json','reward_val.json']:
                path = OUT/name/filename
                if path.exists():
                    z.write(path,f'output/{name}/{filename}')
    print('Отчёт, графики и архив созданы:',package,flush=True)


if __name__ == '__main__':
    main()
