# Воспроизведение

Используйте Python 3.12 и версии из `requirements.txt`. Команды ниже выполняются из корня репозитория. Сохранённые результаты, модели и данные проверяются до нового запуска:

```bash
.venv/bin/python -B scripts/reproduce.py verify
.venv/bin/python -B -m unittest discover -s tests -v
```

## Отдельная рабочая копия

Код использует относительные пути первоначального протокола. Команда `init` копирует исходники в исходную структуру внутри `.cache/reproduction/` и подключает доступный исходный Parquet. Источники сверяются с картой SHA-256. Рабочая копия имеет собственный `output/`; сохранённые веса и исходы D из `results/` туда не подставляются. Это предотвращает перезапись опубликованных чисел и смешение нового обучения с окончательной оценкой.

```bash
.venv/bin/python -B scripts/reproduce.py init
.venv/bin/python -B scripts/reproduce.py run download
.venv/bin/python -B scripts/reproduce.py run prepare-baseline
for seed in 11 22 33; do
  .venv/bin/python -B scripts/reproduce.py run baseline -- --name "baseline_s$seed" --seed "$seed" --steps 6000 --batch 32 --lr 0.001 --save-every 250 --eval-every 1000
done
```

`download` проверяет уже подключённый Parquet и скачивает файл только при его отсутствии. Для другого рабочего каталога добавьте `--workspace .cache/reproduction-2` к каждой команде. Повторное выполнение `init` принимает существующую рабочую копию только при совпадении исходников и её манифеста.

## Итоговый эксперимент

```bash
.venv/bin/python -B scripts/reproduce.py run prepare
.venv/bin/python -B scripts/reproduce.py run checks
.venv/bin/python -B scripts/reproduce.py run benchmark
.venv/bin/python -B scripts/reproduce.py run reward
.venv/bin/python -B scripts/reproduce.py run pilot
.venv/bin/python -B scripts/reproduce.py run retention
.venv/bin/python -B scripts/reproduce.py run training
.venv/bin/python -B scripts/reproduce.py run freeze
.venv/bin/python -B scripts/reproduce.py run holdout
.venv/bin/python -B scripts/reproduce.py run audit
.venv/bin/python -B scripts/reproduce.py run serving
.venv/bin/python -B scripts/reproduce.py run report
```

`holdout` сначала подготавливает отложенную выборку после проверки фиксации, затем вызывает совместимый драйвер. `report` формирует документ и архив внутри новой рабочей копии. Корневые `REPORT.md` и `report_short.html` относятся к зафиксированному эксперименту и автоматически не заменяются результатами нового запуска.

Во время переорганизации повторное обучение не выполнялось. Перечисленные команды воспроизводят исходный протокол; полный повтор на новом оборудовании может отличаться численно. Перед новой исследовательской итерацией нужен отдельный заранее заданный протокол и независимый временной период. [Известные особенности технического повтора](technical-notes.md).
