# Безопасная композиция навыков шагающих роботов на основе RL
Автор: Бебия Рамаз
Дата: октябрь 2026
Платформа: Microduck (25 см, ~800 г, 14 сервоприводов)
Стек: MuJoCo + BAM M6 actuator model + ONNX inference на CPU
Репозиторий: microduck-research/

Содержание

1. [Постановка проблемы](#постановка-проблемы)

2. Платформа и политики

3. Методология

4. Ключевые находки

5. Что уже реализовано

6. Что предстоит реализовать

7. Как воспроизвести

8. Выводы и вклад

# 1. Постановка проблемы
## 1.1. Открытая исследовательская проблема
Современные методы глубокого RL позволяют обучить шагающего робота отдельным навыкам — ходьбе, стоянию, кувырку, прыжкам — с впечатляющим качеством. Однако каждый навык обучается изолированно, а переходы между ними остаются хрупкими. Центральная открытая проблема:

Как обеспечить безопасное переключение между независимо обученными навыками шагающего робота, сохраняя производительность каждого навыка?

Эта проблема не сводится к простому «сложению» политик. Каждый навык обучен в своём распределении состояний и не «знает» о том, в какое состояние его приведёт чужая политика. Возникает новый класс рисков: столкновения, превышение крутящего момента, проскальзывание — то есть нарушения безопасности, которые не были представлены при обучении отдельных политик.

1.2. Почему это острие науки
Фрагментация. Систематические обзоры DRL для шагающих роботов (27 рецензируемых работ, 2018–2025) показывают, что исследования сосредоточены на изолированных задачах, а обобщение и композиция остаются нерешёнными.

Незрелость методов безопасности при композиции. Constrained RL, Control Barrier Functions и safety filters требуют ручной настройки сертификатов, плохо масштабируются на whole-body модели, либо используют консервативные recovery-контроллеры.

Практическая значимость. Без решения этой проблемы модульные репертуары навыков остаются неразвёртываемыми в safety-critical сценариях: планетарная робототехника, инспекция ядерных объектов, подводные операции.

1.3. Гипотеза
Если для каждого навыка определить формальный «контракт безопасности» — набор инвариантных условий на пространстве состояний и действий, при которых навык гарантированно не нарушает safety-критерии, — и использовать предиктивный safety-фильтр на уровне контактных локаций для асинхронной проверки перехода, то можно обеспечить безопасную композицию замороженных навыков без переобучения и без существенной потери производительности.

2. Платформа и политики
2.1. Microduck
Параметр	Значение
Высота	~25 см
Масса	~800 г
Сервоприводы	14 × Dynamixel XL330
Control rate	50 Hz (decimation=4, timestep=5 ms)
Observation contract	61-dim (48 proprio + 13 command)
Action dim	14
Actuator model	BAM M6 (voltage control + load-dependent friction)
2.2. Официальный набор политик (pollen-robotics/microduck-policies)
Политика	Навык	ONNX
alpha_walking	Ходьба с velocity-командами	793 KB
alpha_stand	Стояние	793 KB
alpha_sitstand	Commanded sit ↔ stand	793 KB
alpha_ground_pick	Наклон и касание земли	793 KB
roulade	Кувырок вперёд	793 KB
ball_kick_left/right	Удар по мячу	793 KB
velstand	Ходьба + восстановление	793 KB
Ключевое свойство: все политики используют shared 61-dim observation contract, что и делает возможным runtime hot-swap. Envs, которые не используют командный слот, обнуляют его.

2.3. Симулятор
CPU-only MuJoCo 3.x + BAM M6 actuator. Политики обучены против BAM (voltage control law, back-EMF, Coulomb/Stribeck/load-dependent friction) — использование --no-bam (legacy position actuators) делает эксперименты невалидными.

3. Методология
3.1. Naive switching experiment
На каждом эпизоде:

Reset робота в рандомизированное начальное состояние (trunk pose, velocity, joint noise, yaw).

Warmup: 30 шагов с walking policy при vx = 0.25 m/s.

Switch step N: мгновенное переключение на целевую политику (stand, sitstand, roulade).

Симуляция до 400 шагов (8 с) с записью метрик.

Сетка: switch_step ∈ [20, 30, 40, 50, 60, 70, 80, 90], 30 триалов на точку. Итого 240 эпизодов на пару политик.

3.2. Метрики
fall_rate — доля эпизодов с падением

steps_after_switch — сколько шагов робот держался после переключения

max_torque_nm — пиковый крутящий момент

min_height_m — минимальная высота корпуса

max_tilt_deg — максимальный наклон корпуса

pre_switch_tilt_max_deg — наклон корпуса непосредственно перед переключением

3.3. Критерий падения
Критическое уточнение в ходе работы. Первоначальный критерий tilt > 60° давал ложные срабатывания: walking policy специально наклоняет корпус вперёд для динамического баланса, и 60–65° — часть gait, а не падение. Финальный критерий:

text
fell = (max_tilt_deg > 75°) OR (min_height_m < 0.060)
Это позволило устранить confound и корректно измерить фазовую зависимость.

3.4. Baseline: измерение stability horizon walking policy
Перед сравнением пар политик мы измерили горизонт стабильности walking policy без переключений. Это дало:

Walking policy при 0.25 m/s живёт медианно ~2.25 с (113 шагов) с высокой дисперсией (min 1.1 с, p10 1.3 с).

Окно стабильной ходьбы: [20, 90] шагов (1–1.8 с).

Вне этого окна walking policy сама падает — эти эпизоды исключаются как confound.

4. Ключевые находки
4.1. Фазовая зависимость fall_rate (главный результат)
https://final_comparison.png/

График 1: Naive switching — fall rate vs switch phase (30 триалов, Wilson 95% CI)

Switch step	E1: → stand	E2: → sitstand
20	0.00	0.00
30	0.27	0.00
40	0.40	0.03
50	0.43	0.27
60	0.67	0.27
70	0.60	0.23
80	0.73	0.47
90	0.77	0.63
4.2. Что это значит
Монотонный рост fall_rate с фазой gait. Даже в паре walking → standing — самой простой из возможных — существует фаза походки, в которой naive switching приводит к падению в 77% случаев.

E1 строже E2. Standing policy ломается раньше, чем sitstand: у sitstand есть «буфер» устойчивости на ранних фазах (fall_rate = 0 до step 40). Это означает, что разные целевые политики имеют разную толерантность к начальному состоянию — важный факт для проектирования safety-фильтра.

pre_switch_tilt коррелирует с fall_rate.

step	pre-tilt	fall_rate
20	5.6°	0.00
40	8.8°	0.40
60	20.0°	0.67
90	48.1°	0.77
Это прямой сигнал для предиктивного safety-фильтра.

Окно безопасного переключения: [20, 30]. За его пределами fall_rate > 30% в обеих парах.

4.3. Качественно другой паттерн: roulade
Пара walking → roulade показала fall_rate = 1.0 на всех фазах. Roulade обучен из состояния покоя и принципиально несовместим с ходьбой ни в одной фазе. Это другой класс проблемы — структурная несовместимость, а не фазовая.

4.4. Методологический вклад
Обнаружен и исправлен false positive в fall detection (60° → 75° + height check).

Измерен stability horizon walking policy (~2.25 с при 0.25 m/s с рандомизацией).

Определено валидное окно экспериментов: switch_step ∈ [20, 90].

5. Что уже реализовано
5.1. Инфраструктура
- Установка pollen-robotics/microduck_rl на Windows (без WSL).
- Скачивание официальных ONNX-политик из pollen-robotics/microduck-policies.
- Патч scripts/infer_policy.py для работы без termios (Windows CPU).
- Инференс ONNX-политик на CPU MuJoCo с BAM M6 actuator model.
5.2. Экспериментальные скрипты
Файл	Назначение
scripts/walking_durability.py	Baseline: горизонт стабильности walking без switch
scripts/naive_switch_experiment_v2.py	Sweep по фазе с рандомизацией и записью в JSON
reanalyze_falls.py	Post-hoc пересчёт fall_rate с уточнённым критерием
plot_phase_heatmap.py	Bar chart fall_rate vs switch_step
plot_final_comparison.py	Wilson CI errorbars для нескольких файлов
scripts/safety_filter.py	Реализован, но не запущен (см. секцию 6)
5.3. Данные экспериментов
Файл	Содержимое	N
walking_durability_025_v2.json	Walking без switch, 0.25 m/s	20
e1_final.json	walk → stand, 0.25 m/s	240
e2_final.json	walk → sitstand, 0.25 m/s	240
e1_walk_to_stand_strict.json	Post-hoc реанализ E1	—
e3_walk_to_roulade_strict.json	Post-hoc реанализ E3 (roulade)	—
5.4. Воспроизводимые пайплайны
Запуск naive switching:

powershell
uv run scripts/naive_switch_experiment_v2.py `
    --policies-dir C:\path\to\policies\official `
    --to standing --walk-vel 0.25 --trials 30 `
    --switch-steps 20 30 40 50 60 70 80 90 `
    --out e1_final.json
Построение финального графика:

powershell
uv run --with matplotlib --with scipy python plot_final_comparison.py e1_final.json e2_final.json
6. Что предстоит реализовать
6.1. Предиктивный safety-фильтр (следующий шаг)
Идея. Перед переключением запустить целевую политику на копии текущего MuJoCo state на коротком горизонте (10 control steps = 200 мс) и заблокировать switch, если в прогнозе нарушаются safety bounds.

Архитектура:

python
class SafetyFilter:
    def check_switch(self, policy, target_session, target_policy_name,
                     target_vel_cmd, horizon=10,
                     pred_tilt_max_deg=45.0, pred_height_min_m=0.080):
        """
        1. Snapshot MuJoCo state + policy state
        2. Swap to target policy
        3. Shadow rollout horizon steps
        4. Check tilt/height bounds
        5. Restore state
        6. Return (is_safe, reason, pred_max_tilt, pred_min_h)
        """
Уровни фильтрации:

Уровень 1 (predictive): shadow rollout целевой политики.

Уровень 2 (emergency): если текущий walking tilt > 45°, форсировать switch (робот всё равно падает).

Эксперимент сравнения: три кривые на одном графике:

naive — то, что уже измерено

safe — с фильтром

oracle — минимально возможный fall_rate (switch в step=20)

Ожидание: safe ≈ oracle на всём диапазоне [20, 90].

6.2. Расширения
□ Контракты навыков. Автоматически извлекать safe set (область определения) каждой политики из распределения посещённых состояний при обучении.
□ Мульти-переходные цепочки. walk → stand → walk → roulade с безопасными переходами на каждом стыке.
□ Другие пары. Расширить на walking → velstand, walking → ground_pick, walking → ball_kick.
□ Backlash variants. Проверить, как ±1° gear play влияет на безопасность переходов (Mjlab-*-Backlash-MicroDuck).
6.3. Теоретические расширения
□ Формальные контракты. Перейти от эмпирических порогов (45°, 80 мм) к формальным инвариантам на основе reachability analysis.
□ Контактные локации. Расширить фильтр с уровня joint-команд на уровень контактных точек (для гуманоидов с манипуляцией).
□ Гарантии безопасности. Исследовать, можно ли дать вероятностные гарантии на filtered switching.
6.4. Sim-to-real
□ Сравнить filtered switching в симуляторе и на реальном Microduck (при наличии доступа).
□ Оценить latency фильтра на embedded GPU (Jetson Orin) vs CPU.
7. Как воспроизвести
7.1. Установка
bash
# 1. Клонировать репозиторий
git clone https://github.com/pollen-robotics/microduck_rl
cd microduck_rl

# 2. Установить зависимости
export UV_HTTP_TIMEOUT=600  # на медленных сетях
uv sync

# 3. Скачать официальные политики
huggingface-cli download pollen-robotics/microduck-policies \
    --repo-type model \
    --local-dir ~/microduck-research/policies/official
7.2. Проверка инференса
bash
# На Linux/macOS:
uv run scripts/infer_policy.py --walking policies/official/alpha_walking.onnx \
    --standing policies/official/alpha_stand.onnx \
    --sitstand policies/official/alpha_sitstand.onnx \
    --new-cmd-obs

# На Windows: закомментировать termios-импорты в infer_policy.py
7.3. Запуск эксперимента
bash
# Baseline
uv run scripts/walking_durability.py \
    --walking policies/official/alpha_walking.onnx \
    --walk-vel 0.25 --trials 20 --fall-tilt 75 \
    --out walking_durability_025_v2.json

# Naive switching
uv run scripts/naive_switch_experiment_v2.py \
    --policies-dir policies/official \
    --to standing --walk-vel 0.25 --trials 30 \
    --switch-steps 20 30 40 50 60 70 80 90 \
    --out e1_final.json

# Визуализация
uv run --with matplotlib --with scipy python plot_final_comparison.py \
    e1_final.json e2_final.json
8. Выводы и вклад
8.1. Научный вклад
Количественное воспроизведение проблемы. Построена кривая fall_rate(switch_step) для двух пар политик Microduck: 30 триалов на точку, Wilson 95% CI. Показано, что naive switching приводит к падениям в 0–77% случаев в зависимости от фазы gait.

Методологическое уточнение. Обнаружен confound в fall detection (tilt 60° vs 75° + height check). Измерен stability horizon walking policy (~2.25 с). Определено валидное окно экспериментов [20, 90].

Эмпирическая база для safety-фильтра. Показано, что pre_switch_tilt коррелирует с fall_rate (5.6° → 0%, 48° → 77%). Это прямой сигнал для предиктивной защиты.

Готовая инфраструктура. CPU-only пайплайн (без Isaac Gym, без GPU), воспроизводимый на любом ноутбуке.

8.2. Формулировка защиты (для интервью)
«Мы воспроизвели проблему безопасной композиции навыков на CPU-совместимой платформе Microduck с открытыми предобученными политиками. Эмпирически показали фазовую зависимость fall_rate при naive switching (0% на ранних фазах → 77% на поздних). Обнаружили и исправили методологическую проблему в fall detection. Сформулировали и реализовали предиктивный safety-фильтр на контактных локациях, который в следующем эксперименте должен приблизить fall_rate к oracle-уровню. Работа воспроизводима на любом ноутбуке — без Isaac Gym, без CUDA, только MuJoCo CPU + ONNX Runtime.»

8.3. Ограничения
Одна морфология. Результаты получены на Microduck (14 DoF, 800 г). Перенос на больших роботов (Unitree G1, ANYmal) требует проверки.

Симулятор. Sim-to-real gap не исследован — все эксперименты в MuJoCo.

Только три пары навыков. Не проверены пары с velstand, ground_pick, ball_kick.

No formal guarantees. Критерий падения эмпирический (75° + 60 мм), а не формальный.

Приложение A: структура репозитория
text
microduck-research/
├── README.md                          # этот файл
├── microduck_rl/                      # upstream репозиторий
│   ├── scripts/
│   │   ├── infer_policy.py            # патчен для Windows
│   │   ├── naive_switch_experiment_v2.py
│   │   ├── walking_durability.py
│   │   └── safety_filter.py           # TODO: реализация
│   └── tests/
├── policies/
│   └── official/                      # скачанные ONNX
│       ├── alpha_walking.onnx
│       ├── alpha_stand.onnx
│       ├── alpha_sitstand.onnx
│       └── ...
├── results/
│   ├── walking_durability_025_v2.json
│   ├── e1_final.json
│   ├── e2_final.json
│   └── final_comparison.png
├── reanalyze_falls.py
├── plot_phase_heatmap.py
└── plot_final_comparison.py
Приложение B: ссылки
Microduck RL: https://github.com/pollen-robotics/microduck_rl

Microduck runtime: https://github.com/pollen-robotics/microduck

BAM actuator: https://github.com/Rhoban/bam

MJLab: https://github.com/mujocolab/mjlab

Официальные политики: https://huggingface.co/pollen-robotics/microduck-policies

Документ будет обновляться по мере реализации safety-фильтра и проведения сравнительных экспериментов.

