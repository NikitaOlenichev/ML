# Вспомогательные библиотеки
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


pd.set_option('display.width', 160)
pd.set_option('display.max_columns', 12)

# Загрузка данных
df = pd.read_csv('telemetry_dzz_sem1.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

params = ['temperature', 'voltage', 'current', 'angular_velocity']
titles = {'temperature': 'Температура, °C', 'voltage': 'Напряжение, В', 'current': 'Сила тока, А',
          'angular_velocity': 'Угловая скорость'}
colors = {'temperature': 'crimson', 'voltage': 'royalblue', 'current': 'darkorange',
          'angular_velocity': 'seagreen'}

# 1. Анализируем норму по режимам
print("1) Норма по режимам работы (mode):")
mode_stats = df.groupby('mode')[params].agg(['mean', 'min', 'max', 'std'])
print(mode_stats.round(2))
print("\nПодробно по каждому параметру:")
for p in params:
    print(p)
    print(df.groupby('mode')[p].describe().round(2))
    print()
print()

# 2. Статистический детектор (1σ, 2σ, 3σ) - глобальный и по режимам
print('2) Статистический детектор (отклонение от среднего в сигмах):')
TARGET = 'temperature'
mean = df[TARGET].mean()
std = df[TARGET].std()
print(f"Параметр: {TARGET}\n"
      f"global mean={mean:.2f}, global std={std:.2f}")
sigma_counts_global = {}
for k in [1, 2, 3]:
    flag = (df[TARGET] - mean).abs() > k * std
    sigma_counts_global[k] = flag.sum()
    df[f'anomaly_{k}sigma_global'] = flag.astype(int)
    print(f"  {k}σ (глобально): порог = mean ± {k*std:.2f} -> {flag.sum()} аномалий "
          f"({100*flag.mean():.1f}%)")
print(f"\nТот же {TARGET}, но mean/std считаются отдельно для каждого mode:")
mode_mean = df.groupby('mode')[TARGET].transform('mean')
mode_std = df.groupby('mode')[TARGET].transform('std')
sigma_counts_mode = {}
for k in [1, 2, 3]:
    flag = (df[TARGET] - mode_mean).abs() > k * mode_std
    sigma_counts_mode[k] = flag.sum()
    df[f'anomaly_{k}sigma_mode'] = flag.astype(int)
    print(f"  {k}σ (по режиму): -> {flag.sum()} аномалий ({100*flag.mean():.1f}%)")
print('\n')

# 3. Резкие изменения между соседними измерениями
print("3) Резкие изменения (|Δ| между соседними отсчётами)")
for p in params:
    df[f'{p}_diff'] = df[p].diff()
diff_thresholds = {}
for p in params:
    thr = df[f'{p}_diff'].abs().quantile(0.99)
    diff_thresholds[p] = thr
    flag = df[f'{p}_diff'].abs() > thr
    df[f'{p}_sharp_change'] = flag.astype(int)
    print(f"  {p:18s}: порог (99-й перцентиль |Δ|) = {thr:.3f}  -> {flag.sum()} точек")
df['mode_changed'] = df['mode'] != df['mode'].shift(1)
for p in params:
    df[f'{p}_sharp_change_samemode'] = ((df[f'{p}_diff'].abs() > diff_thresholds[p])
                                        & (~df['mode_changed']))
print('\n')

# 4. Комбинируем несколько признаков
print('4) Оценка подозрительности (score) - с учетом режима:')


def mode_z(frame, col):
    m = frame.groupby('mode')[col].transform('mean')
    s = frame.groupby('mode')[col].transform('std').replace(0, np.nan)
    return (frame[col] - m) / s


df['z_temp'] = mode_z(df, 'temperature')
df['z_volt'] = mode_z(df, 'voltage')
df['z_curr'] = mode_z(df, 'current')
df['z_omega'] = mode_z(df, 'angular_velocity')
Z_THR = 2.0
temperature_is_high = df['z_temp'] > Z_THR
voltage_is_low = df['z_volt'] < -Z_THR
current_is_high = df['z_curr'] > Z_THR
angular_velocity_is_high = df['z_omega'] > Z_THR
score = (temperature_is_high.astype(int) + voltage_is_low.astype(int) +
         current_is_high.astype(int) + angular_velocity_is_high.astype(int))
df['score'] = score
df['anomaly_combo'] = (score >= 2).astype(int)
print("Срабатывания отдельных признаков (mode-aware, порог 2σ внутри режима):")
print(f"  temperature_is_high : {temperature_is_high.sum()}")
print(f"  voltage_is_low      : {voltage_is_low.sum()}")
print(f"  current_is_high     : {current_is_high.sum()}")
print(f"  angular_velocity_is_high : {angular_velocity_is_high.sum()}")
print(f"\nАномалий при score >= 2: {df['anomaly_combo'].sum()} из {len(df)} "
      f"({100*df['anomaly_combo'].mean():.1f}%)")
print("Распределение по score:")
print(score.value_counts().sort_index())
print('\n')

# 5. Сравниваем три детектора
print('5) Сравнение трех детекторов:')

# Детектор №1 - жёсткие пороги из Задания 1
d1 = (
    (df['temperature'] > 60) |
    (df['voltage'] < 23.5) |
    (df['current'] > 8.0) |
    (df['angular_velocity'] > 1.0)
).astype(int)
df['anomaly_d1_thresholds'] = d1

# Детектор №2 - статистическое отклонение (глобальная 2σ, по всем 4 параметрам)
d2 = (
    ((df['temperature'] - df['temperature'].mean()).abs() > 2 * df['temperature'].std()) |
    ((df['voltage'] - df['voltage'].mean()).abs() > 2 * df['voltage'].std()) |
    ((df['current'] - df['current'].mean()).abs() > 2 * df['current'].std()) |
    ((df['angular_velocity'] - df['angular_velocity'].mean()).abs() > 2 * df['angular_velocity'].std())
).astype(int)
df['anomaly_d2_statistical'] = d2

# Детектор №3 - комбинация признаков с учётом режима (score >= 2)
d3 = df['anomaly_combo']
print(f"Детектор №1 (жёсткие пороги):        {d1.sum():3d} точек  "
      f"({100 * d1.mean():.1f}%)")
print(f"Детектор №2 (глобальная статистика): {d2.sum():3d} точек  "
      f"({100 * d2.mean():.1f}%)")
print(f"Детектор №3 (комбинация, mode-aware):{d3.sum():3d} точек  "
      f"({100 * d3.mean():.1f}%)")
overlap_12 = ((d1 == 1) & (d2 == 1)).sum()
overlap_13 = ((d1 == 1) & (d3 == 1)).sum()
overlap_23 = ((d2 == 1) & (d3 == 1)).sum()
only_d3 = ((d3 == 1) & (d1 == 0) & (d2 == 0)).sum()
print(f"\nСовпадение Д1 и Д2: {overlap_12}, Д1 и Д3: {overlap_13}, Д2 и Д3: {overlap_23}")
print(f"Найдено ТОЛЬКО детектором №3 (пропущено Д1 и Д2): {only_d3} точек")
print('\n')

# Графики сравнения по трём параметрам
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
plot_params = ['temperature', 'current', 'angular_velocity']
markers = {'anomaly_d1_thresholds': ('o', 'gray', 'Детектор №1 (пороги)'),
           'anomaly_d2_statistical': ('^', 'darkorange', 'Детектор №2 (статистика)'),
           'anomaly_combo': ('x', 'crimson', 'Детектор №3 (комбинация)')}
for ax, p in zip(axes, plot_params):
    ax.plot(df['timestamp'], df[p], color=colors[p], linewidth=0.8, zorder=1)
    for col, (mk, c, lbl) in markers.items():
        sub = df[df[col] == 1]
        ax.scatter(sub['timestamp'], sub[p], marker=mk, s=28, facecolors='none' if mk == 'o' else c,
                   linewidths=1.3, label=lbl, zorder=3)
    ax.set_ylabel(titles[p])
    ax.grid(alpha=0.3)
axes[0].set_title("Сравнение трёх детекторов на телеметрии")
axes[0].legend(loc='upper right', fontsize=9)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
axes[-1].set_xlabel('Время')
fig.tight_layout()
fig.savefig('task2_01_three_detectors.png', dpi=130)
plt.close(fig)

# График: резкие изменения, отмеченные на графике
fig, axes = plt.subplots(len(params), 1, figsize=(14, 12), sharex=True)
for ax, p in zip(axes, params):
    ax.plot(df['timestamp'], df[p], color=colors[p], linewidth=0.8)
    sub = df[df[f'{p}_sharp_change_samemode']]
    ax.scatter(sub['timestamp'], sub[p], color='black', s=22, zorder=3,
               label='резкое изменение (внутри режима)')
    ax.set_ylabel(titles[p])
    ax.grid(alpha=0.3)
axes[0].set_title("Точки резкого изменения параметров (без учёта плановой смены режима)")
axes[0].legend(loc='upper right')
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
fig.tight_layout()
fig.savefig('task2_02_sharp_changes.png', dpi=130)
plt.close(fig)

# График: mode-aware статистический детектор (temperature) 1/2/3 sigma
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df['timestamp'], df['temperature'], color='gray', linewidth=0.8, label='temperature')
sig_markers = {1: ('yellow', 12), 2: ('orange', 20), 3: ('red', 30)}
for k in [3, 2, 1]:
    sub = df[df[f'anomaly_{k}sigma_mode'] == 1]
    c, s = sig_markers[k]
    ax.scatter(sub['timestamp'], sub['temperature'], color=c, s=s,
               label=f'{k}σ внутри режима (n={len(sub)})', zorder=3)
ax.set_ylabel("Температура, °C")
ax.set_title("Статистический детектор (mode-aware): 1σ vs 2σ vs 3σ")
ax.legend(loc='upper right')
ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
fig.tight_layout()
fig.savefig('task2_03_sigma_comparison.png', dpi=130)
plt.close(fig)

# 6. Итоговая таблица
summary = pd.DataFrame([
    {
        'Детектор': '№1 - жёсткие пороги',
        'Аномалий': int(d1.sum()),
        'Признаки': 'temperature>60, voltage<23.5, current>8.0, ang_vel>1.0 (фикс. пороги)',
        'Недостаток': 'Пороги произвольны и одинаковы для всех режимов; не видит "слабых", '
                      'но нетипичных отклонений',
    },
    {
        'Детектор': '№2 - статистика (глобальная 2σ)',
        'Аномалий': int(d2.sum()),
        'Признаки': '|x - mean| > 2σ по всем 4 параметрам, mean/std по всему датасету',
        'Недостаток': 'Путает "режимный сдвиг" (imaging теплее standby) с реальной аномалией; '
                      'ложные срабатывания на границах режимов',
    },
    {

        'Детектор': '№3 - комбинация (score, mode-aware)',
        'Аномалий': int(d3.sum()),
        'Признаки': 'score = сумма 4 mode-aware признаков (|z| > 2σ внутри mode), '
                    'anomaly если score>=2',
        'Недостаток': 'Всё ещё нужно вручную выбрать Z_THR и порог score; '
                      'не учитывает скорость изменения и историю',
    },
])
print('6) Итоговая таблица:')
print(summary.to_string(index=False))
print('\n')

# Финальные таблицы
df.to_csv('telemetry_task2_result.csv', index=False)
summary.to_csv('task2_summary_table.csv', index=False)

# Получившиеся файлы
print('Получившиеся файлы:')
print('- task2_01_three_detectors.png')
print('- task2_02_sharp_changes.png')
print('- task2_03_sigma_comparison.png')
print('- telemetry_task2_result.csv')
print('- task2_summary_table.csv')