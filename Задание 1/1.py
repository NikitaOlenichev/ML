# Вспомогательные библиотеки
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Константы для проверки аномалий
TEMP = 60.0
VOLT = 23.5
CURR = 8.0
OMEGA = 1.0
DTEMP = 3.0

pd.set_option('display.width', 160)
pd.set_option('display.max_columns', 10)

# 1. Загрузка данных
df = pd.read_csv('telemetry_dzz_sem1.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# 2. Первые строки таблицы, размер набора данных и основные статистические характеристики
print("1) Первые строки таблицы:")
print(df.head(10))
print('\n')

print("2) Размер набора данных:")
print(f"Строк: {df.shape[0]}, столбцов: {df.shape[1]}")
print(f"Период: {df['timestamp'].min()} - {df['timestamp'].max()}")
print(f"Режим работы: {df['mode'].unique().tolist()}")
print(df["mode"].value_counts())
print('\n')

print("3) Основные статистические характеристики:")
print(df[['temperature', 'voltage', 'current', 'angular_velocity']].describe())
print("\nСтатистика по параметрам в разрезе режима работы (mode):")
print(df.groupby('mode')[['temperature', 'voltage', 'current',
                          'angular_velocity']].agg(['mean', 'min', 'max']))
print('\n')

# 3. Графики изменения параметров во времени
params = ['temperature', 'voltage', 'current', 'angular_velocity']
titles = {'temperature': 'Температура, °C', 'voltage': 'Напряжение, В', 'current': 'Сила тока, А',
          'angular_velocity': 'Угловая скорость'}
colors = {'temperature': 'crimson', 'voltage': 'royalblue',
          'current': 'darkorange', 'angular_velocity': 'seagreen'}
fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
for ax, p in zip(axes, params):
    ax.plot(df['timestamp'], df[p], color=colors[p], linewidth=0.9)
    ax.set_ylabel(titles[p])
    ax.grid(alpha=0.3)
axes[0].set_title('Телеметрия спутника: температура, напряжение, ток, угловая скорость')
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
axes[-1].set_xlabel('Время')
fig.tight_layout()
fig.savefig('01_telemetry_raw.png', dpi=130)
plt.close(fig)

# 4. Анализ и выбор правил
df['d_temp'] = df['temperature'].diff().abs()
df['mode_changed'] = df['mode'] != df['mode'].shift(1)

# 5-6. Реализация правил + столбец anomaly
rule_temp = df['temperature'] > TEMP
rule_volt = df['voltage'] < VOLT
rule_curr = df['current'] > CURR
rule_omega = df['angular_velocity'] > OMEGA
rule_dtemp = (df['d_temp'] > DTEMP) & (~df['mode_changed'])
df['anomaly'] = (rule_temp | rule_volt | rule_curr | rule_omega | rule_dtemp).astype(int)
print("4-6) Результат работы детектора:")
print(f"Всего аномальных точек: {df['anomaly'].sum()} из {len(df)} "
      f"({100 * df['anomaly'].mean():.1f}%)")
print('Срабатывания по правилам:')
print(f"temperature > {TEMP}: {rule_temp.sum()}")
print(f"voltage < {VOLT}:     {rule_volt.sum()}")
print(f"current > {CURR}:      {rule_curr.sum()}")
print(f"omega > {OMEGA}:        {rule_omega.sum()}")
print(f"d_temp > {DTEMP}:       {rule_dtemp.sum()}")
print('\n')

# 7. Временные интервалы аномалий
def get_intervals(frame, fl='anomaly'):
    flags = frame[fl].values
    ts = frame['timestamp'].values
    intervals = []
    start = None
    for i, f in enumerate(flags):
        if f and start is None:
            start = ts[i]
        if (not f or i == len(flags) - 1) and start is not None:
            end = ts[i - 1] if not f else ts[i]
            intervals.append((start, end))
            start = None
    return intervals


intervals = get_intervals(df)
print("7) Интервалы потенциально нештатных ситуаций:")
for i, (s, e) in enumerate(intervals, 1):
    s = pd.Timestamp(s)
    e = pd.Timestamp(e)
    sub = df[(df['timestamp'] >= s) & (df['timestamp'] <= e)]
    reasons = []
    if (sub['temperature'] > TEMP).any():
        reasons.append('перегрев')
    if (sub['voltage'] < VOLT).any():
        reasons.append('просадка напряжения')
    if (sub['current'] > CURR).any():
        reasons.append('скачок тока')
    if (sub['angular_velocity'] > OMEGA).any():
        reasons.append('аномальное вращение')
    if (sub['d_temp'] > DTEMP).any():
        reasons.append('резкий скачок температуры')
    dur = (e - s).total_seconds() / 60 + 1
    print(f'[{i}] {s} - {e} (~{dur:.0f} мин) режим: {sub["mode"].unique().tolist()} '
          f'причины: {", ".join(reasons)}')
print('\n')

# 8. Разметка аномалий на графиках
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
for ax, p in zip(axes, params):
    ax.plot(df['timestamp'], df[p], color=colors[p], linewidth=0.9, zorder=2)
    ax.scatter(df.loc[df['anomaly'] == 1, 'timestamp'], df.loc[df['anomaly'] == 1, p],
               color='black', s=10, zorder=3, label='anomaly=1')
    for s, e in intervals:
        ax.axvspan(s, e, color='red', alpha=0.15)
    ax.set_ylabel(titles[p])
    ax.grid(alpha=0.3)
axes[0].set_title('Телеметрия с разметкой потенциально нештатных интервалов')
axes[0].legend(loc='upper right')
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
axes[-1].set_xlabel('Время')
fig.tight_layout()
fig.savefig('02_telemetry_anomalies.png', dpi=130)
plt.close(fig)

# Дополнительная часть: чувствительный и консервативный детекторы
def sensitive_detector(frame):
    same_mode = ~frame['mode_changed']
    return (
            (frame['temperature'] > 40) |
            (frame['voltage'] < 27.0) |
            (frame['current'] > 6.5) |
            (frame['angular_velocity'] > 0.6) |
            ((frame['temperature'].diff().abs() > 1.5) & same_mode)
    ).astype(int)


def conservative_detector(frame):
    same_mode = ~frame['mode_changed']
    return (
            (frame['temperature'] > 62) |
            (frame['voltage'] < 24.0) |
            (frame['current'] > 9.0) |
            (frame['angular_velocity'] > 1.5) |
            ((frame['temperature'].diff().abs() > 5.0) & same_mode)
    ).astype(int)


df['anomaly_sensitive'] = sensitive_detector(df)
df['anomaly_conservative'] = conservative_detector(df)
intervals_sens = get_intervals(df, 'anomaly_sensitive')
intervals_cons = get_intervals(df, 'anomaly_conservative')
print("Дополнительная часть (сравнение детекторов))")
print(f"Детектор №1 (чувствительный):  {df["anomaly_sensitive"].sum()} точек, "
      f"{len(intervals_sens)} интервал(ов)")
print(f"Детектор №2 (консервативный):  {df["anomaly_conservative"].sum()} точек, "
      f"{len(intervals_cons)} интервал(ов)")
print(f"Базовый детектор:              {df["anomaly"].sum()} точек, "
      f"{len(intervals)} интервал(ов)")
print('\n')

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df['timestamp'], df['temperature'], color='gray', linewidth=0.8, label='temperature')
ax.scatter(df.loc[df['anomaly_sensitive'] == 1, 'timestamp'],
           df.loc[df['anomaly_sensitive'] == 1, 'temperature'],
           color='orange', s=14,
           label=f'Детектор №1 чувствительный (n={df["anomaly_sensitive"].sum()})', zorder=3)
ax.scatter(df.loc[df['anomaly_conservative'] == 1, 'timestamp'],
           df.loc[df['anomaly_conservative'] == 1, 'temperature'],
           color='darkred', s=20, marker='x',
           label=f'Детектор №2 консервативный (n={df["anomaly_conservative"].sum()})', zorder=4)
ax.set_ylabel('Температура, °C')
ax.set_title('Сравнение чувствительного и консервативного детекторов')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
fig.tight_layout()
fig.savefig('03_detectors_comparison.png', dpi=130)
plt.close(fig)

# Финальная таблица
df.to_csv('telemetry_with_anomalies.csv', index=False)

# Получившиеся файлы
print('Получившиеся файлы:')
print('- 01_telemetry_raw.png')
print('- 02_telemetry_anomalies.png')
print('- 03_detectors_comparison.png')
print('- telemetry_with_anomalies.csv')


