# Вспомогательные библиотеки
import pandas as pd


pd.set_option('display.width', 160)

# Задание 1 (Матрица ошибок).
rows = [
    (1, 1, 1), (2, 0, 0), (3, 1, 1), (4, 1, 0), (5, 0, 1),
    (6, 0, 0), (7, 1, 1), (8, 0, 0), (9, 1, 0), (10, 0, 0),
]
df1 = pd.DataFrame(rows, columns=['#', 'Истина', 'Прогноз'])


def label(row):
    if row['Истина'] == 1 and row['Прогноз'] == 1:
        return 'TP'
    if row['Истина'] == 0 and row['Прогноз'] == 0:
        return 'TN'
    if row['Истина'] == 0 and row['Прогноз'] == 1:
        return 'FP'
    return 'FN'


df1['Тип'] = df1.apply(label, axis=1)

print("ЗАДАНИЕ 1) Матрица ошибок (построчно):")
print(df1.to_string(index=False))
counts = df1['Тип'].value_counts()
TP1 = int(counts.get('TP', 0))
TN1 = int(counts.get('TN', 0))
FP1 = int(counts.get('FP', 0))
FN1 = int(counts.get('FN', 0))
print(f"\nTP={TP1}, TN={TN1}, FP={FP1}, FN={FN1}")
print("\nМатрица ошибок (confusion matrix):")
conf = pd.DataFrame(
    [[TP1, FN1], [FP1, TN1]],
    index=['Истина=1 (авария)', 'Истина=0 (норма)'],
    columns=['Прогноз=1 (авария)', 'Прогноз=0 (норма)']
)
print(conf)
print('\n')

# Задания 2-5 (TP=32, TN=54, FP=6, FN=8).
TP, TN, FP, FN = 32, 54, 6, 8
total = TP + TN + FP + FN
accuracy = (TP + TN) / total
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)
print("ЗАДАНИЕ 2) Accuracy (детектор перегрева, TP=32,TN=54,FP=6,FN=8, n=100):")
print(f"Accuracy = ({TP} + {TN}) / ({TP} + {TN} + {FP} + {FN}) "
      f"= {TP+TN} / {total} = {accuracy:.4f} = {accuracy*100:.1f}%")
print('\n')

print("ЗАДАНИЕ 3) Precision:")
print(f"Precision = {TP} / ({TP} + {FP}) = {TP} / {TP + FP} = {precision:.4f} "
      f"= {precision*100:.1f}%")
print('\n')

print("ЗАДАНИЕ 4) Recall:")
print(f"Recall = {TP} / ({TP} + {FN}) = {TP} / {TP + FN} = {recall:.4f} = {recall*100:.1f}%")
print('\n')

print("ЗАДАНИЕ 5) F1-score:")
print(f"F1 = 2 * {precision:.4f} * {recall:.4f} / ({precision:.4f}+{recall:.4f})")
print(f"F1 = {f1:.4f} -> округлённо {round(f1, 2)}")
print('\n')

reg = [
    (1, 62, 60), (2, 65, 66), (3, 68, 70), (4, 70, 69), (5, 72, 75),
    (6, 75, 73), (7, 78, 80), (8, 80, 79), (9, 84, 86), (10, 88, 85),
]
dfr = pd.DataFrame(reg, columns=['#', 'Факт', 'Прогноз'])

# Задание 6 (ошибка e = y - y_hat).
dfr['Ошибка (e=y-ŷ)'] = dfr['Факт'] - dfr['Прогноз']
dfr['Ошибка^2'] = dfr['Ошибка (e=y-ŷ)'] ** 2
dfr['Ошибка'] = dfr['Ошибка (e=y-ŷ)'].abs()
dfr['Ошибка/Факт, %'] = 100 * dfr['Ошибка'] / dfr['Факт']
print("ЗАДАНИЕ 6) Таблица ошибок для регрессии:")
print(dfr.to_string(index=False))
print('\n')

# Задание 7 (MSE).
mse = dfr['Ошибка^2'].mean()
print("ЗАДАНИЕ 7) MSE:")
print(f"Сумма квадратов ошибок = {dfr["Ошибка^2"].sum():.0f}")
print(f"MSE = {dfr["Ошибка^2"].sum():.0f} / 10 = {mse:.2f} °C²")
print('\n')

# Задание 8 (MAE).
mae = dfr['Ошибка'].mean()
print("ЗАДАНИЕ 8) MAE:")
print(f"Сумма ошибок = {dfr["Ошибка"].sum():.0f}")
print(f"MAE = {dfr["Ошибка"].sum():.0f} / 10 = {mae:.2f} °C")
print('\n')

# Задание 9 (MAPE).
mape = dfr['Ошибка/Факт, %'].mean()
print("ЗАДАНИЕ 9) MAPE:")
print(f"MAPE = {mape:.2f} %")
print('\n')

# Итоговая сводка.
summary_class = pd.DataFrame([{
    'Accuracy': f'{accuracy:.2%}',
    'Precision': f'{precision:.2%}',
    'Recall': f'{recall:.2%}',
    'F1': round(f1, 2),
}])
summary_reg = pd.DataFrame([{
    'MSE, °C²': round(mse, 2),
    'MAE, °C': round(mae, 2),
    'MAPE, %': round(mape, 2),
}])

print("Итоговая сводка)")
print("Классификация:")
print(summary_class.to_string(index=False))
print("\nРегрессия:")
print(summary_reg.to_string(index=False))
print('\n')

# Результаты
df1.to_csv('task3_1_confusion_rows.csv', index=False)
conf.to_csv('task3_1_confusion_matrix.csv')
dfr.to_csv('task3_regression_errors.csv', index=False)
summary_class.to_csv('task3_classification_summary.csv', index=False)
summary_reg.to_csv('task3_regression_summary.csv', index=False)
