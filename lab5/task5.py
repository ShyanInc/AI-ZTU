import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesRegressor

# Завантаження даних із файлу traffic_data.txt
input_file = 'traffic_data.txt'  # Шлях до файлу з даними
data = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        items = line.strip().split(',')  # Видалення зайвих символів нового рядка та розділення рядків за комами
        data.append(items)
data = np.array(data)  # Перетворення списку в масив NumPy

# Кодування нечислових ознак
label_encoder = []  # Список для зберігання енкодерів для кожної категоріальної ознаки
X_encoded = np.empty(data.shape, dtype=object)  # Створюємо масив для збереження закодованих ознак

# Перевірка кожної ознаки, чи є вона числовою чи категоріальною
for i, item in enumerate(data[0]):  # Припускаємо, що перший рядок містить назви колонок
    if item.isdigit():  # Якщо ознака числова
        X_encoded[:, i] = data[:, i]  # Присвоюємо числові значення без змін
    else:  # Якщо ознака категоріальна
        label_encoder.append(preprocessing.LabelEncoder())  # Створюємо новий LabelEncoder для категоріальної ознаки
        X_encoded[:, i] = label_encoder[-1].fit_transform(data[:, i])  # Кодуємо категоріальні значення

# Розділення на ознаки (X) та мітки (y)
X = X_encoded[:, :-1].astype(int)  # Ознаки (все крім останнього стовпця)
y = X_encoded[:, -1].astype(int)   # Мітки (останній стовпець)

# Розбиття даних на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Створення регресора на основі ExtraTrees (Гранично випадковий ліс)
params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}  # Налаштування параметрів
regressor = ExtraTreesRegressor(**params)  # Ініціалізація регресора
regressor.fit(X_train, y_train)  # Навчання моделі на навчальних даних

# Оцінка ефективності моделі на тестових даних
y_pred = regressor.predict(X_test)  # Прогнозування на тестовому наборі
print("Mean absolute error:", round(mean_absolute_error(y_test, y_pred), 3))  # Виведення середньої абсолютної помилки

# Тестування кодування на одному прикладі
test_datapoint = ['Saturday', '10:20', 'Atlanta', 'no']  # Приклад даних для тестування
test_datapoint_encoded = [-1] * len(test_datapoint)  # Ініціалізація масиву для закодованих значень
count = 0  # Лічильник для категоріальних ознак

# Кодування кожної ознаки для тестового прикладу
for i, item in enumerate(test_datapoint):
    if not item.isdigit():  # Якщо ознака не числова, її потрібно закодувати
        test_datapoint_encoded[i] = label_encoder[count].transform([item])[0]  # Кодуємо категоріальне значення
        count += 1  # Збільшуємо лічильник для категоріальних ознак
    else:
        test_datapoint_encoded[i] = int(item)  # Якщо числова ознака, просто перетворюємо її на ціле число

test_datapoint_encoded = np.array(test_datapoint_encoded)  # Перетворюємо закодовані дані в масив NumPy

# Прогнозування трафіку для тестового прикладу
print("Predicted traffic:", int(regressor.predict([test_datapoint_encoded])[0]))  # Прогнозування та виведення результату
