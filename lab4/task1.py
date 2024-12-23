import pickle
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt
# Шлях до файлу з даними
input_file = 'data_singlevar_regr.txt'
# Завантаження даних з текстового файлу
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]
# Поділ даних на навчальний і тестовий набори
num_training = int(0.8 * len(X))
num_test = len(X) - num_training
# Навчальний набір
X_train, y_train = X[:num_training], y[:num_training]
# Тестовий набір
X_test, y_test = X[num_training:], y[num_training:]
# Ініціалізація лінійного регресора
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)
# Прогнозування результатів
y_test_pred = regressor.predict(X_test)
# Візуалізація фактичних і прогнозованих даних
plt.scatter(X_test, y_test, color='green', label='Фактичні значення')
plt.plot(X_test, y_test_pred, color='black', linewidth=2, label='Прогнозовані значення')
plt.xlabel('Вхідні дані X')
plt.ylabel('Вихідні дані y')
plt.legend()
plt.show()
print("Результати роботи лінійного регресора:")
print("Середня абсолютна помилка =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Середньоквадратична помилка =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Медіанна абсолютна помилка =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Пояснена дисперсія =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 коефіцієнт =", round(sm.r2_score(y_test, y_test_pred), 2))
# Шлях до файлу для збереження моделі
output_model_file = 'model.pkl'
# Збереження моделі у файл
with open(output_model_file, 'wb') as f:
    pickle.dump(regressor, f)
# Завантаження моделі з файлу
with open(output_model_file, 'rb') as f:
    regressor_model = pickle.load(f)
# Прогнозування з використанням завантаженої моделі
y_test_pred_new = regressor_model.predict(X_test)
print("\nОновлена середня абсолютна помилка =", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))
