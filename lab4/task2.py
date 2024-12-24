import pickle
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt
# Шлях до файлу з даними
input_file = 'data_regr_1.txt'
# Завантаження вхідних даних з файлу
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, 0].reshape(-1, 1), data[:, 1]  # Виділення вхідної змінної (X) та цільової змінної (y)
# Поділ даних на навчальний та тестовий набори
num_training = int(0.8 * len(X))
num_test = len(X) - num_training
# Вибір навчальних даних
X_train, y_train = X[:num_training], y[:num_training]
# Вибір тестових даних
X_test, y_test = X[num_training:], y[num_training:]
# Ініціалізація моделі лінійної регресії
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)
# Прогнозування на тестових даних
y_test_pred = regressor.predict(X_test)
# Побудова графіка для візуалізації результатів
plt.scatter(X_test, y_test, color='green', label='Фактичні дані')
plt.plot(X_test, y_test_pred, color='black', linewidth=2, label='Прогноз')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
print("Результати лінійного регресора:")
print("Середня абсолютна похибка =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Середньоквадратична похибка =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Медіанна абсолютна похибка =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Коефіцієнт детермінації =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 =", round(sm.r2_score(y_test, y_test_pred), 2))
# Шлях для збереження моделі
output_model_file = 'model.pkl'
# Збереження моделі у файл
with open(output_model_file, 'wb') as f:
    pickle.dump(regressor, f)
# Завантаження моделі з файлу
with open(output_model_file, 'rb') as f:
    regressor_model = pickle.load(f)
# Прогнозування за допомогою завантаженої моделі
y_test_pred_new = regressor_model.predict(X_test)
print("\nНова середня абсолютна похибка =", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))
