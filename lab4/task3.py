import numpy as np
import pickle
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt  # Імпорт бібліотеки matplotlib для побудови графічного представлення результатів

# Вказівка шляху до вхідного файлу, що містить дані
input_file = 'data_multivar_regr.txt'

# Завантаження даних з файлу з роздільником кома
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, 0].reshape(-1, 1), data[:, 1]  # Розподіл даних на змінну X та цільову змінну y

# Визначення розміру навчальної та тестової вибірок
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# Виділення навчальних даних
X_train, y_train = X[:num_training], y[:num_training]

# Виділення тестових даних
X_test, y_test = X[num_training:], y[num_training:]

# Створення та навчання моделі лінійної регресії
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)

# Прогнозування значень для тестових даних
y_test_pred = regressor.predict(X_test)

# Візуалізація результатів у вигляді графіка
plt.scatter(X_test, y_test, color='green', label='Фактичні дані')
plt.plot(X_test, y_test_pred, color='black', linewidth=2, label='Прогноз')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Оцінка якості моделі за допомогою кількох метрик
print("Результати лінійного регресора:")
print("Середня абсолютна похибка =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Середньоквадратична похибка =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Медіанна абсолютна похибка =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Коефіцієнт детермінації =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 =", round(sm.r2_score(y_test, y_test_pred), 2))

# Визначення шляху для збереження моделі
output_model_file = 'model.pkl'

# Збереження тренованої моделі в файл
with open(output_model_file, 'wb') as f:
    pickle.dump(regressor, f)

# Завантаження моделі з файлу
with open(output_model_file, 'rb') as f:
    regressor_model = pickle.load(f)

# Прогнозування за допомогою завантаженої моделі
y_test_pred_new = regressor_model.predict(X_test)
print("\nНова середня абсолютна похибка =", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))

# Трансформація даних для поліноміальної регресії з використанням ступеня 10
polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)

# Створення нових даних для прогнозування
datapoint = [[7.75]]  # Приклад одного нового значення для передбачення
# Трансформація нового значення згідно з поліноміальною регресією
poly_datapoint = polynomial.transform(datapoint)

# Створення та навчання поліноміальної моделі
poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)

# Прогнозування для нового значення за допомогою поліноміальної моделі
print("\nПрогноз поліноміальної регресії для точки даних", datapoint, ":\n", poly_linear_model.predict(poly_datapoint))
