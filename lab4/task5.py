import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Генерація випадкових даних
m = 60  # Кількість точок
X = np.linspace(-3, 3, m)  # Створення рівномірного простору даних від -3 до 3
y = 4 + np.sin(X) + np.random.uniform(-0.6, 0.6, m)  # Цільова змінна з шумом

# Побудова графіка для візуалізації випадкових даних
plt.scatter(X, y, color='green', label='Данні')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Випадкові дані')
plt.legend()
plt.show()

# Лінійна регресія
linear_regressor = LinearRegression()  # Створення об'єкта лінійної регресії
linear_regressor.fit(X.reshape(-1, 1), y)  # Навчання моделі на даних
y_pred_linear = linear_regressor.predict(X.reshape(-1, 1))  # Прогнозування значень

# Побудова графіка для візуалізації результатів лінійної регресії
plt.scatter(X, y, color='green', label='Данні')
plt.plot(X, y_pred_linear, color='blue', linewidth=3, label='Лінійна регресія')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Лінійна регресія')
plt.legend()
plt.show()

# Поліноміальна регресія
poly_features = PolynomialFeatures(degree=3)  # Створення поліноміальної функції з ступенем 3
X_poly = poly_features.fit_transform(X.reshape(-1, 1))  # Перетворення X в поліноміальну форму
poly_regressor = LinearRegression()  # Створення об'єкта для поліноміальної регресії
poly_regressor.fit(X_poly, y)  # Навчання поліноміальної моделі
y_pred_poly = poly_regressor.predict(X_poly)  # Прогнозування значень

# Побудова графіка для візуалізації результатів поліноміальної регресії
plt.scatter(X, y, color='green', label='Данні')
plt.plot(np.sort(X), poly_regressor.predict(poly_features.transform(np.sort(X).reshape(-1, 1))), color='red', linewidth=3, label='Поліноміальна регресія')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Поліноміальна регресія (ступінь 3)')
plt.legend()
plt.show()

# Оцінка якості моделей
# Лінійна регресія
mse_linear = mean_squared_error(y, y_pred_linear)  # Обчислення середньоквадратичної похибки
r2_linear = r2_score(y, y_pred_linear)  # Обчислення коефіцієнта детермінації
mae_linear = mean_absolute_error(y, y_pred_linear)  # Обчислення середньої абсолютної похибки
print("Лінійна регресія:")
print("Середньоквадратична похибка:", round(mse_linear, 2))
print("R2:", round(r2_linear, 2))
print("Середня абсолютна похибка:", round(mae_linear, 2))

# Виведення коефіцієнтів та перехоплення для лінійної регресії
print("Коефіцієнти лінійної регресії:", linear_regressor.coef_)
print("Перехоплення лінійної регресії:", linear_regressor.intercept_)

# Поліноміальна регресія
mse_poly = mean_squared_error(y, y_pred_poly)  # Обчислення середньоквадратичної похибки для поліноміальної регресії
r2_poly = r2_score(y, y_pred_poly)  # Обчислення коефіцієнта детермінації для поліноміальної регресії
mae_poly = mean_absolute_error(y, y_pred_poly)  # Обчислення середньої абсолютної похибки для поліноміальної регресії
print("\nПоліноміальна регресія:")
print("Середньоквадратична похибка:", round(mse_poly, 2))
print("R2:", round(r2_poly, 2))
print("Середня абсолютна похибка:", round(mae_poly, 2))

# Виведення коефіцієнтів та перехоплення для поліноміальної регресії
print("Коефіцієнти поліноміальної регресії:", poly_regressor.coef_)
print("Перехоплення поліноміальної регресії:", poly_regressor.intercept_)
