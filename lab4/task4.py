import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Завантаження набору даних для діабету
diabetes = datasets.load_diabetes()
X = diabetes.data  # Вхідні дані
y = diabetes.target  # Цільова змінна

# Розподіл даних на навчальну та тестову вибірки (50% для кожної)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=0)

# Створення та тренування моделі лінійної регресії
regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)

# Прогнозування за допомогою натренованої моделі
ypred = regr.predict(Xtest)

# Виведення коефіцієнтів моделі та значення перехоплення (intercept)
print("Коефіцієнти регресії:", regr.coef_)
print("Перехоплення (intercept):", regr.intercept_)

# Обчислення основних показників якості моделі
r2 = r2_score(ytest, ypred)  # R2 показник якості моделі
mae = mean_absolute_error(ytest, ypred)  # Середня абсолютна похибка
mse = mean_squared_error(ytest, ypred)  # Середня квадратична похибка

# Виведення результатів оцінки моделі
print("R2 score:", round(r2, 2))
print("Mean Absolute Error:", round(mae, 2))
print("Mean Squared Error:", round(mse, 2))

# Побудова графіка для порівняння фактичних та прогнозованих значень
fig, ax = plt.subplots()
ax.scatter(ytest, ypred, edgecolors=(0, 0, 0))  # Точки для тестових даних
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)  # Лінія, що відображає ідеальні прогнози
ax.set_xlabel('Виміряно')  # Підпис для фактичних значень
ax.set_ylabel('Передбачено')  # Підпис для прогнозованих значень
plt.show()  # Відображення графіка
