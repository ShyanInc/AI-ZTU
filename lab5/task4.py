import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Завантаження набору даних про житло Каліфорнії
housing = datasets.fetch_california_housing()

# Перемішування даних для уникнення будь-якої систематичної залежності
X, y = shuffle(housing.data, housing.target, random_state=7)

# Розділення на навчальний та тестовий набори (80% для навчання, 20% для тестування)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Створення моделі AdaBoost з базовим регресором DecisionTreeRegressor
regressor = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=4),  # Використовуємо дерево рішень з обмеженою глибиною для зменшення складності
    n_estimators=400,  # Кількість дерев у лісі AdaBoost
    random_state=7
)

# Навчання моделі на навчальних даних
regressor.fit(X_train, y_train)

# Прогнозування значень на тестовому наборі
y_pred = regressor.predict(X_test)

# Обчислення середньоквадратичної помилки та поясненої дисперсії
mse = mean_squared_error(y_test, y_pred)  # Середньоквадратична помилка
evs = explained_variance_score(y_test, y_pred)  # Пояснена дисперсія

# Виведення оцінок моделі
print("АДАБОСТ РЕГРЕСОР")
print("Середньоквадратична помилка:", round(mse, 2))  # Виведення MSE
print("Пояснена дисперсія:", round(evs, 2))  # Виведення EVS

# Отримання важливості ознак
feature_importances = regressor.feature_importances_
feature_names = housing.feature_names  # Назви ознак у наборі даних

# Нормалізація важливості ознак (щоб максимальна важливість була рівна 100)
feature_importances = 100.0 * (feature_importances / max(feature_importances))

# Сортування ознак за їх важливістю (в порядку спадання)
index_sorted = np.argsort(feature_importances)[::-1]  # Сортуємо в зворотньому порядку

# Позиції для побудови графіка
pos = np.arange(index_sorted.shape[0]) + 0.5

# Побудова стовпчастої діаграми для важливості ознак
plt.figure()
plt.barh(pos, feature_importances[index_sorted], align='center')  # Створюємо горизонтальну діаграму
plt.yticks(pos, np.array(feature_names)[index_sorted])  # Встановлюємо назви ознак по осі Y
plt.xlabel('Важливість ознак')
plt.title('Важливість ознак для моделі AdaBoost')  # Заголовок графіка
plt.show()  # Відображаємо графік
