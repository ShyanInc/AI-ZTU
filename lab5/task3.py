import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier

# Завантаження даних із файлу
input_file = 'data_random_forests.txt'  # Шлях до файлу з даними
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]  # Розділення на ознаки та мітки класів

# Розподіл даних на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5  # 25% даних використовуються для тестування
)

# Опис сітки параметрів для налаштування моделі
parameter_grid = {
    'n_estimators': [25, 50, 100, 250],  # Кількість дерев в лісі
    'max_depth': [2, 4, 8, 12, 16]  # Максимальна глибина дерев
}

metrics = ['precision_weighted', 'recall_weighted']  # Метрики для оцінки

# Перебір різних метрик для вибору оптимальних параметрів
for metric in metrics:
    print("\n### Searching optimal parameters for", metric)

    # Створення класифікатора з пошуком по сітці параметрів
    classifier = GridSearchCV(
        ExtraTreesClassifier(random_state=0),  # Використовуємо ExtraTreesClassifier
        parameter_grid,
        cv=5,  # Кількість фолдів для крос-валідації
        scoring=metric  # Оцінка на основі поточної метрики
    )

    # Навчання моделі на тренувальних даних
    classifier.fit(X_train, y_train)

    # Виведення результатів перебору параметрів
    print("\nGrid scores for the parameter grid:\n")
    results = classifier.cv_results_
    for mean, params in zip(results['mean_test_score'], results['params']):
        print(params, '-->', round(mean, 3))  # Виведення середнього результату для кожної комбінації параметрів

    print("\nBest parameters for", metric, ":\n", classifier.best_params_)  # Виведення найкращих параметрів

# Оцінка якості класифікатора на тестовому наборі
print("\nPerformance report on test set:\n")
y_pred = classifier.predict(X_test)  # Прогнозування на тестових даних
print(classification_report(y_test, y_pred))  # Виведення звіту про класифікацію
