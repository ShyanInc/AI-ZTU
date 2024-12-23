import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# Створення графічного зображення меж класифікації для моделі
def plot_decision_boundaries(classifier, X, y, title, subplot_position):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax = plt.subplot(3, 2, subplot_position)
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.Paired)
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax)

# Завантаження та обробка даних з текстового файлу
input_file = 'data_random_forests.txt'  # Вкажіть шлях до файлу, якщо необхідно
data = np.loadtxt(input_file, delimiter=',')

X, y = data[:, :-1], data[:, -1]

# Виділення даних для трьох класів
class_0 = X[y == 0]
class_1 = X[y == 1]
class_2 = X[y == 2]

# Відображення даних на двовимірному графіку
plt.figure(figsize=(12, 6))
plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='white', edgecolors='black', linewidth=1, marker='s', label='Class-0')
plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white', edgecolors='black', linewidth=1, marker='o', label='Class-1')
plt.scatter(class_2[:, 0], class_2[:, 1], s=75, facecolors='white', edgecolors='black', linewidth=1, marker='^', label='Class-2')
plt.title('Графічне зображення вхідних даних')
plt.legend()
plt.show()

# Розділення даних на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

# Параметри для створення класифікаторів
params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}

# Ініціалізація двох моделей: RandomForest і ExtraTrees
rf_classifier = RandomForestClassifier(**params)
erf_classifier = ExtraTreesClassifier(**params)

# Навчання класифікаторів на тренувальних даних
rf_classifier.fit(X_train, y_train)
erf_classifier.fit(X_train, y_train)

# Візуалізація меж класифікації для кожної моделі
plt.figure(figsize=(12, 12))

# Візуалізація для RandomForestClassifier на навчальних даних
plot_decision_boundaries(rf_classifier, X_train, y_train, 'Random Forest (Тренування)', 1)

# Візуалізація для ExtraTreesClassifier на навчальних даних
plot_decision_boundaries(erf_classifier, X_train, y_train, 'Extra Trees (Тренування)', 2)

# Візуалізація для RandomForestClassifier на тестових даних
plot_decision_boundaries(rf_classifier, X_test, y_test, 'Random Forest (Тест)', 3)

# Візуалізація для ExtraTreesClassifier на тестових даних
plot_decision_boundaries(erf_classifier, X_test, y_test, 'Extra Trees (Тест)', 4)

# Тестові точки для прогнозування
test_datapoints = np.array([[5, 5], [3, 6], [6, 4], [7, 2], [4, 4], [5, 2]])
plot_decision_boundaries(rf_classifier, test_datapoints, [0] * len(test_datapoints), 'Random Forest (Тестові точки)', 5)

# Візуалізація тестових точок для ExtraTreesClassifier
plot_decision_boundaries(erf_classifier, test_datapoints, [0] * len(test_datapoints), 'Extra Trees (Тестові точки)', 6)

plt.show()

# Оцінка точності моделей на тестових даних
y_test_pred_rf = rf_classifier.predict(X_test)
y_test_pred_erf = erf_classifier.predict(X_test)

# Виведення статистики для RandomForestClassifier
class_names = ['Class-0', 'Class-1', 'Class-2']
print("\n" + "#" * 40)
print("\nОцінка результатів RandomForestClassifier на тренувальних даних\n")
print(classification_report(y_train, rf_classifier.predict(X_train), target_names=class_names))
print("#" * 40 + "\n")

print("#" * 40)
print("\nОцінка результатів RandomForestClassifier на тестових даних\n")
print(classification_report(y_test, y_test_pred_rf, target_names=class_names))
print("#" * 40 + "\n")

# Виведення статистики для ExtraTreesClassifier
print("\n" + "#" * 40)
print("\nОцінка результатів ExtraTreesClassifier на тренувальних даних\n")
print(classification_report(y_train, erf_classifier.predict(X_train), target_names=class_names))
print("#" * 40 + "\n")

print("#" * 40)
print("\nОцінка результатів ExtraTreesClassifier на тестових даних\n")
print(classification_report(y_test, y_test_pred_erf, target_names=class_names))
print("#" * 40 + "\n")

# Прогнозування ймовірностей для тестових точок з обох моделей
print("\nАналіз ймовірностей для RandomForestClassifier:")
for datapoint in test_datapoints:
    probabilities = rf_classifier.predict_proba([datapoint])[0]
    predicted_class = 'Class-' + str(np.argmax(probabilities))
    print('\nТочка:', datapoint)
    print('Прогнозована категорія:', predicted_class)

print("\nАналіз ймовірностей для ExtraTreesClassifier:")
for datapoint in test_datapoints:
    probabilities = erf_classifier.predict_proba([datapoint])[0]
    predicted_class = 'Class-' + str(np.argmax(probabilities))
    print('\nТочка:', datapoint)
    print('Прогнозована категорія:', predicted_class)
