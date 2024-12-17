import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from matplotlib import pyplot as plt


input_file = "income_data.txt"  
names = ['feature1', 'feature2', 'feature3', 'feature4', 'target']


dataset = pd.read_csv(input_file, delim_whitespace=True, header=None, names=names)


print(dataset.head())


dataset = dataset.replace(',', '', regex=True)


dataset = dataset.apply(pd.to_numeric, errors='coerce')


print(dataset.isnull().sum())


dataset = dataset.dropna()

print(f"Розмір даних після очищення: {dataset.shape}")


if not dataset.empty:
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

    models = []
    models.append(('LR', LogisticRegression(solver='liblinear')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))

    results = []
    names = []
    scoring = 'accuracy'

    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        print(f"{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})")

    plt.boxplot(results, labels=names)
    plt.title('Algorithm Comparison')
    plt.show()
else:
    print("Набір даних порожній після очищення!")