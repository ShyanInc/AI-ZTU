import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_data(url):
    try:
        data = pd.read_csv(url)
        print("Data loaded successfully!")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(data):
    data = data.dropna()

    categorical_columns = ['train_type', 'train_class', 'fare']
    for col in categorical_columns:
        data[col] = pd.factorize(data[col])[0]

    y = (data['price'] > data['price'].median()).astype(int)
    
    X = data[['train_type', 'train_class', 'fare']]
    
    return X, y

def train_model(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    print("Model trained successfully!")
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))


def main():
    url = "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/renfe_small.csv"
    
    data = load_data(url)
    if data is None:
        return
    
    X, y = preprocess_data(data)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train)
    
    evaluate_model(model, X_test, y_test)

    print("\nPrice Statistics:")
    print(data['price'].describe())


if __name__ == "__main__":
    main()
