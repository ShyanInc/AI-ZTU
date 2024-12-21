import pandas as pd
OUTLOOK = "Outlook"
HUMIDITY = "Humidity"
WIND = "Wind"
PLAY = "Play"

data = [
    ["Sunny", "High", "Weak", "No"],
    ["Sunny", "High", "Strong", "No"],
    ["Overcast", "High", "Weak", "Yes"],
    ["Rain", "High", "Weak", "Yes"],
    ["Rain", "Normal", "Weak", "Yes"],
    ["Rain", "Normal", "Strong", "No"],
    ["Overcast", "Normal", "Strong", "Yes"],
    ["Sunny", "High", "Weak", "No"],
    ["Sunny", "Normal", "Weak", "Yes"],
    ["Rain", "High", "Weak", "Yes"],
    ["Sunny", "Normal", "Strong", "Yes"],
    ["Overcast", "High", "Strong", "Yes"],
    ["Overcast", "Normal", "Weak", "Yes"],
    ["Rain", "High", "Strong", "No"]
]

def create_dataframe(data, columns):
    """
    Create a pandas DataFrame from input data.
    """
    return pd.DataFrame(data, columns=columns)

def calculate_frequency_table(df):
    return df.groupby([OUTLOOK, PLAY]).size().unstack()

def calculate_probabilities(df):
    total = len(df)
    total_yes = len(df[df[PLAY] == "Yes"])
    total_no = len(df[df[PLAY] == "No"])

    prob_yes = total_yes / total
    prob_no = total_no / total

    return prob_yes, prob_no

def display_results(frequency_table, prob_yes, prob_no):
    print("Частотна таблиця:")
    print(frequency_table)
    
    print("\nЙмовірності:")
    print(f"Ймовірність гри (Yes): {prob_yes:.2f}")
    print(f"Ймовірність відмови (No): {prob_no:.2f}")

def main():
    df = create_dataframe(data, [OUTLOOK, HUMIDITY, WIND, PLAY])
    frequency_table = calculate_frequency_table(df)
    prob_yes, prob_no = calculate_probabilities(df)
    
    display_results(frequency_table, prob_yes, prob_no)

if __name__ == "__main__":
    main()
