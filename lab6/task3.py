import pandas as pd
from typing import Tuple, List

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


def create_dataframe(data: List[List[str]]) -> pd.DataFrame:
    """Create a pandas DataFrame from input data."""
    columns = [OUTLOOK, HUMIDITY, WIND, PLAY]
    return pd.DataFrame(data, columns=columns)


def calculate_conditional_probabilities(df: pd.DataFrame, total_yes: int, total_no: int) -> Tuple[float, float, float, float, float, float]:
    prob_outlook_sunny_yes = len(df[(df[OUTLOOK] == "Sunny") & (df[PLAY] == "Yes")]) / total_yes
    prob_humidity_high_yes = len(df[(df[HUMIDITY] == "High") & (df[PLAY] == "Yes")]) / total_yes
    prob_wind_weak_yes = len(df[(df[WIND] == "Weak") & (df[PLAY] == "Yes")]) / total_yes

    prob_outlook_sunny_no = len(df[(df[OUTLOOK] == "Sunny") & (df[PLAY] == "No")]) / total_no
    prob_humidity_high_no = len(df[(df[HUMIDITY] == "High") & (df[PLAY] == "No")]) / total_no
    prob_wind_weak_no = len(df[(df[WIND] == "Weak") & (df[PLAY] == "No")]) / total_no

    return prob_outlook_sunny_yes, prob_humidity_high_yes, prob_wind_weak_yes, \
           prob_outlook_sunny_no, prob_humidity_high_no, prob_wind_weak_no


def calculate_bayes_probabilities(prob_yes: float, prob_no: float,
                                  prob_outlook_sunny_yes: float, prob_humidity_high_yes: float,
                                  prob_wind_weak_yes: float, prob_outlook_sunny_no: float,
                                  prob_humidity_high_no: float, prob_wind_weak_no: float) -> Tuple[float, float]:
    prob_yes_given_conditions = prob_yes * prob_outlook_sunny_yes * prob_humidity_high_yes * prob_wind_weak_yes
    prob_no_given_conditions = prob_no * prob_outlook_sunny_no * prob_humidity_high_no * prob_wind_weak_no

    return prob_yes_given_conditions, prob_no_given_conditions


def display_results(prob_yes_given_conditions: float, prob_no_given_conditions: float) -> None:
    print(f"Probability of 'Yes': {prob_yes_given_conditions}")
    print(f"Probability of 'No': {prob_no_given_conditions}")

    if prob_yes_given_conditions > prob_no_given_conditions:
        print("\nThe match will take place!")
    else:
        print("\nThe match will not take place.")


def main() -> None:
    df = create_dataframe(data)
    total = len(df)
    total_yes = len(df[df[PLAY] == "Yes"])
    total_no = len(df[df[PLAY] == "No"])

    prob_yes = total_yes / total
    prob_no = total_no / total
    prob_outlook_sunny_yes, prob_humidity_high_yes, prob_wind_weak_yes, \
    prob_outlook_sunny_no, prob_humidity_high_no, prob_wind_weak_no = \
        calculate_conditional_probabilities(df, total_yes, total_no)
    prob_yes_given_conditions, prob_no_given_conditions = calculate_bayes_probabilities(
        prob_yes, prob_no, prob_outlook_sunny_yes, prob_humidity_high_yes, prob_wind_weak_yes,
        prob_outlook_sunny_no, prob_humidity_high_no, prob_wind_weak_no
    )

    display_results(prob_yes_given_conditions, prob_no_given_conditions)


if __name__ == "__main__":
    main()
