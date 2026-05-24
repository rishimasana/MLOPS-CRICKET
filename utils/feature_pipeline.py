import pandas as pd

FEATURES = [
    "Match ID",
    "Overs Played",
    "Wickets Lost",
    "Run Rate",
    "Home/Away",
    "Opponent Strength",
    "Pitch Condition",
    "Weather"
]


def validate_input(data):

    missing_features = []

    for feature in FEATURES:

        if feature not in data:
            missing_features.append(feature)

    if missing_features:
        return False, missing_features

    return True, []


def prepare_features(data):

    ordered_data = []

    for feature in FEATURES:
        ordered_data.append(data[feature])

    df = pd.DataFrame([ordered_data], columns=FEATURES)

    return df