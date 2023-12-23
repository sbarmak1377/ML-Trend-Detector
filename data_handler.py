import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def read_csv_data(file_path, mode):
    if mode == "yahoo_finance":
        # Read data with dtype specified
        columns = ["Date", "Open", "High", "Low", "Close"]
        date_parser = lambda x: pd.to_datetime(x, format="%Y-%m-%d")
        df = pd.read_csv(file_path, parse_dates=["Date"],
                         dtype={"Date": "str", "Open": float, "High": float, "Low": float, "Close": float},
                         date_parser=date_parser)

    elif mode == "dukas_copy":
        # Read data with timestamp and convert to datetime
        columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        df = pd.read_csv(file_path,
                         dtype={"timestamp": int, "open": float, "high": float, "low": float, "close": float})
        df["Date"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df[["Date", "open", "high", "low", "close", "volume"]]

    else:
        raise ValueError("Invalid mode. Use either 'yahoo_finance' or 'dukas_copy'.")

    # Rename columns and drop old columns
    df.columns = columns
    return df


def detect_trend(label_sum, threshold):
    if label_sum >= threshold:
        return 0
    elif label_sum <= -1 * threshold:
        return 1
    else:
        return 2


def create_label_column(df):
    # Create columns for Close shifted by 5 and 10 rows
    df['Close_5'] = df['Close'].shift(-5)
    df['Close_10'] = df['Close'].shift(-10)

    # Create the Label column based on the specified conditions
    conditions = [
        (df['Close_10'] > df['Close_5']) & (df['Close_5'] > df['Close']),
        (df['Close_10'] < df['Close_5']) & (df['Close_5'] < df['Close'])
    ]

    choices = [1, -1]

    df['Trend'] = np.select(conditions, choices, default=0)

    # df['Label'] = np.select(conditions, choices, default=0)
    # df['LabelSum'] = df['Label'].rolling(window=sum_window).sum()
    #
    # df['Trend'] = df['LabelSum'].apply(lambda x: detect_trend(x, threshold))
    # Drop the intermediate columns used for calculations
    # df.drop(['Close_5', 'Close_10', 'Label', 'LabelSum'], axis=1, inplace=True)
    df.drop(['Close_5', 'Close_10'], axis=1, inplace=True)

    return df


def split_and_normalize(df):
    # Splitting the DataFrame into features (X) and target (y)
    X = df.drop('Trend', axis=1)
    y = df['Trend']

    # Splitting the data into 90% train and 10% test without shuffling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
    X_train.drop('Date', axis=1, inplace=True)
    X_test.drop('Date', axis=1, inplace=True)

    # Replace the `inf` values with `NaN`
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

    column_names = X_train.columns

    # Using StandardScaler to normalize train feature vector
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(data=X_train_scaled, columns=column_names)

    # Applying the same transformation to test feature vector
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(data=X_test_scaled, columns=column_names)

    return X_train_scaled, X_test_scaled, y_train, y_test, X_test


def create_dir_if_not_exist(dir_path):
    os.makedirs(dir_path, exist_ok=True)
