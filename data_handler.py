import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


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
                         dtype={"timestamp": 'int64', "open": float, "high": float, "low": float, "close": float})
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

    choices = [1, 2]

    df['Trend'] = np.select(conditions, choices, default=0)

    # df['Label'] = np.select(conditions, choices, default=0)
    # df['LabelSum'] = df['Label'].rolling(window=sum_window).sum()
    #
    # df['Trend'] = df['LabelSum'].apply(lambda x: detect_trend(x, threshold))
    # Drop the intermediate columns used for calculations
    # df.drop(['Close_5', 'Close_10', 'Label', 'LabelSum'], axis=1, inplace=True)
    df.drop(['Close_5', 'Close_10'], axis=1, inplace=True)

    return df


def split_and_normalize_val(df, val_split_date, test_split_date):
    # Splitting the DataFrame into features (X) and target (y)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_train = df[df['Date'] < val_split_date]
    y_train = df_train['Trend']
    X_train = df_train.drop(['Date', 'Trend'], axis=1)

    df_val = df[(df['Date'] >= val_split_date) & (df['Date'] < test_split_date)]
    y_val = df_val['Trend']
    X_val = df_val.drop(['Date', 'Trend'], axis=1)

    column_names = X_train.columns

    # Using StandardScaler to normalize train feature vector
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(data=X_train_scaled, columns=column_names)

    # Applying the same transformation to test feature vector
    X_val_scaled = scaler.transform(X_val)
    X_val_scaled = pd.DataFrame(data=X_val_scaled, columns=column_names)

    return X_train_scaled, X_val_scaled, y_train, y_val, X_val


def split_and_normalize_test(df, test_split_date):
    # Splitting the DataFrame into features (X) and target (y)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_train = df[df['Date'] < test_split_date]
    y_train = df_train['Trend']
    X_train = df_train.drop(['Date', 'Trend'], axis=1)

    df_test = df[df['Date'] >= test_split_date]
    y_test = df_test['Trend']
    X_test = df_test.drop(['Date', 'Trend'], axis=1)

    column_names = X_train.columns

    # Using StandardScaler to normalize train feature vector
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(data=X_train_scaled, columns=column_names)

    # Applying the same transformation to test feature vector
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(data=X_test_scaled, columns=column_names)

    return X_train_scaled, X_test_scaled, y_train, y_test, X_test


def create_dir_if_not_exist(dir_path):
    os.makedirs(dir_path, exist_ok=True)
