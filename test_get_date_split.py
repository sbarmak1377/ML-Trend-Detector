import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from data_handler import read_csv_data, create_label_column
from feature_extract import create_features

for time_frame in ["d1", "h4"]:
    for pair_curr in ["audusd", "gbpusd", "eurusd", "usdcad"]:
        print(time_frame + ": " + pair_curr)
        imbalance_solution = "down"
        data_path = 'data/dukascopy/forex/' + time_frame + '/' + pair_curr + '.csv'
        data_mode = "dukas_copy"

        df = read_csv_data(file_path=data_path, mode=data_mode)

        df = create_label_column(df)

        df = create_features(df)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        X = df.drop('Trend', axis=1)
        y = df['Trend']

        # Splitting the data into 90% train and 10% test without shuffling
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
        print("Train:")
        print(X_train['Date'].head())
        print('*' * 10)
        print(X_train['Date'].tail())
        print('*' * 10)
        print("Test")
        print(X_test['Date'].head())
        print('*' * 10)
        print(X_test['Date'].tail())
        print("*" * 100)
