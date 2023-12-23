import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import KFold


def select_features(X_train, y_train, save_feature_bars_path, threshold=0.9):
    # Train RandomForest and ExtraTreeClassifier
    et_classifier = ExtraTreesClassifier()

    et_classifier.fit(X_train, y_train)

    # Get feature importances from both models
    et_feature_importances = et_classifier.feature_importances_

    plt.rcParams.update({'font.size': 12})

    # Sort features based on importance
    sorted_indices = et_feature_importances.argsort()[::-1]

    # Calculate cumulative importance
    cumulative_importance = et_feature_importances[sorted_indices].cumsum()

    # Find the threshold for 80% cumulative importance
    selected_features = X_train.columns[sorted_indices[cumulative_importance <= threshold]]

    # Plot feature importance
    plt.figure(figsize=(20, 12))
    plt.bar(range(len(sorted_indices)), et_feature_importances[sorted_indices], align="center")
    plt.xticks(range(len(sorted_indices)), X_train.columns[sorted_indices], rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Selected Feature Importance")
    plt.savefig(save_feature_bars_path + "selected_features.png")

    return selected_features


def reshape_x(x: pd.DataFrame):
    return x.reshape(x.shape[0], x.shape[1], 1)


def create_cross_validation_sets(X_train, y_train, n_split=10):
    # Create a 10-fold cross-validation set using StratifiedKFold
    kf = KFold(n_splits=n_split, shuffle=False)

    # Lists to store training and validation sets
    X_train_cv_sets = list()
    X_val_cv_sets = list()
    y_train_cv_sets = list()
    y_val_cv_sets = list()

    # Split the data into training and validation sets for each fold
    for train_index, val_index in kf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        X_train_cv_sets.append(reshape_x(X_train_fold))
        X_val_cv_sets.append(reshape_x(X_val_fold))
        y_train_cv_sets.append(y_train_fold)
        y_val_cv_sets.append(y_val_fold)

    return X_train_cv_sets, X_val_cv_sets, y_train_cv_sets, y_val_cv_sets
