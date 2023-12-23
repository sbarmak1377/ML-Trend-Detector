from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from data_handler import *
from feature_extract import *
from models import svm, random_forest, mlp, logistic_regression, knn, guassian_nb, gradient_boosting, decision_tree
from preprocess import *

for time_frame in ["d1"]:
    for pair_curr in ["eurusd", "gbpusd", "audusd", "usdcad"]:

        print(time_frame + ": " + pair_curr)
        imbalance_solution = "down"
        data_path = 'data/dukascopy/forex/' + time_frame + '/' + pair_curr + '.csv'
        data_mode = "dukas_copy"
        ##############################################################################################

        df = read_csv_data(file_path=data_path, mode=data_mode)

        df = create_label_column(df)

        df = create_features(df)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        test = False
        if test:
            df = df.head(1000)

        if imbalance_solution == "up":
            sampler = RandomOverSampler()
        else:
            sampler = RandomUnderSampler()

        X_train_scaled, X_test_scaled, y_train, y_test, X_test_original = split_and_normalize(df)
        X_train_scaled, y_train = sampler.fit_resample(X_train_scaled, y_train)
        print(y_train.value_counts())
        print(y_test.value_counts())
        print(X_train_scaled.shape)
        print(y_test.shape)
        print(X_test_original.shape)
        print(y_test.shape)

        base_path = 'results/' + imbalance_solution + '/' + pair_curr + '/' + time_frame + '/'
        feature_select_path = base_path + 'feature_selection/'

        create_dir_if_not_exist(feature_select_path)

        print("feature selection...")
        selected_features = select_features(X_train_scaled, y_train,
                                            save_feature_bars_path=feature_select_path,
                                            threshold=0.9)
        X_train_scaled = X_train_scaled[selected_features].values
        X_test_scaled = X_test_scaled[selected_features].values
        print("feature selection complete!")

        print("Starting Random Forest...")
        random_forest.train_and_evaluate(X_train_scaled, y_train, X_test_scaled, y_test, X_test_original,
                                         base_path + 'random_forest/')
        print("Random Forest Test Done!")

        print("Starting SVM...")
        svm.train_and_evaluate(X_train_scaled, y_train, X_test_scaled, y_test, X_test_original, base_path + 'svm/')
        print("SVM Test Done!")

        print("Starting MLP Classifier...")
        mlp.train_and_evaluate(X_train_scaled, y_train, X_test_scaled, y_test, X_test_original, base_path + 'mlp/')
        print("MLP Classifier Test Done!")

        print("Starting Logistic Regression Classifier...")
        logistic_regression.train_and_evaluate(X_train_scaled, y_train, X_test_scaled, y_test, X_test_original,
                                               base_path + 'logistic_regression/')
        print("Logistic Regression Test Done!")

        print("Starting KNN Classifier...")
        knn.train_and_evaluate(X_train_scaled, y_train, X_test_scaled, y_test, X_test_original, base_path + 'knn/')
        print("KNN Test Done!")

        print("Starting Guassian NB Classifier...")
        guassian_nb.train_and_evaluate(X_train_scaled, y_train, X_test_scaled, y_test, X_test_original,
                                       base_path + 'guassian_nb/')
        print("Guassian NB Test Done!")

        print("Starting Gradient Boosting Classifier...")
        gradient_boosting.train_and_evaluate(X_train_scaled, y_train, X_test_scaled, y_test, X_test_original,
                                             base_path + 'gradient_boosting/')
        print("Gradient Boosting Test Done!")

        print("Starting Gradient Boosting Classifier...")
        decision_tree.train_and_evaluate(X_train_scaled, y_train, X_test_scaled, y_test, X_test_original,
                                         base_path + 'decision_tree/')
        print("Gradient Boosting Test Done!")
