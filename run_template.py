import gc
import pickle
import warnings


from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data_handler import *
from feature_extract import *
from preprocess import *

classifiers = {
    # 'RandomForest': {'model': RandomForestClassifier(),
    #                  'params': {'n_estimators': [10, 50, 100]}},
    # 'GradientBoosting': {'model': GradientBoostingClassifier(),
    #                      'params': {'n_estimators': [50, 100]}},
    'SVM': {'model': SVC(),
            'params': {'C': [10, 25, 50, 100], 'kernel': ['linear', 'rbf']}},
    'KNeighbors': {'model': KNeighborsClassifier(),
                   'params': {'n_neighbors': [5, 7, 10, 15, 20, 30]}},
    'LogisticRegression': {'model': LogisticRegression(),
                           'params': {'C': [10, 25, 50, 100]}},
    'DecisionTree': {'model': DecisionTreeClassifier(),
                     'params': {'max_depth': [None, 10, 20, 30, 50, 75, 100]}},
    'GaussianNB': {'model': GaussianNB(), 'params': {}},
    'MLPClassifier': {'model': MLPClassifier(),
                      'params': {
                          'hidden_layer_sizes': [(32,), (64,), (128,), (256,), (64, 64), (128, 128)],
                          'activation': ['identity', 'logistic', 'tanh', 'relu'],
                          'solver': ['sgd', 'adam'],
                          'batch_size': [128, 256],
                          'learning_rate': ['adaptive'],
                          'max_iter': [1000],
                          'shuffle': [False]}},
}


warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

# for pair_curr in ["audusd", "eurusd", "gbpusd", "usdcad"]:
for pair_curr in ["audusd"]:
    for time_frame in ["h1"]:
        print(pair_curr + " in time frame " + time_frame)
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

        X_train_scaled, X_test_scaled, y_train, y_test = split_and_normalize(df)
        X_train_scaled, y_train = sampler.fit_resample(X_train_scaled, y_train)
        print(y_train.value_counts())
        print(y_test.value_counts())
        print(X_train_scaled.shape)
        print(y_test.shape)
        print(X_test_scaled.shape)
        print(y_test.shape)

        base_path = 'classic_results/' + imbalance_solution + '/' + pair_curr + '/' + time_frame + '/'
        feature_select_path = base_path + 'feature_selection/'

        create_dir_if_not_exist(feature_select_path)

        selected_features, f_et, f_rf, f_avg = select_features(X_train_scaled, y_train,
                                                               save_feature_bars_path=feature_select_path,
                                                               threshold=0.9)
        X_train_scaled = X_train_scaled[selected_features].values
        X_test_scaled = X_test_scaled[selected_features].values

        for clf_name, clf_info in classifiers.items():
            print("Running model " + clf_name + ".")
            param_grid = clf_info['params']
            grid_search = GridSearchCV(clf_info['model'], param_grid, cv=5, scoring='accuracy', verbose=10)
            grid_search.fit(X_train_scaled, y_train)

            # Get the best parameters
            best_params = grid_search.best_params_

            # Fit the model with the best parameters
            best_clf = grid_search.best_estimator_
            best_clf.fit(X_train_scaled, y_train)

            # Make predictions on the test set
            y_pred = best_clf.predict(X_test_scaled)

            # Get classification report
            clf_report = classification_report(y_test, y_pred)

            # Print and save the classification report to a text file
            print(f"Classification Report for {clf_name}:\n{clf_report}")

            model_filename = base_path + f"{clf_name}_model_{str(best_params).replace(':', '_').replace(' ', '')}.pkl"
            with open(model_filename, 'wb') as f:
                pickle.dump(best_clf, f)
            # Save the classification report to a text file

            report_filename = base_path + f"{clf_name}_report_{str(best_params).replace(':', '_').replace(' ', '')}.txt"
            with open(report_filename, 'w') as file:
                file.write(f"Classification Report for {clf_name}:\n{clf_report}")

            # Get confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(cm, index=range(cm.shape[0]), columns=range(cm.shape[1]))

            # Save confusion matrix to a CSV file
            cm_filename = base_path + f"{clf_name}_confusion_matrix_{str(best_params).replace(':', '_').replace(' ', '')}.csv"
            cm_filename_img = base_path + f"{clf_name}_confusion_matrix_{str(best_params).replace(':', '_').replace(' ', '')}.png"
            cm_df.to_csv(cm_filename)

            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            plt.figure(figsize=(8, 6))
            disp.plot(cmap='Blues', values_format='g')

            # Save the confusion matrix as a PNG file
            plt.savefig(cm_filename_img)
            plt.clf()

            result_df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})

            # Save the dataframe to a CSV file
            result_filename = base_path + f"{clf_name}_results_{str(best_params).replace(':', '_').replace(' ', '')}.csv"
            result_df.to_csv(result_filename, index=False, sep='\t')

            print(f"Classification report saved as {report_filename}\n")

            del grid_search
            print("model " + clf_name + "Tuned Successfully.")
