from itertools import product

import pandas as pd
from sklearn.linear_model import LogisticRegression

from data_handler import create_dir_if_not_exist
from evaluation_metrics import calculate_and_save_metrics, calculate_and_save_return

params_dict = {'C': [0.5, 1, 10, 25, 50, 100], 'penalty': ['l2'],
               'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'], 'max_iter': [1000]}


def train_and_evaluate(x_train, y_train, x_test, y_test, x_test_original, save_dir):
    if save_dir[-1] != '/':
        save_dir += '/'
    best_acc = 0.0
    best_return_percent = -10
    best_params_return = None
    best_params_acc = None
    for params in product(*params_dict.values()):
        test_parameters = dict(zip(params_dict.keys(), params))
        print(f"Parameters: {test_parameters}")
        save_dir_param = save_dir + str(test_parameters).replace(':', '_').replace('\'', '') + '/'
        create_dir_if_not_exist(save_dir_param)
        classifier = LogisticRegression()

        # Update classifier with current parameters
        classifier.set_params(**test_parameters)

        # Step 5: Train the Model
        print("Fitting Model...")
        classifier.fit(x_train, y_train)
        print("Model Fit Complete!")

        # Step 6: Make Predictions
        y_pred = classifier.predict(x_test)
        y_pred_proba = classifier.predict_proba(x_test)

        # Step 7: Evaluate Performance
        print("Save Results...")
        acc = calculate_and_save_metrics(y_pred, y_test, y_pred_proba, save_dir_param)
        return_percent = calculate_and_save_return(x_test_original, y_pred, save_dir_param)
        if acc > best_acc:
            best_acc = acc
            best_params_acc = test_parameters
        if return_percent > best_return_percent:
            best_return_percent = return_percent
            best_params_return = test_parameters
        print("Save Results Complete!")

        print("Save Prediction Data...")
        test_pred_data = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
        test_pred_data.to_csv(save_dir_param + 'test_pred_data.csv', sep='\t', index=False)
        print("Save Prediction Data Complete!")

        print("Done.")
        print("*" * 100)
    with open(save_dir + "best_return.txt", "w") as file:
        for k, v in best_params_return.items():
            dictionary_content = k + ": " + v + "\n"
            file.write(dictionary_content)

    with open(save_dir + "best_acc.txt", "w") as file:
        for k, v in best_params_acc.items():
            dictionary_content = k + ": " + v + "\n"
            file.write(dictionary_content)
