import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score, roc_curve, auc, \
    ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize


def calculate_and_save_metrics(y_pred, y_test, y_pred_proba, save_dir):
    accuracy = accuracy_score(y_test, y_pred)
    with open(save_dir + "Accuracy.txt", "w") as text_file:
        text_file.write(str(accuracy))
    r2 = r2_score(y_test, y_pred)
    with open(save_dir + "r2.txt", "w") as text_file:
        text_file.write(str(r2))
    report = classification_report(y_test, y_pred)
    with open(save_dir + "classification_report.txt", "w") as text_file:
        text_file.write(str(report))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot().figure_.savefig(save_dir + 'confusion_matrix.png')

    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    # y_pred_bin = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test_bin.shape[1]

    for i in range(n_classes):
        # Compute ROC curve and ROC area for each class
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-AUC Curve - Class {}'.format(i))
        plt.legend(loc="lower right")

        # Save the plot as a PNG file
        plt.savefig(save_dir + 'roc_auc_class_{}.png'.format(i))
        plt.clf()
        plt.close('all')
    return accuracy


def calculate_and_save_return(x_test_original, y_prediction, save_dir, init_curr_1=1000, init_curr_2=0, trade_rate=1.0):
    full_data = x_test_original[['Close']].copy()
    full_data['Prediction'] = y_prediction
    full_data.to_csv(save_dir + 'exchange_data.csv', index=False, sep='\t')
    curr_1, curr_2, rate = init_curr_1, init_curr_2, trade_rate
    index = 2
    current_action = 0
    while index < len(full_data):
        pred_0 = int(full_data.iloc[index - 2]['Prediction'])
        pred_1 = int(full_data.iloc[index - 1]['Prediction'])
        pred_2 = int(full_data.iloc[index]['Prediction'])

        if pred_2 == pred_1 and pred_1 == pred_0:
            if pred_2 == 1 and current_action != 1:
                current_action = 1
                if curr_2 > 0:
                    exchange = curr_2 * rate
                    curr_2 -= exchange
                    curr_1 += exchange * 1.0 / full_data.iloc[index]['Close']

            elif pred_2 == 2 and current_action != 2:
                current_action = -1
                if curr_1 > 0:
                    exchange = curr_1 * rate
                    curr_1 -= exchange
                    curr_2 += exchange * full_data.iloc[index]['Close']

        index += 1
    curr_1 = curr_1 + (curr_2 * 1.0 / full_data.iloc[-1]['Close'])
    actual_return = (curr_1 - init_curr_1)
    return_percent = actual_return * 100 / init_curr_1
    with open(save_dir + "returns.txt", "w") as text_file:
        text_file.write("Init Currency 1: " + str(init_curr_1) + "\n")
        text_file.write("Init Currency 2: " + str(init_curr_2) + "\n")
        text_file.write("Actual Return: " + str(actual_return) + "\n")
        text_file.write("Percent Return: " + str(return_percent) + "%\n")
    return return_percent
