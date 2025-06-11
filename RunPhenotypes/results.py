import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

def calculate_sensitivity_specificity(confusion_matrix, labels=None):
    sensitivity = []
    specificity = []
    for i in range(len(confusion_matrix)):
        tp = confusion_matrix[i, i]
        fn = confusion_matrix[i].sum() - tp
        fp = confusion_matrix[:, i].sum() - tp
        tn = confusion_matrix.sum() - (tp + fn + fp)
        
        sens = float(round(tp / (tp + fn) if (tp + fn) > 0 else 0, 2))
        spec = float(round(tn / (tn + fp) if (tn + fp) > 0 else 0, 2))
        sensitivity.append(sens)
        specificity.append(spec)
        if labels:
            print(f'Label {labels[i]}: Sensitivity = {sens}, Specificity = {spec}')
    return sensitivity, specificity

filepath = ''

## get the number of ground truth outcomes
patients_df = pd.read_csv('pts_with_outcomes.csv')
gt_outcome_counts = pd.DataFrame(patients_df['outcome'].value_counts().sort_index())
gt_outcome_counts['outcome_percentage'] = patients_df['outcome'].value_counts(normalize=True).sort_index()
gt_outcome_counts['outcome_percentage'] = gt_outcome_counts['outcome_percentage'].apply(lambda x: f'{x:.1%}')
print(f'Ground truth outcome counts:\n{gt_outcome_counts}')
del patients_df, gt_outcome_counts

models = ['mistralsmall24binstruct2501q80', 'gemma227binstructq80', 'phi414bq80']
for m in models:
    print(f'\n\nProcessing model: {m}')
    print(f'Filepath: {filepath}/{m}_cot.csv')
    if os.path.exists(f'{filepath}/final_phenotyped_pts_{m}.csv'):
        print(f'Loading results for model {m}...')
        results_df = pd.read_csv(f'{filepath}/final_phenotyped_pts_{m}.csv')
    else:
        print(f'File not found for model {m}. Skipping...')
        continue

    print('\nResults DataFrame:')
    print(f'Shape of results_df: {results_df.shape}')
    print('Columns in results_df:', results_df.columns)
    print('Head of results_df:', results_df.head())

    ## get average response latency
    distinct_records = results_df.drop_duplicates(subset=['records', 'latency'])
    avg_latency = distinct_records['latency'].mean() if 'latency' in distinct_records.columns else None
    print(f'Average response latency for distinct records: {avg_latency:.2f} seconds' if avg_latency is not None else 'No latency data available.')

    ## get the number of encounters per outcome for the model specifically
    print('\nOutcome counts in results_df:')
    outcome_counts = results_df['llm_outcome'].value_counts().sort_index()
    print(f'Outcome counts:\n{outcome_counts}')

    ## calculate accuracy, confusion matrix, AUC, sensitivity, and specificity
    print('\nCalculating accuracy, confusion matrix, AUC, sensitivity, and specificity...')
    accuracy = results_df['is_correct'].sum() / len(results_df)
    print(f'Accuracy: {accuracy:.2f}')

    y_true = results_df['gt_outcome']
    y_pred = results_df['llm_outcome']
    confusion = confusion_matrix(y_true, y_pred)
    confusion_df = pd.DataFrame(confusion, index = ['True -1', 'True 0', 'True 1', 'True 2', 'True 3', 'True 4', 'True 5', 'True 6'], columns = ['Pred -1', 'Pred 0', 'Pred 1', 'Pred 2', 'Pred 3', 'Pred 4', 'Pred 5', 'Pred 6'])
    print(f'Confusion Matrix:\n{confusion_df}')

    classes = sorted(set(y_true))
    y_true_binarized = label_binarize(y_true, classes=classes)
    class_specific_auc = {}
    for i, cls in enumerate(classes):
        y_true_binary = y_true_binarized[:, i]
        y_pred_binary = (y_pred == cls).astype(int)
        auc = roc_auc_score(y_true_binary, y_pred_binary)
        class_specific_auc[cls] = auc
    for cls, auc in class_specific_auc.items():
        print(f'Class {cls}: AUC = {auc:.2f}')

    sensitivity, specificity = calculate_sensitivity_specificity(confusion, labels=classes)

    ## calculate average of the aucs
    avg_auc = np.mean(list(class_specific_auc.values()))
    print(f'Average AUC: {avg_auc:.2f}')

    avg_for_classes_012 = np.mean([class_specific_auc[0], class_specific_auc[1], class_specific_auc[2], class_specific_auc[-1]])
    print(f'Average AUC for classes 0, 1, 2, and -1 (single treatment classes): {avg_for_classes_012:.2f}')

    avg_for_remaining_classes = np.mean([class_specific_auc[3], class_specific_auc[4], class_specific_auc[5], class_specific_auc[6]])
    print(f'Average AUC for remaining classes 3, 4, 5, and 6 (multiple treatment classes): {avg_for_remaining_classes:.2f}')
