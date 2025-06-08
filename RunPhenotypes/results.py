import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

def calculate_sensitivity_specificity(confusion_matrix):
    sensitivity = []
    specificity = []
    for i in range(len(confusion_matrix)):
        tp = confusion_matrix[i, i]
        fn = confusion_matrix[i].sum() - tp
        fp = confusion_matrix[:, i].sum() - tp
        tn = confusion_matrix.sum() - (tp + fn + fp)
        
        sensitivity.append(float(round(tp / (tp + fn) if (tp + fn) > 0 else 0, 2)))
        specificity.append(float(round(tn / (tn + fp) if (tn + fp) > 0 else 0, 2)))
    
    return sensitivity, specificity

filepath = ''

models = ['mistralsmall24binstruct2501q80', 'gemma227binstructq80', 'phi414bq80']
for m in models:
    print(f'\n\nProcessing model: {m}')
    if os.path.exists(f'{filepath}/classified_descs_{m}_cot.csv'):
        results_df = pd.read_csv(f'{filepath}/classified_descs_{m}_cot.csv')
        patients_df = pd.read_csv(f'{filepath}/phenotyped_pts_{m}.csv')
    else:
        print(f'File not found for model {m}. Skipping...')
        continue

    print('\nPatients DataFrame:')
    print(f'Shape of patients_df: {patients_df.shape}')
    print('Columns in patients_df:', patients_df.columns)
    print('Head of patients_df:', patients_df.head())

    print('\nResults DataFrame:')
    print(f'Shape of results_df: {results_df.shape}')
    print('Columns in results_df:', results_df.columns)
    print('Head of results_df:', results_df.head())

    ## get average response latency
    if 'latency' in results_df.columns:
        avg_latency = results_df['latency'].mean()
        print(f'Average response latency: {avg_latency:.2f} seconds')

    results_df = pd.merge(patients_df, results_df, on='records', how='left')
    results_df = results_df.dropna(subset=['llm_outcome', 'gt_outcome'])
    print('\nMerged DataFrame:')
    print('Number of unique patients:', results_df['patientunitstayid'].nunique())
    print('Number of unique records:', results_df['records'].nunique())
    print('Shape of the merged DataFrame:', results_df.shape)
    print('Head of the merged DataFrame:', results_df.head())
    results_df.to_csv(f'{filepath}/merged_results_df_{m}.csv', index=False)

    results_df['is_correct'] = np.where(results_df['llm_outcome'] == results_df['gt_outcome'], 1, 0).astype('int64')

    ## get the number of encounters per outcome
    print('\nOutcome counts in results_df:')
    outcome_counts = results_df['gt_outcome'].value_counts().sort_index()
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

    sensitivity, specificity = calculate_sensitivity_specificity(confusion)
    print(f'Sensitivity: {sensitivity}')
    print(f'Specificity: {specificity}')
