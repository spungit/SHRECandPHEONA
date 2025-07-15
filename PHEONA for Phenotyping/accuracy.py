import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

def calculate_sensitivity_specificity(y_true, y_pred):
    outcomes = sorted(set(y_true))
    metrics = {}

    for outcome in outcomes:
        # True Positives (TP): Correctly predicted as the current outcome
        tp = sum((y_pred == outcome) & (y_true == outcome))
        # False Negatives (FN): Actual outcome but predicted as something else
        fn = sum((y_pred != outcome) & (y_true == outcome))
        # False Positives (FP): Predicted as the outcome but actually something else
        fp = sum((y_pred == outcome) & (y_true != outcome))
        # True Negatives (TN): Neither predicted nor actual as the outcome
        tn = sum((y_pred != outcome) & (y_true != outcome))

        sensitivity = round(tp / (tp + fn), 2) if (tp + fn) > 0 else 0
        specificity = round(tn / (tn + fp), 2) if (tn + fp) > 0 else 0

        metrics[outcome] = [sensitivity, specificity]

    return metrics


filepath = '/Users/spungit/Documents/Graduate School/Dissertation Work/Dissertation Work/LLM Phenotyping/Results/PhenotypingResults/EvaluationFramework/'

models = ['mistralsmall24binstruct2501q80', 'gemma227binstructq80', 'phi414bq80']

for m in models:
    print(f'\nProcessing model: {m}')
    if os.path.exists(f'{filepath}/final_phenotyped_pts_{m}_sample_2.csv'):
        results_df = pd.read_csv(f'{filepath}/final_phenotyped_pts_{m}_sample_2.csv')
    else:
        print(f'Filepath: {filepath}')
        print(f'classified_descs_{m}_cot_sample_2.csv or phenotyped_pts_{m}.csv not found. Skipping...')
        continue

    print(f'\nShape of results_df: {results_df.shape}')
    print('Columns in results_df:', results_df.columns)
    print('Head of results_df:', results_df.head())

    average_latency = results_df['latency'].mean()
    print(f'Average latency: {average_latency:.2f} seconds')

    accuracy = results_df['is_correct'].sum() / len(results_df)
    print(f'Accuracy: {accuracy:.2f}')

    y_true = results_df['gt_outcome'].values
    y_pred = results_df['llm_outcome'].values

    metrics = calculate_sensitivity_specificity(y_true, y_pred)
    for met in metrics:
        print(f'Outcome {met}: Sensitivity: {metrics[met][0]}, Specificity: {metrics[met][1]}')

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

    confusion = confusion_matrix(y_true, y_pred)
    print(f'Confusion Matrix:\n{confusion}')

    ## group by the record and get the max of the outcome
    grouped_df = results_df.groupby('records').agg({'gt_outcome': 'max'})
    print(f'Shape of the grouped DataFrame: {grouped_df.shape}')
    print(f'Grouped DataFrame:\n{grouped_df.head()}')
    grouped_df_filtered = results_df.drop_duplicates(subset=['records'])[['records','llm_outcome']].merge(grouped_df, on='records', how='left')
    print(f'Shape of the filtered DataFrame: {grouped_df_filtered.shape}')
    print(f'Filtered DataFrame:\n{grouped_df_filtered.head()}')
    grouped_df_filtered['is_correct'] = np.where(grouped_df_filtered['llm_outcome'] == grouped_df_filtered['gt_outcome'], 1, 0).astype('int64')
    grouped_df_filtered.to_csv(f'{filepath}/grouped_results_df_{m}.csv', index=False)