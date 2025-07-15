import pandas as pd
import os
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
    if os.path.exists(filepath + f'classified_descs_{m}_cot_sample_2_QA.xlsx'):
        results_df = pd.read_excel(filepath + f'classified_descs_{m}_cot_sample_2_QA.xlsx')
    else:
        print(f'File classified_descs_{m}_cot_sample_2_QA.xlsx not found. Skipping...')
        continue

    print(f'\nShape of results_df: {results_df.shape}')
    print('Columns in results_df:', results_df.columns)
    print('Head of results_df:', results_df.head())

    y_true = results_df['upd_ground_truth'].values
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