import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

def categorization_results(results_filepath, meds):
    print('\nFilepath:', results_filepath)
    results_df = pd.read_csv(results_filepath).drop_duplicates()
    print('Columns:', results_df.columns)
    print('Distinct values in resp_concept_tf:', results_df['resp_concept_tf'].unique())

    y_true = results_df['ground_truth']
    y_pred = results_df['total_tf'] 

    results_df['meds_tf'] = np.where(results_df['meds_tf'] == True, 1, 0)
    results_df['resp_concept_tf'] = np.where(results_df['resp_concept_tf'] == True, 1, 0)

    ## Paper metrics: Total descriptions
    print('\nALL METRICS FOR PAPER:')
    print('Total number of selected descriptions:', results_df['total_tf'].sum())
    roc_auc = roc_auc_score(y_true, y_pred)
    print('ROC AUC:', roc_auc)
    results_df['total_latency'] = results_df['resp_concept_latency'] + results_df['meds_latency']
    print('Mean total latency:', results_df['total_latency'].mean())

    ## Paper metrics: Respiratory therapy concepts
    results_df = pd.merge(results_df, meds[['description', 'meds']], on='description', how='left')
    print('\nRESPIRATORY CONCEPTS FOR PAPER:')
    print('Number of respiratory therapy concepts:', results_df['is_intubated'].sum() + results_df['is_non_invasive'].sum() + results_df['is_hfni'].sum() - results_df['meds'].sum())
    print('Number of selected respiratory therapy concepts:', results_df['resp_concept_tf'].sum())
    results_df['ground_truth_resp'] = np.where((results_df['meds'] == 0) & (results_df['ground_truth'] == 1), 1, 0)
    y_true_resp = results_df['ground_truth_resp']
    y_pred_resp = results_df['resp_concept_tf']
    roc_auc_resp = roc_auc_score(y_true_resp, y_pred_resp)
    print('ROC AUC:', roc_auc_resp)
    print('Mean latency:', results_df['resp_concept_latency'].mean())

    ## Paper metrics: Medications
    print('\nMEDICATIONS FOR PAPER:')
    print('Number of medications:', results_df['meds'].sum())
    print('Number of selected medications:', results_df['meds_tf'].sum())
    y_true_meds = results_df['meds']
    y_pred_meds = results_df['meds_tf']
    roc_auc_meds = roc_auc_score(y_true_meds, y_pred_meds)
    print('ROC AUC:', roc_auc_meds)
    print('Mean latency:', results_df['meds_latency'].mean())

    return 0

results_filepath = ''
all_combined_filepath = ''

## add in meds concepts since these were not taken out when the concepts were categorized
all_data = pd.read_parquet(all_combined_filepath + 'combined_records_with_outcome.parquet')
meds_concepts = all_data.drop_duplicates(subset=['description', 'meds']).reset_index(drop=True)
del all_data

categorization_results(results_filepath + 'description_mapping_gemma227binstructq80.csv', meds=meds_concepts)
categorization_results(results_filepath + 'description_mapping_mistralsmall24binstruct2501q80.csv', meds=meds_concepts)
categorization_results(results_filepath + 'description_mapping_phi414bq80.csv', meds=meds_concepts)