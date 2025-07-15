import os
import pandas as pd
import numpy as np
import sys

from filepaths import *
from prompts import *
from utils import *

def get_descriptions(input_classified_records_filepath, output_classified_records_filepath, gts_by_record, data_model_name, temperature, top_p, port='11434'):
    records = pd.read_parquet(input_classified_records_filepath, columns=['ordered_string'])
    unique_descriptions = records['ordered_string'].unique()
    print(f'\nLoaded {len(unique_descriptions)} unique descriptions.')
    del records

    seed = 42
    np.random.seed(seed)
    np.random.shuffle(unique_descriptions)
    print(f'\nShuffled descriptions with seed {seed}.')

    unique_descriptions = pd.DataFrame(unique_descriptions, columns = ['records'])
    unique_descriptions = pd.merge(unique_descriptions, gts_by_record, how='left', on = 'records')
    print(f'Number of records:', unique_descriptions.shape[0])
    print('Head of the new records dataframe:')
    print(unique_descriptions.head())

    m = 5

    desc_df_rows = []
    n_saved = 0
    n_to_save = 100
    for index, row in unique_descriptions.iterrows():
        print('\nNumber of saved records:', n_saved)
        if n_saved >= n_to_save:
            print(f'\nSaved {n_saved} records. Stopping early.')
            break

        description, ground_truth = row['records'], row['gt_outcome']
        print(f'Description:\n{description}')
        print('Ground truth:', ground_truth)

        nocot_formatted_prompt = instructions.format(description=description) + no_cot_questions
        somecot_formatted_prompt = instructions.format(description=description) + some_cot_questions
        fullcot_formatted_prompt = instructions.format(description=description) + full_cot_questions

        nocot_response, nocot_parsed_response, nocot_latency = get_response(nocot_formatted_prompt, description, data_model_name, temperature, top_p, port, '1')
        somecot_response, somecot_parsed_response, somecot_latency = get_response(somecot_formatted_prompt, description, data_model_name, temperature, top_p, port, '4')
        fullcot_response, fullcot_parsed_response, fullcot_latency = get_response(fullcot_formatted_prompt, description, data_model_name, temperature, top_p, port, '8')

        print('Parsed responses:')
        print('NoCoT:', nocot_parsed_response)
        print('SomeCoT:', somecot_parsed_response)
        print('FullCoT:', fullcot_parsed_response)

        if (nocot_parsed_response == ground_truth and somecot_parsed_response == ground_truth and fullcot_parsed_response == ground_truth):
            desc_dict = {
                'description': description,
                'gt_outcome': ground_truth,
                'nocot_outcome': nocot_parsed_response,
                'nocot_latency': nocot_latency,
                'nocot_response': nocot_response,
                'nocot_restoration_error': np.nan,
                'nocot_unfaithfulshortcut_error': np.nan,
                'somecot_outcome': somecot_parsed_response,
                'somecot_latency': somecot_latency,
                'somecot_response': somecot_response,
                'somecot_restoration_error': np.nan,
                'somecot_unfaithfulshortcut_error': np.nan,
                'fullcot_outcome': fullcot_parsed_response,
                'fullcot_latency': fullcot_latency,
                'fullcot_response': fullcot_response,
                'fullcot_restoration_error': np.nan,
                'fullcot_unfaithfulshortcut_error': np.nan,
                'formatted_nocot_prompt': nocot_formatted_prompt,
                'formatted_somecot_prompt': somecot_formatted_prompt,
                'formatted_fullcot_prompt': fullcot_formatted_prompt
            }
            desc_df_rows.append(desc_dict)
            n_saved += 1
            print('Response saved to final dataframe.')

        # Save every m records that were correct
        if ((len(desc_df_rows) % m == 0) & (index != 0)):
            desc_df = pd.DataFrame(desc_df_rows)
            desc_df.to_csv(output_classified_records_filepath, mode='a', header=not os.path.exists(output_classified_records_filepath), index=False)
            desc_df_rows = []
    
    # Save any remaining records
    if desc_df_rows:
        desc_df = pd.DataFrame(desc_df_rows)
        desc_df.to_csv(output_classified_records_filepath, mode='a', header=not os.path.exists(output_classified_records_filepath), index=False)

    print('\nCompleted processing all descriptions.')

if __name__ == "__main__":
    ## setup runtime variables
    model_name = str(sys.argv[1])
    port = str(sys.argv[2])
    print('Model name:', model_name)
    print('Port:', port)

    os.environ['OLLAMA_MODELS'] = model_filepath

    if model_name == 'mistral':
        ollama_model_name = 'mistral-small:24b-instruct-2501-q8_0'
    elif model_name == 'gemma':
        ollama_model_name = 'gemma2:27b-instruct-q8_0'
    elif model_name == 'phi':
        ollama_model_name = 'phi4:14b-q8_0'
    elif model_name == 'deepseek':
        ollama_model_name = 'deepseek-r1:32b'
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
    ## column names for final df: ['patientunitstayid', 'gt_outcome', 'llm_outcome', 'latency', 'records', 'full_response', 'formatted_prompt']
    input_patients_outcomes_filepath = input_filepath + 'pts_with_outcomes.parquet'
    input_classified_records_filepath = input_filepath + 'phenotyping_processed_' + make_filename_safe(ollama_model_name) + '_ordered.parquet' if model_name != 'deepseek' else input_filepath + 'phenotyping_processed_' + make_filename_safe('mistral-small:24b-instruct-2501-q8_0') + '_ordered.parquet'
    output_patients_outcomes_filepath = output_filepath + 'phenotyped_pts_' + make_filename_safe(ollama_model_name) + '.csv'
    output_classified_records_filepath = output_filepath + 'natural_results_' + make_filename_safe(ollama_model_name) + '.csv'
    
    ## get the patients and their outcomes and records
    if os.path.exists(output_patients_outcomes_filepath):
        patients_df = pd.read_csv(output_patients_outcomes_filepath)
    else:
        patients_df = get_final_df(input_patients_outcomes_filepath, input_classified_records_filepath)
        patients_df.to_csv(output_patients_outcomes_filepath, index=False)

    ## classify the unique descriptions
    gts_by_record = patients_df.groupby('records')['gt_outcome'].max().reset_index()
    get_descriptions(input_classified_records_filepath, output_classified_records_filepath, gts_by_record, ollama_model_name, 0.5, 0.99, port=port)
