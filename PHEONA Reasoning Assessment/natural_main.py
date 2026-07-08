import os
import pandas as pd
import numpy as np
import sys

from filepaths import *
from prompts import *
from utils import *


def get_explanation_correctness_results(input_filepath, output_filepath, data_model_name, temperature, top_p, port='11434', trials: int = 3):
    df = pd.read_parquet(input_filepath)
    print(f'\nLoaded input data with shape: {df.shape}')
    print('Head of the input dataframe:')
    print(df.head())

    # Check if output file exists for resuming
    df_existing = None
    if os.path.exists(output_filepath):
        df_existing = pd.read_excel(output_filepath)
        print(f'Loaded existing output file with {len(df_existing)} records.')
    else:
        print('No existing output file found. Starting fresh.')

    results_rows = []
    for index, row in df.iterrows():
        description, outcome, record = row['description'], row['outcome'], row['record']
        print(f'\nProcessing record {index+1}/{len(df)}')
        print(f'Description:\n{description}')
        print('Ground truth:', outcome)

        for trial in range(1, trials + 1):
            # Check if this record and trial already processed
            if df_existing is not None and ((df_existing['record'] == record) & (df_existing['trial'] == trial)).any():
                print(f'  Trial {trial}/{trials} already processed. Skipping.')
                continue

            print(f'  Trial {trial}/{trials}')

            nocot_formatted_prompt = instructions.format(description=description) + no_cot_questions
            somecot_formatted_prompt = instructions.format(description=description) + some_cot_questions
            fullcot_formatted_prompt = instructions.format(description=description) + full_cot_questions

            nocot_response, nocot_parsed_response, nocot_latency = get_response(nocot_formatted_prompt, description, data_model_name, temperature, top_p, port, '1')
            somecot_response, somecot_parsed_response, somecot_latency = get_response(somecot_formatted_prompt, description, data_model_name, temperature, top_p, port, '4')
            fullcot_response, fullcot_parsed_response, fullcot_latency = get_response(fullcot_formatted_prompt, description, data_model_name, temperature, top_p, port, '8')

            print('  Parsed responses:')
            print('  NoCoT:', nocot_parsed_response)
            print('  SomeCoT:', somecot_parsed_response)
            print('  FullCoT:', fullcot_parsed_response)

            result_dict = {
                'trial': trial,
                'record': record,
                'description': description,
                'outcome': outcome,
                'nocot_outcome': nocot_parsed_response,
                'nocot_latency': nocot_latency,
                'nocot_response': nocot_response,
                'nocot_explanationcorrectness_error': np.nan,
                'somecot_outcome': somecot_parsed_response,
                'somecot_latency': somecot_latency,
                'somecot_response': somecot_response,
                'somecot_explanationcorrectness_error': np.nan,
                'fullcot_outcome': fullcot_parsed_response,
                'fullcot_latency': fullcot_latency,
                'fullcot_response': fullcot_response,
                'fullcot_explanationcorrectness_error': np.nan,
                'formatted_nocot_prompt': nocot_formatted_prompt,
                'formatted_somecot_prompt': somecot_formatted_prompt,
                'formatted_fullcot_prompt': fullcot_formatted_prompt
            }
            results_rows.append(result_dict)
            print('  Response saved to final dataframe.')

            # Save immediately after processing
            results_df = pd.DataFrame([result_dict])
            if os.path.exists(output_filepath):
                existing_df = pd.read_excel(output_filepath)
                results_df = pd.concat([existing_df, results_df], ignore_index=True)
            results_df.to_excel(output_filepath, index=False)
            results_rows = []  # Reset since saved

def get_restoration_unfaithful_error_results(input_classified_records_filepath, output_classified_records_filepath, data_model_name, temperature, top_p, port='11434'):
    records = pd.read_parquet(input_classified_records_filepath)
    unique_descriptions = records['record'].unique()
    gts_to_description = records.drop_duplicates(['record'])[['record', 'outcome']].set_index('record')['outcome'].to_dict()
    descriptions_to_record = records.drop_duplicates(['record'])[['record', 'description']].set_index('record')['description'].to_dict()
    print(f'\nLoaded {len(unique_descriptions)} unique descriptions.')
    del records

    # Check if output file exists for resuming
    df_existing = None
    if os.path.exists(output_classified_records_filepath):
        df_existing = pd.read_excel(output_classified_records_filepath)
        print(f'Loaded existing output file with {len(df_existing)} records.')
        print('Existing trials:', df_existing['trial'].value_counts().sort_index())
    else:
        print('No existing output file found. Starting fresh.')

    trials = 3
    n_to_save = 80

    for trial in range(1, trials + 1):
        print(f'\nStarting trial {trial}/{trials}')

        # Determine already saved descriptions for this trial
        already_saved = set()
        if df_existing is not None and trial in df_existing['trial'].values:
            already_saved = set(df_existing[df_existing['trial'] == trial]['record'])
            print(f'Trial {trial} already has {len(already_saved)} saved records.')

        # Set seed for this trial
        seed = 42 + trial - 1
        np.random.seed(seed)
        np.random.shuffle(unique_descriptions)
        print(f'Shuffled descriptions with seed {seed}.')

        unique_descriptions_df = pd.DataFrame(unique_descriptions, columns=['records'])
        unique_descriptions_df['outcome'] = unique_descriptions_df['records'].map(gts_to_description)
        print(f'Number of records for trial {trial}:', unique_descriptions_df.shape[0])

        desc_df_rows = []
        n_saved = 0

        for index, row in unique_descriptions_df.iterrows():
            record_id, ground_truth = row['records'], row['outcome']
            description = descriptions_to_record[record_id]
            
            if record_id in already_saved:
                continue

            print(f'\nProcessing record {index+1}/{len(unique_descriptions_df)} for trial {trial}')
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
                    'trial': trial,
                    'record': record_id,
                    'description': description,
                    'outcome': ground_truth,
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
                print(f'Response saved to final dataframe for trial {trial}. Total saved this trial: {n_saved}')

                # Save immediately after processing
                desc_df = pd.DataFrame([desc_dict])
                if os.path.exists(output_classified_records_filepath):
                    existing_df = pd.read_excel(output_classified_records_filepath)
                    desc_df = pd.concat([existing_df, desc_df], ignore_index=True)
                desc_df.to_excel(output_classified_records_filepath, index=False)
                desc_df_rows = []  # Reset since saved

                if n_saved >= n_to_save:
                    print(f'\nSaved {n_saved} records for trial {trial}. Stopping early.')
                    break

    print('\nCompleted processing all restoration/unfaithful records for all trials.')

    # Ensure file exists even if no records saved
    if not os.path.exists(output_classified_records_filepath):
        pd.DataFrame().to_excel(output_classified_records_filepath, index=False)
        print(f'Created empty file {output_classified_records_filepath}')

if __name__ == "__main__":
    ## setup runtime variables
    model_name = str(sys.argv[1])
    port = str(sys.argv[2])
    print('Model name:', model_name)
    print('Port:', port)

    os.environ['OLLAMA_MODELS'] = model_filepath

    if model_name == 'mistral':
        ollama_model_name = 'mistral-small:24b-instruct-2501-q4_K_M'
    elif model_name == 'phi':
        ollama_model_name = 'phi4:14b-q4_K_M'
    elif model_name == 'deepseek':
        ollama_model_name = 'deepseek-r1:32b-qwen-distill-q4_K_M'
    elif model_name == "phireason":
        ollama_model_name = 'phi4-reasoning:14b-plus-q4_K_M'
    elif model_name == "magistral":
        ollama_model_name = 'magistral:24b-small-2506-q4_K_M'
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    explanation_correctness_inputdata_filepath = input_filepath + 'explanation_correctness_inputdata.parquet'
    restoration_unfaithful_inputdata_filepath = input_filepath + 'restoration_unfaithful_inputdata.parquet'
    
    output_explanation_correctness_filepath = output_filepath + 'explanation_correctness_' + make_filename_safe(ollama_model_name) + '.xlsx'
    output_restoration_unfaithful_filepath = output_filepath + 'restoration_unfaithful_' + make_filename_safe(ollama_model_name) + '.xlsx'

    get_explanation_correctness_results(explanation_correctness_inputdata_filepath, output_explanation_correctness_filepath, ollama_model_name, 0.5, 0.99, port=port)
    get_restoration_unfaithful_error_results(restoration_unfaithful_inputdata_filepath, output_restoration_unfaithful_filepath, ollama_model_name, 0.5, 0.99, port=port)
