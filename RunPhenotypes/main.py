import os
import re
import ollama
import pandas as pd
from datetime import datetime
import string
import numpy as np
import sys
import tiktoken

from filepaths import *
from prompts import *

def make_filename_safe(input_string):
    safe_string = ''.join(char for char in input_string if char not in string.punctuation and not char.isspace())
    return safe_string

def parse_response(response_content):
    replace_dict = {
        'IMV ONLY': 0,
        'NIPPV ONLY': 1,
        'HFNI ONLY': 2,
        'NIPPV TO IMV': 3,
        'HFNI TO IMV': 4,
        'IMV TO NIPPV': 5,
        'IMV TO HFNI': 6
    }

    match = re.search(r'A8\)(.*)', response_content, re.IGNORECASE | re.DOTALL) # update based on question in the prompt
    if match:
        response = match.group(1).strip()
        for k, v in replace_dict.items():
            if k in response:
                return v
    return -1

def get_context_length(text, model_name="gpt-3.5-turbo"):
    tokenizer = tiktoken.encoding_for_model(model_name)
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_response(prompt, record_str, data_model_name, temperature, top_p, port):
    formatted_prompt = prompt.format(description = record_str)
    num_ctx = 2048 if get_context_length(formatted_prompt) < 2012 else 7500

    start_time = datetime.now()
    
    client = ollama.Client(
        host=f'http://localhost:{port}'   
    )
    response = client.chat(
        model=data_model_name,
        messages=[
            {
                "role": "user",
                "content": formatted_prompt
            }
        ],
        options={
            "temperature": temperature,
            "top_p": top_p,
            "num_ctx": num_ctx
        }
    )
    end_time = datetime.now()
    latency = (end_time - start_time).total_seconds()
    response_content = response["message"]["content"].strip().upper()
    parsed_response = parse_response(response_content)
    
    return response_content, parsed_response, latency

def get_final_df(input_patients_outcomes_filepath, input_classified_records_filepath):
    pt_records = pd.read_parquet(input_patients_outcomes_filepath, columns=['patientunitstayid', 'outcome']).rename(columns={'outcome': 'gt_outcome'})
    pt_records['patientunitstayid'] = pt_records['patientunitstayid'].astype('int64')
    desc_records = pd.read_parquet(input_classified_records_filepath, columns=['patientunitstayid', 'ordered_string']).rename(columns={'ordered_string': 'records'})
    desc_records['patientunitstayid'] = desc_records['patientunitstayid'].astype('int64')
    
    all_records = pd.merge(pt_records, desc_records, on='patientunitstayid', how='left')
    all_records['gt_outcome'] = all_records['gt_outcome'].fillna(-1).astype('int64')
    all_records['records'] = all_records['records'].fillna('NO RECORDS')
    print(f'\nLoaded {len(all_records)} records.')
    print(f'Shape of the records: {all_records.shape}')
    print(f'Columns in the records: {all_records.columns.tolist()}')
    print(f'Unique patientunitstayid: {all_records["patientunitstayid"].nunique()}')
    print(f'Unique records: {all_records["records"].nunique()}')
    print(f'Unique gt_outcome: {all_records["gt_outcome"].nunique()}')
    print(f'Value counts of gt_outcome: {all_records["gt_outcome"].value_counts()}')
    print(f'Number of records that are NO RECORDS: {all_records[all_records["records"] == "NO RECORDS"].shape[0]}')
    print(f'Number of missing values across all columns: {all_records.isnull().sum().sum()}')
    print(f'Head of the records: {all_records.head()}')

    return all_records

def get_descriptions(prompt, input_classified_records_filepath, output_classified_records_filepath, data_model_name, temperature, top_p, port='11434'):
    records = pd.read_parquet(input_classified_records_filepath, columns=['ordered_string'])
    unique_descriptions = records['ordered_string'].unique()
    print(f'\nLoaded {len(unique_descriptions)} unique descriptions.')
    del records

    if os.path.exists(output_classified_records_filepath):
        already_processed = pd.read_csv(output_classified_records_filepath)
        already_processed_descriptions = already_processed["records"].unique()
        unique_descriptions = [description for description in unique_descriptions if description not in already_processed_descriptions]
        print(f"\nAlready processed {len(already_processed)} descriptions. {len(unique_descriptions)} remaining.")

    m = 5

    desc_df_rows = []
    for i, d in enumerate(unique_descriptions):
        formatted_prompt = prompt.format(description=d)
        total_response, parsed_response, latency = get_response(prompt, d, data_model_name, temperature, top_p, port)
        desc_dict = {'llm_outcome': parsed_response,
                     'latency': latency,
                     'records': d,
                     'full_response': total_response,
                     'formatted_prompt': formatted_prompt}
        desc_df_rows.append(desc_dict)
        print(f'\nProcessed {i + 1} out of {len(unique_descriptions)} descriptions')
        print(f'Parsed response: {parsed_response} for description: {d}')

        # Save every m records
        if (i + 1) % m == 0:
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
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    prompt_to_test = cot_prompt
    prompt_name = 'cot'
    
    ## column names for final df: ['patientunitstayid', 'gt_outcome', 'llm_outcome', 'latency', 'records', 'full_response', 'formatted_prompt']
    input_patients_outcomes_filepath = input_filepath + 'pts_with_outcomes.parquet'
    input_classified_records_filepath = input_filepath + 'phenotyping_processed_' + make_filename_safe(ollama_model_name) + '_ordered.parquet'
    output_patients_outcomes_filepath = output_filepath + 'phenotyped_pts_' + make_filename_safe(ollama_model_name) + '.csv'
    output_classified_records_filepath = output_filepath + 'classified_descs_' + make_filename_safe(ollama_model_name) + '_' + prompt_name + '.csv'
    output_final_filepath = output_filepath + 'final_phenotyped_pts_' + make_filename_safe(ollama_model_name) + '.csv'
    
    ## get the patients and their outcomes and records
    if os.path.exists(output_patients_outcomes_filepath):
        patients_df = pd.read_csv(output_patients_outcomes_filepath)
    else:
        patients_df = get_final_df(input_patients_outcomes_filepath, input_classified_records_filepath)
        patients_df.to_csv(output_patients_outcomes_filepath, index=False)

    ## classify the unique descriptions
    get_descriptions(prompt_to_test, input_classified_records_filepath, output_classified_records_filepath, ollama_model_name, 0, 0.99, port=port)

    ## merge the descriptions with the patients
    desc_df = pd.read_csv(output_classified_records_filepath)
    patients_df = pd.merge(patients_df, desc_df, on='records', how='inner')
    patients_df['llm_outcome'] = patients_df['llm_outcome'].fillna(-1).astype('int64')
    patients_df['latency'] = patients_df['latency'].fillna(np.nan).astype('float64')
    patients_df['full_response'] = patients_df['full_response'].fillna('NO RECORDS')
    patients_df['formatted_prompt'] = patients_df['formatted_prompt'].fillna('NO RECORDS')

    patients_df['is_correct'] = np.where(patients_df['llm_outcome'] == patients_df['gt_outcome'], 1, 0).astype('int64')
    patients_df.to_csv(output_final_filepath, index=False)
