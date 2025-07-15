import re
import ollama
import pandas as pd
from datetime import datetime
import string
import tiktoken

replace_dict = {
        'IMV ONLY': 0,
        'NIPPV ONLY': 1,
        'HFNI ONLY': 2,
        'NIPPV TO IMV': 3,
        'HFNI TO IMV': 4,
        'IMV TO NIPPV': 5,
        'IMV TO HFNI': 6
    }

reverse_replace_dict = {v: k for k, v in replace_dict.items()}

def make_filename_safe(input_string):
    safe_string = ''.join(char for char in input_string if char not in string.punctuation and not char.isspace())
    return safe_string

def parse_response(response_content, question):
    re_string = f'A{question}\)(.*)'
    match = re.search(re_string, response_content, re.IGNORECASE | re.DOTALL)
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

def get_response(prompt, record_str, data_model_name, temperature, top_p, port, question):
    formatted_prompt = prompt.format(description = record_str)
    num_ctx = 2048 if get_context_length(formatted_prompt) < 2000 else 7950

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
    parsed_response = parse_response(response_content, question)
    
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
