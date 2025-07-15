import pandas as pd
import string
import random

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

input_filepath = ''
ollama_model_names = ['mistral-small:24b-instruct-2501-q8_0', 'gemma2:27b-instruct-q8_0', 'phi4:14b-q8_0']

for ollama_model_name in ollama_model_names:
    print(f'\n\nProcessing model: {ollama_model_name}')
    input_classified_records_filepath = input_filepath + 'phenotyping_processed_' + make_filename_safe(ollama_model_name) + '_ordered.parquet'

    records_df = pd.read_parquet(input_classified_records_filepath)[['outcome', 'ordered_string']].drop_duplicates()
    print(f'Loaded records DataFrame with shape: {records_df.shape}')
    print('Columns in records_df:', records_df.columns.tolist())
    print('Head of records_df:', records_df.head())

    n_examples = 3

    # ### random order (set seed to ensure reproducibility)
    # records_df = records_df.sample(frac=1, random_state=42).reset_index(drop=True)
    # print('Shuffled records_df head:', records_df.head())

    # print('\nRandomly selected records with outcomes:')
    # for index, row in records_df.iterrows():
    #     records, outcome = row['ordered_string'], row['outcome']
    #     print(f'Example {index + 1}:')
    #     print(f'Outcome: {reverse_replace_dict.get(outcome, outcome)}')
    #     print(f'Records: {records}')
    #     if index + 1 >= n_examples:
    #         break

    ### specific order
    print('\nSpecific order:')
    ### uncomment to run once and save the specific order in the dictionary
    # outcomes = records_df['outcome'].unique()
    # print(f'Unique outcomes: {outcomes}')
    # random_outcome = random.choice(list(outcomes))
    # print(f'Random outcome: {random_outcome}')
    random_outcome_to_models = {
        'mistral-small:24b-instruct-2501-q8_0': 6,
        'gemma2:27b-instruct-q8_0': 6,
        'phi4:14b-q8_0': 6
    }

    filtered_df = records_df[records_df['outcome'] == random_outcome_to_models[ollama_model_name]].copy()
    filtered_df = filtered_df.sample(frac=1, random_state=42).reset_index(drop=True)
    for index, row in filtered_df.iterrows():
        records, outcome = row['ordered_string'], row['outcome']
        print(f'Example {index + 1}:')
        print(f'Outcome: {reverse_replace_dict.get(outcome, outcome)}')
        print(f'Records: {records}')
        if index + 1 >= n_examples:
            break
