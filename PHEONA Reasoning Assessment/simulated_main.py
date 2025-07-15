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

    if os.path.exists(output_classified_records_filepath):
        already_processed_df = pd.read_csv(output_classified_records_filepath)
        n_processed = already_processed_df.shape[0]
        already_processed_descriptions = already_processed_df["description"].unique()
        print(f'\nAlready processed {n_processed} descriptions.')

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
    n_saved = 0 if not os.path.exists(output_classified_records_filepath) else n_processed
    n_to_save = 1000
    for index, row in unique_descriptions.iterrows():
        print('\n\nNumber of saved records:', n_saved)
        if n_saved >= n_to_save:
            print(f'\nSaved {n_saved} records. Stopping early.')
            break

        description, ground_truth = row['records'], int(row['gt_outcome'])
        if description in already_processed_descriptions:
            print(f'\nDescription already processed. Skipping: {description}')
            continue
        hint_ground_truth = reverse_replace_dict.get(-1 if ground_truth + 1 == 6 else ground_truth + 1, '')
        print(f'Description:\n{description}')
        print('Ground truth:', ground_truth)
        print('Hint ground truth:', hint_ground_truth)
        if hint_ground_truth == '':
            print('Hint ground truth is empty. Skipping this description.')
            continue

        if int(ground_truth) != 6: # all of the examples provided are for the 'IMV TO HFNI' outcome, which is represented by 6 in the replace_dict

            ### get the original responses
            formatted_no_cot_prompt = instructions.format(description = description) + no_cot_questions
            formatted_some_cot_prompt = instructions.format(description = description) + some_cot_questions
            formatted_full_cot_prompt = instructions.format(description = description) + full_cot_questions

            no_cot_response, no_cot_parsed_response, no_cot_latency = get_response(formatted_no_cot_prompt, description, data_model_name, temperature, top_p, port, '1')
            some_cot_response, some_cot_parsed_response, some_cot_latency = get_response(formatted_some_cot_prompt, description, data_model_name, temperature, top_p, port, '4')
            full_cot_response, full_cot_parsed_response, full_cot_latency = get_response(formatted_full_cot_prompt, description, data_model_name, temperature, top_p, port, '8')
            
            ## get the responses for the different few-shot prompts with and without hints
            fewshot_types = [
                ("randomfewshot", random_order_inputs),
                ("specificfewshot", specific_order_inputs)
            ]
            cot_levels = [
                ("nocot", no_cot_questions, '1'),
                ("somecot", some_cot_questions, '4'),
                ("fullcot", full_cot_questions, '8')
            ]
            hint_options = [
                ("nohint", ""),
                ("hint", hint_prompt.format(ground_truth=hint_ground_truth))
            ]

            responses = {}

            for fewshot_name, fewshot_inputs in fewshot_types:
                for hint_name, hint_str in hint_options:
                    for cot_name, cot_questions, question_num in cot_levels:
                        prompt = (
                            instructions.format(description=description)
                            + examples_prompt.format(examples=fewshot_inputs.get(data_model_name, ''))
                            + hint_str
                            + cot_questions
                        )
                        response, parsed_response, latency = get_response(
                            prompt, description, data_model_name, temperature, top_p, port, question_num
                        )
                        key = f"{cot_name}_{fewshot_name}_{hint_name}"
                        responses[f"{key}_response"] = response
                        responses[f"{key}_parsed_response"] = parsed_response
                        responses[f"{key}_latency"] = latency
                        responses[f"{key}_formatted_prompt"] = prompt

            ## get just the responses for hinting only
            nocot_noexamples_hint_prompt = instructions.format(description=description) + hint_prompt.format(ground_truth=hint_ground_truth) + no_cot_questions
            somecot_noexamples_hint_prompt = instructions.format(description=description) + hint_prompt.format(ground_truth=hint_ground_truth) + some_cot_questions
            fullcot_noexamples_hint_prompt = instructions.format(description=description) + hint_prompt.format(ground_truth=hint_ground_truth) + full_cot_questions

            nocot_noexamples_hint_response, nocot_noexamples_hint_parsed_response, nocot_noexamples_hint_latency = get_response(nocot_noexamples_hint_prompt, description, data_model_name, temperature, top_p, port, '1')
            somecot_noexamples_hint_response, somecot_noexamples_hint_parsed_response, somecot_noexamples_hint_latency = get_response(somecot_noexamples_hint_prompt, description, data_model_name, temperature, top_p, port, '4')
            fullcot_noexamples_hint_response, fullcot_noexamples_hint_parsed_response, fullcot_noexamples_hint_latency = get_response(fullcot_noexamples_hint_prompt, description, data_model_name, temperature, top_p, port, '8')

            ## save all the responses to a dictionary
            desc_dict = {
                'description': description,
                'gt_outcome': ground_truth,
                'hint_gt_outcome': hint_ground_truth,
                'no_cot_response': no_cot_response,
                'no_cot_parsed_response': no_cot_parsed_response,
                'no_cot_latency': no_cot_latency,
                'no_cot_formatted_prompt': formatted_no_cot_prompt,
                'some_cot_response': some_cot_response,
                'some_cot_parsed_response': some_cot_parsed_response,
                'some_cot_latency': some_cot_latency,
                'some_cot_formatted_prompt': formatted_some_cot_prompt,
                'full_cot_response': full_cot_response,
                'full_cot_parsed_response': full_cot_parsed_response,
                'full_cot_latency': full_cot_latency,
                'full_cot_formatted_prompt': formatted_full_cot_prompt,
                'nocot_noexamples_hint_response': nocot_noexamples_hint_response,
                'nocot_noexamples_hint_parsed_response': nocot_noexamples_hint_parsed_response,
                'nocot_noexamples_hint_latency': nocot_noexamples_hint_latency,
                'nocot_noexamples_hint_formatted_prompt': nocot_noexamples_hint_prompt,
                'somecot_noexamples_hint_response': somecot_noexamples_hint_response,
                'somecot_noexamples_hint_parsed_response': somecot_noexamples_hint_parsed_response,
                'somecot_noexamples_hint_latency': somecot_noexamples_hint_latency,
                'somecot_noexamples_hint_formatted_prompt': somecot_noexamples_hint_prompt,
                'fullcot_noexamples_hint_response': fullcot_noexamples_hint_response,
                'fullcot_noexamples_hint_parsed_response': fullcot_noexamples_hint_parsed_response,
                'fullcot_noexamples_hint_latency': fullcot_noexamples_hint_latency,
                'fullcot_noexamples_hint_formatted_prompt': fullcot_noexamples_hint_prompt,
                **responses  # This unpacks and adds all key-value pairs from the responses dictionary
            }

            desc_df_rows.append(desc_dict)
            n_saved += 1
            print('Response saved to final dataframe.')

        # Save every m records
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
    output_classified_records_filepath = output_filepath + 'simulated_results_' + make_filename_safe(ollama_model_name) + '.csv'
    
    ## get the patients and their outcomes and records
    if os.path.exists(output_patients_outcomes_filepath):
        patients_df = pd.read_csv(output_patients_outcomes_filepath)
    else:
        patients_df = get_final_df(input_patients_outcomes_filepath, input_classified_records_filepath)
        patients_df.to_csv(output_patients_outcomes_filepath, index=False)

    ## classify the unique descriptions
    gts_by_record = patients_df.groupby('records')['gt_outcome'].max().reset_index()
    get_descriptions(input_classified_records_filepath, output_classified_records_filepath, gts_by_record, ollama_model_name, 0.5, 0.99, port=port)
