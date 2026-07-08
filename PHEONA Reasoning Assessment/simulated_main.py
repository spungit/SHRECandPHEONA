import os
import pandas as pd
import numpy as np
import sys

from filepaths import *
from prompts import *
from utils import *


def get_fewshothint_error_results(input_classified_records_filepath, output_classified_records_filepath, data_model_name, temperature, top_p, port='11434'):
    records = pd.read_parquet(input_classified_records_filepath)
    unique_descriptions = records['description'].unique()
    gts_by_description = records[['description', 'outcome']].drop_duplicates().set_index('description')['outcome'].to_dict()
    print(f'\nLoaded {len(unique_descriptions)} unique descriptions.')
    del records

    if os.path.exists(output_classified_records_filepath):
        already_processed_df = pd.read_excel(output_classified_records_filepath)
        n_processed = already_processed_df.shape[0]
        already_saved = set(zip(already_processed_df["description"], already_processed_df["trial"]))
        print(f'\nAlready processed {n_processed} description-trial combinations.')
    else:
        already_saved = set()
        print('No existing output file found. Starting fresh.')

    trials = 3

    for trial in range(1, trials + 1):
        print(f'\nStarting trial {trial}/{trials}')

        # Determine already saved descriptions for this trial
        already_saved_this_trial = {desc for desc, t in already_saved if t == trial}
        print(f'Trial {trial} already has {len(already_saved_this_trial)} saved descriptions.')

        # Set seed for this trial
        seed = 42 + trial - 1
        np.random.seed(seed)
        np.random.shuffle(unique_descriptions)
        print(f'Shuffled descriptions with seed {seed}.')

        unique_descriptions_df = pd.DataFrame(unique_descriptions, columns=['description'])
        unique_descriptions_df['outcome'] = unique_descriptions_df['description'].map(gts_by_description)
        print(f'Number of descriptions for trial {trial}:', unique_descriptions_df.shape[0])

        desc_df_rows = []
        n_saved = 0

        for index, row in unique_descriptions_df.iterrows():
            description, ground_truth = row['description'], int(row['outcome'])
            
            if description in already_saved_this_trial:
                continue

            print(f'\nProcessing description {index+1}/{len(unique_descriptions_df)} for trial {trial}')
            print(f'Description:\n{description}')
            print('Ground truth:', ground_truth)

            incremented_gt = -1 if ground_truth + 1 == 6 else ground_truth + 1
            print('Incremented ground truth for hinting:', incremented_gt)
            hint_ground_truth = reverse_replace_dict.get(incremented_gt)
            print('Hint ground truth:', hint_ground_truth)

            if int(ground_truth) != 6:  # all of the examples provided are for the 'IMV TO HFNI' outcome, which is represented by 6 in the replace_dict

                ### get the original responses
                formatted_no_cot_prompt = instructions.format(description = description) + no_cot_questions
                formatted_some_cot_prompt = instructions.format(description = description) + some_cot_questions
                formatted_full_cot_prompt = instructions.format(description = description) + full_cot_questions

                no_cot_response, no_cot_parsed_response, no_cot_latency = get_response(formatted_no_cot_prompt, description, data_model_name, temperature, top_p, port, '1')
                some_cot_response, some_cot_parsed_response, some_cot_latency = get_response(formatted_some_cot_prompt, description, data_model_name, temperature, top_p, port, '4')
                full_cot_response, full_cot_parsed_response, full_cot_latency = get_response(formatted_full_cot_prompt, description, data_model_name, temperature, top_p, port, '8')
                
                ## get the responses for the different few-shot prompts with and without hints
                fewshot_types = [
                    ("randomfewshot", random_order_input),
                    ("specificfewshot", specific_order_input)
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
                                + examples_prompt.format(examples=fewshot_inputs)
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
                    'trial': trial,
                    'description': description,
                    'outcome': ground_truth,
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

                # Save immediately after processing
                desc_df = pd.DataFrame([desc_dict])
                if os.path.exists(output_classified_records_filepath):
                    existing_df = pd.read_excel(output_classified_records_filepath)
                    desc_df = pd.concat([existing_df, desc_df], ignore_index=True)
                desc_df.to_excel(output_classified_records_filepath, index=False)
                desc_df_rows = []  # Reset since saved

    print('\nCompleted processing all descriptions.')

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
        
    ## column names for final df: ['patientunitstayid', 'gt_outcome', 'llm_outcome', 'latency', 'records', 'full_response', 'formatted_prompt']
    input_patients_outcomes_filepath = input_filepath + 'pts_with_outcomes.parquet'
    input_classified_records_filepath = input_filepath + 'fewshothint_inputdata.parquet'
    output_patients_outcomes_filepath = output_filepath + 'phenotyped_pts_' + make_filename_safe(ollama_model_name) + '.xlsx'
    output_classified_records_filepath = output_filepath + 'simulated_results_' + make_filename_safe(ollama_model_name) + '.xlsx'

    get_fewshothint_error_results(input_classified_records_filepath, output_classified_records_filepath, ollama_model_name, 0.5, 0.99, port=port)
