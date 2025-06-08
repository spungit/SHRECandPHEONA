import os
import sys
import re
import ollama
import pandas as pd
from datetime import datetime
import string

from filepaths import *
from prompts import *

def make_filename_safe(input_string):
    safe_string = "".join(char for char in input_string if char not in string.punctuation and not char.isspace())
    return safe_string

def get_response(prompt, record_str, data_model_name, temperature, top_p, port):
    formatted_prompt = prompt.format(description = record_str)

    start_time = datetime.now()
    
    client = ollama.Client(
        host=f'http://localhost:{port}'   
    )
    response = client.chat(model=data_model_name, messages=[
        {
            "role": "user",
            "content": formatted_prompt,
            "temperature": temperature,
            "top_p": top_p
        }
    ])
    end_time = datetime.now()
    latency = (end_time - start_time).total_seconds()
    response_content = response["message"]["content"].strip().upper()
    parsed_response = parse_response(response_content)
    
    return response_content, parsed_response, latency

def parse_response(response_content):
    match = re.search(r"<answer>\s*(YES|NO)\s*</answer>", response_content, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).upper() == "YES"
    return False

def get_descriptions(data_model_name, temperature, top_p, port='11434'):
    description_filename = f"description_mapping_{make_filename_safe(data_model_name)}.csv"

    unique_descriptions_df = pd.read_csv(descriptions_filepath)
    unique_descriptions = unique_descriptions_df["description"].unique()
    print(f"Loaded {len(unique_descriptions)} unique descriptions.")

    if os.path.exists(description_filename):
        already_processed = pd.read_csv(description_filename)
        already_processed_descriptions = already_processed["description"].unique()
        unique_descriptions = [description for description in unique_descriptions if description not in already_processed_descriptions]
        print(f"Already processed {len(already_processed)} descriptions. {len(unique_descriptions)} remaining.")

    m = 5

    description_responses = []
    for i, description in enumerate(unique_descriptions):
        record_str = description.upper()

        is_intubated = unique_descriptions_df[unique_descriptions_df["description"] == description]["intubated"].values[0]
        is_non_invasive = unique_descriptions_df[unique_descriptions_df["description"] == description]["non_invasive"].values[0]
        is_hfni = unique_descriptions_df[unique_descriptions_df["description"] == description]["hfni"].values[0]
        is_ground_truth = unique_descriptions_df[unique_descriptions_df["description"] == description]["ground_truth"].values[0]

        resp_concept_response, resp_concept_parsed_response, resp_concept_latency = get_response(categorization_prompt_wo_meds, record_str, data_model_name, temperature, top_p, port)
        meds_response, meds_parsed_response, meds_latency = get_response(categorization_prompt_w_meds, record_str, data_model_name, temperature, top_p, port)

        description_dict = {
            "description": description,

            "total_tf": True if resp_concept_parsed_response or meds_parsed_response else False,
            "resp_concept_tf": resp_concept_parsed_response,
            "meds_tf": meds_parsed_response,
            "ground_truth": is_ground_truth,
            "is_intubated": is_intubated,
            "is_non_invasive": is_non_invasive,
            "is_hfni": is_hfni,

            "resp_concept_response": resp_concept_response,
            "meds_response": meds_response,

            "resp_concept_latency": resp_concept_latency,
            "meds_latency": meds_latency,

            "resp_concept_prompt": categorization_prompt_wo_meds,
            "meds_prompt": categorization_prompt_w_meds
        }

        description_responses.append(description_dict)
        print(f"\nProcessed {i} out of {len(unique_descriptions)} descriptions")
        print(f"{description_dict.get('description')}")
        print(f'{description_dict.get("total_tf")}')

        # Save every m records
        if (i + 1) % m == 0:
            description_mapping_df = pd.DataFrame(description_responses)
            description_mapping_df.to_csv(description_filename , mode="a", header=not os.path.exists(description_filename), index=False)
            description_responses = []

    # Save any remaining records
    if description_responses:
        description_mapping_df = pd.DataFrame(description_responses)
        description_mapping_df.to_csv(description_filename, mode="a", header=not os.path.exists(description_filename), index=False)

if __name__ == "__main__":
    model_name = str(sys.argv[1])
    port = str(sys.argv[2])
    print('Model name:', model_name)
    print('Port:', port)

    os.environ['OLLAMA_MODELS'] = model_filepath

    if model_name == 'mistral':
        get_descriptions("mistral-small:24b-instruct-2501-q8_0", 0, 0.99, port)
    elif model_name == 'gemma':
        get_descriptions("gemma2:27b-instruct-q8_0", 0, 0.99, port)
    elif model_name == 'llama':
        get_descriptions("llama3.2:3b-instruct-q8_0", 0, 0.99, port)
    elif model_name == 'phi':
        get_descriptions("phi4:14b-q8_0", 0, 0.99, port)
    elif model_name == 'deepseek':
        get_descriptions("deepseek-r1:32b", 0, 0.99, port)