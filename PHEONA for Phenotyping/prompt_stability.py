import os
import re
import ollama
import pandas as pd
from datetime import datetime
import string
import numpy as np
import sys
import tiktoken

cot_prompt_test_1 = """
INSTRUCTIONS:
1) INPUT: The input, delimited by <input></input>, will contain a SERIES OF RECORDS from a patient's stay in the ICU. Each individual record (or row) will contain a description and will be ordered based on the occurrence of the description in the patient's stay. Each record will be in the following format: ORDER OF RECORD: <description>. **DO NOT** fabricate any information or make assumptions about the patient's records.

<input>
{description}
</input>

2) TREATMENTS:

    - **Treatment 1: Invasive Mechanical Ventilation (IMV)**
        - **INCLUSION CRITERIA**:
            1) At least ONE INDIVIDUAL record indicating the patient received **AT LEAST ONE** of the following medications: Specific Sedatives (Etomidate, Ketamine, Midazolam (Versed), Propofol, Dexmedetomidine (Precedex), Fentanyl, Morphine, Hydromorphone (Dilaudid), Thiopental, Cisatracurium) or Specific Paralytics (Rocuronium, Succinylcholine, Vecuronium).
            AND
            2) At least TWO INDIVIDUAL records indicating the patient was on invasive mechanical ventilation (IMV) or intubated. **EXCLUDES** records defining ventilation settings. Invasive mechanical ventilation involves a tube in the trachea (either an endotracheal tube placed through the mouth, or rarely the nose, OR a surgically placed traheostomy tube) connected to a ventilator, delivering mechanical ventilation. Records with the following terms or acronyms should be considered for IMV unless otherwise indicated: ventilator, ETT or ET (endotracheal tube, trach tube), tracheostomy/trach,  PS (pressure support), AC (assist control vent mode), CMV (continuous mandatory ventilation vent mode), SIMV (synchronized intermittent mandatory ventilation vent mode), PRVC (pressure regulated volume control vent mode), APRV or Bi-level (airway pressure release ventilation vent mode).

    - **Treatment 2: Non-Invasive Positive Pressure Ventilation (NIPPV)**
        - **INCLUSION CRITERIA**:
            1) At least TWO INDIVIDUAL records indicating the patient was on non-invasive positive pressure ventilation (NIPPV) **THAT DOES NOT INDICATE** high flow nasal insufflation/cannula or nasal cannula. Also **EXCLUDES** records defining ventilation settings. Non-invasive positive pressure ventilation involves ventilation via a facemask, where the clinician adjusts pressure and oxygen settings. Records with the following terms and acronyms should be considered NIPPV unless otherwise indicated: mask, mask ventilation, NIV (non-invasive ventilation), BiPAP (bilevel positive airway pressure), CPAP (continuous positive airway pressure), IPAP (inspiratory positive airway pressure), EPAP (expiratory positive airway pressure), AVAPS (average volume assured pressure support).

    - **Treatment 3: High-Flow Nasal Insufflation/Nasal Cannula (HFNI/HFNC) or Nasal Cannula**
        - **INCLUSION CRITERIA**:
            1) The criteria for NIPPV is met where the records are **INDEPENDENT** of any records indicating HFNI or nasal cannula.
            AND
            2) At least ONE INDIVIDUAL record indicating the patient was on high flow nasal insufflation/cannula or nasal canula. HFNI/HFNC involves oxygen delivery through a nasal cannula at a flow rate above 15 L/min, with adjustments for oxygen concentration and flow rate. Records with the following terms and acronyms should be considered HFNI/HFNC unless otherwise indicated: nasal canunla (NC), high flow nasal cannula, high flow nasal oxygen, high flow nasal insufflation, high flow nasal therapy, high flow nasal oxygen therapy, high flow nasal oxygen delivery, high flow nasal oxygen therapy (HFNOT), Optiflow, Vapotherm, Airvo.

3) OBJECTIVE: Respond to the questions delimited by the <output></output> tags, including the delimiters in your response. Provide your answer **exactly** in the format specified between the <output></output> tags. Do **NOT** do any of the following:
    - Modify the format of the questions or answers.
    - Provide explanations or additional details beyond the format requested.
    - Fabricate an input or add information that is not present in the input, even if it is empty or unclear.           
                    
OUTPUT:
<output>
SUMMARY:
Q1) Summarize the input records in 3-5 sentences.
A1)

TREATMENT TYPES:
Q2) Are any of the required medications present? If so, are there at least TWO INDIVIDUAL records indicating the patient was on invasive mechanical ventilation (IMV) or intubated? Provide a brief judgment (1-2 sentences) and 'YES' or 'NO' for whether the inclusion criteria for IMV is met.
A2)

Q3) Are there at least TWO INDEPENDENT records indicating the patient was on NIPPV that are ALSO INDEPENDENT of any records indicating HFNI or nasal cannula? Provide a brief judgment (1-2 sentences) and 'YES' or 'NO' for whether the inclusion criteria for NIPPV is met.
A3)

Q4) Based on the records provided, was the criteria for NIPPV met first? If the criteria for NIPPV was not met, then the criteria for HFNI is also not met. If the criteria for NIPPV was met, is there at least ONE ADDITIONAL record indicating HFNI or nasal cannula? Provide a brief judgment (1-2 sentences) and 'YES' or 'NO' for whether the inclusion criteria for HFNI is met.
A4)

TREATMENT ORDERING:
Q5) Based on the previous three questions (Q2-Q4), was there MORE THAN ONE treatment present? **REMEMBER**: If the criteria for HFNI is met, then ONLY HFNI applies, **NOT** NIPPV or HFNI and NIPPV. If so, list the treatments and skip to Q6. If not, skip to Q8.
A5)

Q6) What was the start and end record orders for each of the following: 1) IMV, 2) NIPPV (if applicable), and 3) HFNI (if applicable)? Provide a brief judgment (1-2 sentences).
A6)

Q7) Based on the start and end record orders, are the NIPPV or HFNI records independent of the IMV records? In other words, were ALL of the QUALIFYING RECORDS for NIPPV or HFNI completely BEFORE or AFTER the IMV records and NOT BETWEEN the IMV records? Provide a brief judgment (1-2 sentences). Remember, the QUALIFYING RECORDS for HFNI include the qualifying NIPPV records and an additional record indicating HFNI or nasal cannula so if HFNI was present, ALL THE QUALIFYING RECORDS must be independent of the IMV records.
A7)

FINAL CLASSIFICATION:
Q8) Based on your answers to the previous questions (Q2-Q7), which category does the patient's records fall under? **ONLY** respond with **ONE** of the following: IMV ONLY, NIPPV ONLY, HFNI ONLY, NIPPV TO IMV, HFNI TO IMV, IMV TO NIPPV, IMV TO HFNI, or NONE (if no records or specific treatments were present).
A8)
</output>
"""

cot_prompt_test_2 = """
INSTRUCTIONS:
1) INPUT: The input, delimited by <input></input>, will contain a SERIES OF RECORDS from a patient's stay in the ICU. Each individual record (or row) will contain a description and will be ordered based on the occurrence of the description in the patient's stay. Each record will be in the following format: ORDER OF RECORD: <description>. **DO NOT** fabricate any information or make assumptions about the patient's records.

<input>
{description}
</input>

2) OBJECTIVE: Respond to the questions delimited by the <output></output> tags, including the delimiters in your response. Provide your answer **exactly** in the format specified between the <output></output> tags. Do **NOT** do any of the following:
    - Modify the format of the questions or answers.
    - Provide explanations or additional details beyond the format requested.
    - Fabricate an input or add information that is not present in the input, even if it is empty or unclear.

3) TREATMENTS:

    - **Treatment 1: Invasive Mechanical Ventilation (IMV)**
        - **INCLUSION CRITERIA**:
            1) At least ONE INDIVIDUAL record indicating the patient received **AT LEAST ONE** of the following medications: Specific Sedatives (Etomidate, Ketamine, Midazolam (Versed), Propofol, Dexmedetomidine (Precedex), Fentanyl, Morphine, Hydromorphone (Dilaudid), Thiopental, Cisatracurium) or Specific Paralytics (Rocuronium, Succinylcholine, Vecuronium).
            AND
            2) At least TWO INDIVIDUAL records indicating the patient was on invasive mechanical ventilation (IMV) or intubated. **EXCLUDES** records defining ventilation settings. Invasive mechanical ventilation involves a tube in the trachea (either an endotracheal tube placed through the mouth, or rarely the nose, OR a surgically placed traheostomy tube) connected to a ventilator, delivering mechanical ventilation. Records with the following terms or acronyms should be considered for IMV unless otherwise indicated: ventilator, ETT or ET (endotracheal tube, trach tube), tracheostomy/trach,  PS (pressure support), AC (assist control vent mode), CMV (continuous mandatory ventilation vent mode), SIMV (synchronized intermittent mandatory ventilation vent mode), PRVC (pressure regulated volume control vent mode), APRV or Bi-level (airway pressure release ventilation vent mode).

    - **Treatment 2: Non-Invasive Positive Pressure Ventilation (NIPPV)**
        - **INCLUSION CRITERIA**:
            1) At least TWO INDIVIDUAL records indicating the patient was on non-invasive positive pressure ventilation (NIPPV) **THAT DOES NOT INDICATE** high flow nasal insufflation/cannula or nasal cannula. Also **EXCLUDES** records defining ventilation settings. Non-invasive positive pressure ventilation involves ventilation via a facemask, where the clinician adjusts pressure and oxygen settings. Records with the following terms and acronyms should be considered NIPPV unless otherwise indicated: mask, mask ventilation, NIV (non-invasive ventilation), BiPAP (bilevel positive airway pressure), CPAP (continuous positive airway pressure), IPAP (inspiratory positive airway pressure), EPAP (expiratory positive airway pressure), AVAPS (average volume assured pressure support).

    - **Treatment 3: High-Flow Nasal Insufflation/Nasal Cannula (HFNI/HFNC) or Nasal Cannula**
        - **INCLUSION CRITERIA**:
            1) The criteria for NIPPV is met where the records are **INDEPENDENT** of any records indicating HFNI or nasal cannula.
            AND
            2) At least ONE INDIVIDUAL record indicating the patient was on high flow nasal insufflation/cannula or nasal canula. HFNI/HFNC involves oxygen delivery through a nasal cannula at a flow rate above 15 L/min, with adjustments for oxygen concentration and flow rate. Records with the following terms and acronyms should be considered HFNI/HFNC unless otherwise indicated: nasal canunla (NC), high flow nasal cannula, high flow nasal oxygen, high flow nasal insufflation, high flow nasal therapy, high flow nasal oxygen therapy, high flow nasal oxygen delivery, high flow nasal oxygen therapy (HFNOT), Optiflow, Vapotherm, Airvo.
        
OUTPUT:
<output>
SUMMARY:
Q1) Summarize the input records in 3-5 sentences.
A1)

TREATMENT TYPES:
Q2) Are there at least TWO INDEPENDENT records indicating the patient was on NIPPV that are ALSO INDEPENDENT of any records indicating HFNI or nasal cannula? Provide a brief judgment (1-2 sentences) and 'YES' or 'NO' for whether the inclusion criteria for NIPPV is met.
A2)

Q3) Based on the records provided, was the criteria for NIPPV met first? If the criteria for NIPPV was not met, then the criteria for HFNI is also not met. If the criteria for NIPPV was met, is there at least ONE ADDITIONAL record indicating HFNI or nasal cannula? Provide a brief judgment (1-2 sentences) and 'YES' or 'NO' for whether the inclusion criteria for HFNI is met.
A3)

Q4) Are any of the required medications present? If so, are there at least TWO INDIVIDUAL records indicating the patient was on invasive mechanical ventilation (IMV) or intubated? Provide a brief judgment (1-2 sentences) and 'YES' or 'NO' for whether the inclusion criteria for IMV is met.
A4)

TREATMENT ORDERING:
Q5) Based on the start and end record orders, are the NIPPV or HFNI records independent of the IMV records? In other words, were ALL of the QUALIFYING RECORDS for NIPPV or HFNI completely BEFORE or AFTER the IMV records and NOT BETWEEN the IMV records? Provide a brief judgment (1-2 sentences). Remember, the QUALIFYING RECORDS for HFNI include the qualifying NIPPV records and an additional record indicating HFNI or nasal cannula so if HFNI was present, ALL THE QUALIFYING RECORDS must be independent of the IMV records.
A5)

Q6) What was the start and end record orders for each of the following: 1) IMV, 2) NIPPV (if applicable), and 3) HFNI (if applicable)? Provide a brief judgment (1-2 sentences).
A6)

Q7) Based on the previous three questions (Q2-Q4), was there MORE THAN ONE treatment present? **REMEMBER**: If the criteria for HFNI is met, then ONLY HFNI applies, **NOT** NIPPV or HFNI and NIPPV. If so, list the treatments and skip to Q6. If not, skip to Q8.
A7)

FINAL CLASSIFICATION:
Q8) Based on your answers to the previous questions (Q2-Q7), which category does the patient's records fall under? **ONLY** respond with **ONE** of the following: IMV ONLY, NIPPV ONLY, HFNI ONLY, NIPPV TO IMV, HFNI TO IMV, IMV TO NIPPV, IMV TO HFNI, or NONE (if no records or specific treatments were present).
A8)
</output>
"""

cot_prompt_test_3 = """
INSTRUCTIONS:
1) INPUT: The input, delimited by <input></input>, will contain a SERIES OF RECORDS from a patient's stay in the ICU. Each individual record (or row) will contain a description and will be ordered based on the occurrence of the description in the patient's stay. Each record will be in the following format: ORDER OF RECORD: <description>. **DO NOT** fabricate any information or make assumptions about the patient's records.

<input>
{description}
</input>

2) OBJECTIVE: Respond to the questions delimited by the <output></output> tags, including the delimiters in your response. Provide your answer **exactly** in the format specified between the <output></output> tags. Do **NOT** do any of the following:
    - Modify the format of the questions or answers.
    - Provide explanations or additional details beyond the format requested.
    - Fabricate an input or add information that is not present in the input, even if it is empty or unclear.

3) TREATMENTS:

    - **Treatment 1: High-Flow Nasal Insufflation/Nasal Cannula (HFNI/HFNC) or Nasal Cannula**
        - **INCLUSION CRITERIA**:
            1) The criteria for NIPPV is met where the records are **INDEPENDENT** of any records indicating HFNI or nasal cannula.
            AND
            2) At least ONE INDIVIDUAL record indicating the patient was on high flow nasal insufflation/cannula or nasal canula. HFNI/HFNC involves oxygen delivery through a nasal cannula at a flow rate above 15 L/min, with adjustments for oxygen concentration and flow rate. Records with the following terms and acronyms should be considered HFNI/HFNC unless otherwise indicated: nasal canunla (NC), high flow nasal cannula, high flow nasal oxygen, high flow nasal insufflation, high flow nasal therapy, high flow nasal oxygen therapy, high flow nasal oxygen delivery, high flow nasal oxygen therapy (HFNOT), Optiflow, Vapotherm, Airvo.

    - **Treatment 2: Non-Invasive Positive Pressure Ventilation (NIPPV)**
        - **INCLUSION CRITERIA**:
            1) At least TWO INDIVIDUAL records indicating the patient was on non-invasive positive pressure ventilation (NIPPV) **THAT DOES NOT INDICATE** high flow nasal insufflation/cannula or nasal cannula. Also **EXCLUDES** records defining ventilation settings. Non-invasive positive pressure ventilation involves ventilation via a facemask, where the clinician adjusts pressure and oxygen settings. Records with the following terms and acronyms should be considered NIPPV unless otherwise indicated: mask, mask ventilation, NIV (non-invasive ventilation), BiPAP (bilevel positive airway pressure), CPAP (continuous positive airway pressure), IPAP (inspiratory positive airway pressure), EPAP (expiratory positive airway pressure), AVAPS (average volume assured pressure support).

    - **Treatment 3: Invasive Mechanical Ventilation (IMV)**
        - **INCLUSION CRITERIA**:
            1) At least ONE INDIVIDUAL record indicating the patient received **AT LEAST ONE** of the following medications: Specific Sedatives (Etomidate, Ketamine, Midazolam (Versed), Propofol, Dexmedetomidine (Precedex), Fentanyl, Morphine, Hydromorphone (Dilaudid), Thiopental, Cisatracurium) or Specific Paralytics (Rocuronium, Succinylcholine, Vecuronium).
            AND
            2) At least TWO INDIVIDUAL records indicating the patient was on invasive mechanical ventilation (IMV) or intubated. **EXCLUDES** records defining ventilation settings. Invasive mechanical ventilation involves a tube in the trachea (either an endotracheal tube placed through the mouth, or rarely the nose, OR a surgically placed traheostomy tube) connected to a ventilator, delivering mechanical ventilation. Records with the following terms or acronyms should be considered for IMV unless otherwise indicated: ventilator, ETT or ET (endotracheal tube, trach tube), tracheostomy/trach,  PS (pressure support), AC (assist control vent mode), CMV (continuous mandatory ventilation vent mode), SIMV (synchronized intermittent mandatory ventilation vent mode), PRVC (pressure regulated volume control vent mode), APRV or Bi-level (airway pressure release ventilation vent mode).
        
OUTPUT:
<output>
SUMMARY:
Q1) Summarize the input records in 3-5 sentences.
A1)

TREATMENT TYPES:
Q2) Are any of the required medications present? If so, are there at least TWO INDIVIDUAL records indicating the patient was on invasive mechanical ventilation (IMV) or intubated? Provide a brief judgment (1-2 sentences) and 'YES' or 'NO' for whether the inclusion criteria for IMV is met.
A2)

Q3) Are there at least TWO INDEPENDENT records indicating the patient was on NIPPV that are ALSO INDEPENDENT of any records indicating HFNI or nasal cannula? Provide a brief judgment (1-2 sentences) and 'YES' or 'NO' for whether the inclusion criteria for NIPPV is met.
A3)

Q4) Based on the records provided, was the criteria for NIPPV met first? If the criteria for NIPPV was not met, then the criteria for HFNI is also not met. If the criteria for NIPPV was met, is there at least ONE ADDITIONAL record indicating HFNI or nasal cannula? Provide a brief judgment (1-2 sentences) and 'YES' or 'NO' for whether the inclusion criteria for HFNI is met.
A4)

TREATMENT ORDERING:
Q5) Based on the previous three questions (Q2-Q4), was there MORE THAN ONE treatment present? **REMEMBER**: If the criteria for HFNI is met, then ONLY HFNI applies, **NOT** NIPPV or HFNI and NIPPV. If so, list the treatments and skip to Q6. If not, skip to Q8.
A5)

Q6) What was the start and end record orders for each of the following: 1) IMV, 2) NIPPV (if applicable), and 3) HFNI (if applicable)? Provide a brief judgment (1-2 sentences).
A6)

Q7) Based on the start and end record orders, are the NIPPV or HFNI records independent of the IMV records? In other words, were ALL of the QUALIFYING RECORDS for NIPPV or HFNI completely BEFORE or AFTER the IMV records and NOT BETWEEN the IMV records? Provide a brief judgment (1-2 sentences). Remember, the QUALIFYING RECORDS for HFNI include the qualifying NIPPV records and an additional record indicating HFNI or nasal cannula so if HFNI was present, ALL THE QUALIFYING RECORDS must be independent of the IMV records.
A7)

FINAL CLASSIFICATION:
Q8) Based on your answers to the previous questions (Q2-Q7), which category does the patient's records fall under? **ONLY** respond with **ONE** of the following: IMV ONLY, NIPPV ONLY, HFNI ONLY, NIPPV TO IMV, HFNI TO IMV, IMV TO NIPPV, IMV TO HFNI, or NONE (if no records or specific treatments were present).
A8)
</output>
"""

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

def get_descriptions(stability_prompts, unique_descriptions, data_model_name, temperature, top_p, port='11434'):
    n_trials = 10

    desc_df_rows = []
    for i, d in enumerate(unique_descriptions):
        print(f'\nProcessing description {i + 1} out of {len(unique_descriptions)}: {d}')
        for k in range(n_trials):
            formatted_prompt_1 = stability_prompts[0].format(description=d)
            formatted_prompt_2 = stability_prompts[1].format(description=d)
            formatted_prompt_3 = stability_prompts[2].format(description=d)

            total_response_1, parsed_response_1, latency_1 = get_response(stability_prompts[0], d, data_model_name, temperature, top_p, port)
            total_response_2, parsed_response_2, latency_2 = get_response(stability_prompts[1], d, data_model_name, temperature, top_p, port)
            total_response_3, parsed_response_3, latency_3 = get_response(stability_prompts[2], d, data_model_name, temperature, top_p, port)

            desc_dict = {'trial': k + 1,
                         'records': d,
                         'llm_outcome_1': parsed_response_1,
                         'llm_outcome_2': parsed_response_2,
                         'llm_outcome_3': parsed_response_3,
                         'latency_1': latency_1,
                         'latency_2': latency_2,
                         'latency_3': latency_3,
                         'full_response_1': total_response_1,
                         'full_response_2': total_response_2,
                         'full_response_3': total_response_3,
                         'formatted_prompt_1': formatted_prompt_1,
                         'formatted_prompt_2': formatted_prompt_2,
                         'formatted_prompt_3': formatted_prompt_3}
            
            desc_df_rows.append(desc_dict)
            print(f'Processed trial {k + 1} for description {i + 1} out of {len(unique_descriptions)}')
        print(f'Processed {i + 1} out of {len(unique_descriptions)} descriptions')

    desc_df = pd.DataFrame(desc_df_rows)
    print('\nCompleted processing all descriptions.')
    return desc_df

if __name__ == "__main__":
    model_filepath = ''
    output_filepath = ''

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
        
    ## column names for final df: ['patientunitstayid', 'gt_outcome', 'llm_outcome', 'latency', 'records', 'full_response', 'formatted_prompt']    
    input_classified_records_filepath = output_filepath + 'final_phenotyped_pts_' + make_filename_safe(ollama_model_name) + '_sample_2.csv'
    output_final_filepath = output_filepath + 'prompt_stability_' + make_filename_safe(ollama_model_name) + '.csv'

    ## classify the unique descriptions based on the sample records
    input_classified_records_df = pd.read_csv(input_classified_records_filepath)
    print('Number of patients loaded:', input_classified_records_df['patientunitstayid'].nunique())
    print('Number of unique records loaded:', input_classified_records_df['records'].nunique())
    top_ten_records = input_classified_records_df['records'].value_counts().head(10)
    print(f'Loaded {len(top_ten_records)} unique descriptions.')
    top_ten_records = top_ten_records.index.tolist()

    prompt_stability_results = get_descriptions([cot_prompt_test_1, cot_prompt_test_2, cot_prompt_test_3], top_ten_records, ollama_model_name, 0, 0.99, port=port)
    prompt_stability_results.to_csv(output_final_filepath, index = False)
    print('Prompt stability results saved to:', output_final_filepath)
    print('Shape of the DataFrame:', prompt_stability_results.shape)
    print('Columns in the DataFrame:', prompt_stability_results.columns.tolist())
    print('Head of the DataFrame:')
    print(prompt_stability_results.head())

