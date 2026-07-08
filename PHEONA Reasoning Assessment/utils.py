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