import pandas as pd
import numpy as np

models = ['mistralsmall24binstruct2501q80', 'gemma227binstructq80', 'phi414bq80']

output_filepath = ''

for m in models:
    print(f'\n\nProcessing model: {m}')
    df = pd.read_parquet(output_filepath + 'phenotyping_processed_' + m + '_ordered.parquet')
    print(f'Processing data for model file: {m}')
    print('Shape of model results:', df.shape)
    print('Head of model results:')
    print(df.head())
    print('First ordered string:', df['ordered_string'].iloc[0])
    
    df['ordered_string'] = df['ordered_string'].str.replace('DESCRIPTION: Source = Nurse Charting; Concept = ', '', regex=False)
    df['ordered_string'] = df['ordered_string'].str.replace('DESCRIPTION: Source = Care Plan General; Concept = ', '', regex=False)
    df['ordered_string'] = df['ordered_string'].str.replace('DESCRIPTION: Source = Infusion Drug; Concept = ', '', regex=False)
    df['ordered_string'] = df['ordered_string'].str.replace('DESCRIPTION: Source = Respiratory Care; Concept = ', '', regex=False)
    df['ordered_string'] = df['ordered_string'].str.replace('DESCRIPTION: Source = Respiratory Charting; Concept = ', '', regex=False)
    df['ordered_string'] = df['ordered_string'].str.replace('DESCRIPTION: Source = Treatment; Concept = ', '', regex=False)
    df['ordered_string'] = df['ordered_string'].str.replace('DESCRIPTION: Source = Medication; Concept = ', '', regex=False)
    df['ordered_string'] = df['ordered_string'].str.replace('DESCRIPTION: Source = Nurse Care; Concept = ', '', regex=False)

    print('Head of df after replacing strings:')
    print(df.head())
    print('First ordered string:', df['ordered_string'].iloc[0])

    df.to_parquet(output_filepath + 'phenotyping_processed_' + m + '_ordered.parquet', index=False)