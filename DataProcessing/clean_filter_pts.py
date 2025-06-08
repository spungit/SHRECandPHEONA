import pandas as pd

def transform_column(value, index):
    lines = value.split("\n")
    numbered_lines = []
    for i, line in enumerate(lines):
        description_start = line.find("DESCRIPTION:")
        if description_start != -1:
            numbered_lines.append(f"{i + 1}: {line[description_start:]}")
        else:
            numbered_lines.append(f"{i + 1}: {line}")
    return "\n".join(numbered_lines)

filepath = ''

models = ['phenotyping_processed_phi414bq80.csv', 'phenotyping_processed_gemma227binstructq80.csv', 'phenotyping_processed_mistralsmall24binstruct2501q80.csv']

for m in models:
    print(f'Processing data for model file: {m}')
    results = pd.read_csv(filepath + m, encoding='latin1')
    print('Shape of model results:', results.shape)
    print('Head of model results:')
    print(results.head())

    ## identify the patientunitstayids that are in the cohort
    cohort_pts = pd.read_parquet(filepath + 'pts_with_outcomes.parquet', columns=['patientunitstayid'])
    cohort_pts['patientunitstayid'] = cohort_pts['patientunitstayid'].astype('int64')
    results['patientunitstayid'] = results['patientunitstayid'].astype('int64')
    results = pd.merge(results, cohort_pts, on='patientunitstayid', how='inner')
    print('Shape of model results after merging with cohort:', results.shape)
    print(f'Unique patientunitstayid: {results["patientunitstayid"].nunique()}')

    ## clean the processed_string column to remove the offset value and just order the records
    results['ordered_string'] = results['processed_string'].apply(lambda x: transform_column(x, results.index[results['processed_string'] == x][0]))
    print('Head of transformed results:')
    print(results.head())
    print('First and second rows:')
    print(results.iloc[0:2])
    print('First and second rows of ordered_string:')
    print(results['ordered_string'].iloc[0:2])
    print('First and second rows of processed_string:')
    print(results['processed_string'].iloc[0:2])
    print('Number of Distinct Ordered Strings:', results['ordered_string'].nunique(), ' out of ', results.shape[0])

    ## save the results to a new file
    results.to_csv(filepath + m.replace('.csv', '_ordered.csv'), index=False)
    results.to_parquet(filepath + m.replace('.csv', '_ordered.parquet'), index=False)