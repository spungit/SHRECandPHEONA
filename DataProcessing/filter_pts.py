import pandas as pd
import os
import gc

def process_patient_data(parquet_folder, all_pts_data_filename, description_folder, model_file, chunk_size=100):
    all_pts_data = pd.read_parquet(os.path.join(parquet_folder, all_pts_data_filename))
    descriptions = pd.read_csv(os.path.join(description_folder, model_file), encoding='latin1')
    print('Shape of model results:', descriptions.shape)
    print('Head of model results:')
    print(descriptions.head())

    true_descriptions = descriptions[descriptions['total_tf'].isin([True, 'TRUE'])]['description'].drop_duplicates()
    print('Descriptions to keep:', len(true_descriptions))
    all_pts_data_filtered = all_pts_data.merge(true_descriptions, on='description', how='inner')
    print('Shape of original data:', all_pts_data.shape)
    print('Shape of filtered data:', all_pts_data_filtered.shape)
    print('Head of filtered data:')
    print(all_pts_data_filtered.head())
    del all_pts_data
    gc.collect()

    filtered_pts_filename = 'filtered_data_' + model_file.replace('.csv', '.parquet').replace('description_mapping_', '')
    all_pts_data_filtered.to_parquet(os.path.join(parquet_folder, filtered_pts_filename))
    
    parquet_file_path = os.path.join(parquet_folder, 'phenotyping_processed_' + model_file.replace('.csv', '.parquet').replace('description_mapping_', ''))
    print(f'Parquet file path: {parquet_file_path}')
    if os.path.exists(parquet_file_path):
        processed_pts_df = pd.read_parquet(parquet_file_path)
        processed_patient_ids = set(processed_pts_df['patientunitstayid'].unique())
        print(f'Loaded {len(processed_patient_ids)} already processed pt records.')
    else:
        processed_pts_df = pd.DataFrame(columns=['patientunitstayid', 'outcome', 'processed_string'])
        processed_patient_ids = set()

    unique_pts = all_pts_data_filtered['patientunitstayid'].unique()
    remaining_pts = [pt for pt in unique_pts if pt not in processed_patient_ids]
    print(f'Loaded {len(remaining_pts)} unique pt records to process.')

    for i in range(0, len(remaining_pts), chunk_size):
        chunk_pts = remaining_pts[i:i + chunk_size]
        chunk_processed_pts = []

        for j, unique_pt in enumerate(chunk_pts):
            individual_pt_records = all_pts_data_filtered[all_pts_data_filtered['patientunitstayid'] == unique_pt]
            grouped_records = individual_pt_records.groupby('description').agg({'offset': 'min'}).reset_index().sort_values('offset').rename(columns={'offset': 'first_occurrence'})
            grouped_records['record_description'] = 'FIRST OCCURRENCE OFFSET: ' + grouped_records['first_occurrence'].astype(str) + '; DESCRIPTION: ' + grouped_records['description']
            grouped_records = grouped_records.drop(columns=['first_occurrence', 'description']).drop_duplicates()
            record_str = grouped_records.to_string(index=False, header=False, justify='left')
            record_str = '\n'.join([line.lstrip() for line in record_str.split('\n')])  # Ensure no leading spaces
            phenotype_outcome = individual_pt_records['outcome'].values[0]
            chunk_processed_pts.append({'patientunitstayid': unique_pt, 'outcome': phenotype_outcome, 'processed_string': record_str})

        chunk_processed_pts_df = pd.DataFrame(chunk_processed_pts)
        processed_pts_df = pd.concat([processed_pts_df, chunk_processed_pts_df], ignore_index=True).drop_duplicates()

        processed_pts_df.to_parquet(parquet_file_path)
        print(f'Saved chunk {i // chunk_size + 1} to {parquet_file_path}')
        if i == 0: print('Head of processed pts df:', processed_pts_df.head())

    print('Processing complete.')
    print('Head of processed pts df:')
    print(processed_pts_df.head())


parquet_folder = ''
all_pts_data_filename = 'combined_records_with_outcome.parquet'
description_folder = ''

model_files = ['description_mapping_phi414bq80.csv']
for model_file in model_files:
    print(f'Processing data for model file: {model_file}')
    process_patient_data(parquet_folder, all_pts_data_filename, description_folder, model_file)