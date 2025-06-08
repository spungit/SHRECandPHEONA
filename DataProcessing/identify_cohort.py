import pandas as pd

## TODO: Find the first ICU stay for each patient based on the unitdischargeoffset column
pts_data_folder = ''
pts_data_filename = 'patient.parquet'
pts_data = pd.read_parquet(pts_data_folder + pts_data_filename)

print('Shape of patients data:', pts_data.shape)
print(f'Columns in patients data: {pts_data.columns.tolist()}')
print(f'Unique patientunitstayid: {pts_data["patientunitstayid"].nunique()}')
print(f'Unique patienthealthsystemstayid: {pts_data["patienthealthsystemstayid"].nunique()}')
print('Head of patients data:')
print(pts_data.head())

duplicates = pts_data.groupby('patienthealthsystemstayid')['patientunitstayid'].nunique()
print('Number of patients with multiple ICU stays:', duplicates[duplicates > 1].count())

pts_data['unitdischargeoffset'] = pd.to_numeric(pts_data['unitdischargeoffset'], errors='coerce').dropna().astype('int64')
pts_data['rank'] = pts_data.groupby('patienthealthsystemstayid')['unitdischargeoffset'].rank(method='first', ascending=True)
pts_data = pts_data[pts_data['rank'] == 1][['patientunitstayid', 'patienthealthsystemstayid', 'age']]
print('Shape of patients data after ranking:', pts_data.shape)
print(f'Unique patientunitstayid: {pts_data["patientunitstayid"].nunique()}')
print(f'Unique patienthealthsystemstayid: {pts_data["patienthealthsystemstayid"].nunique()}')
print('Head of patients data after ranking:')
print(pts_data.head())

## TODO: Keep only the individuals who were 18 years or older at the time of their first ICU stay.
pts_data['age'] = pd.to_numeric(pts_data['age'], errors='coerce').dropna().astype('int64')
pts_data = pts_data[pts_data['age'] >= 18]
print('Shape of patients data after filtering by age:', pts_data.shape)
print(f'Unique patientunitstayid: {pts_data["patientunitstayid"].nunique()}')
print(f'Unique patienthealthsystemstayid: {pts_data["patienthealthsystemstayid"].nunique()}')
print('Head of patients data after filtering by age:')
print(pts_data.head())

## TODO: inner join with combined patients and outcomes data
all_pts_data_folder = ''
all_pts_data_filename = 'combined_records_with_outcome.parquet'
all_pts_data = pd.read_parquet(all_pts_data_folder + all_pts_data_filename)
all_pts_data['patientunitstayid'] = all_pts_data['patientunitstayid'].astype('int64')
pts_data['patientunitstayid'] = pts_data['patientunitstayid'].astype('int64')

pts_data = pd.merge(pts_data, all_pts_data, on='patientunitstayid', how='left')
pts_data['outcome'] = pts_data['outcome'].fillna(-1).astype('int64')
print('Shape of patients data after merging with combined data:', pts_data.shape)
print(f'Unique patientunitstayid: {pts_data["patientunitstayid"].nunique()}')
print(f'Unique patienthealthsystemstayid: {pts_data["patienthealthsystemstayid"].nunique()}')
print(f'Columns in patients data after merging with combined data: {pts_data.columns.tolist()}')
print('Head of patients data after merging with combined data:')
print(pts_data.head())

cohort_data = pts_data[['patientunitstayid','outcome']].drop_duplicates()
print('Shape of cohort data:', cohort_data.shape)
print(f'Columns in cohort data: {cohort_data.columns.tolist()}')
print(f'Value counts of outcome in cohort data:\n{cohort_data["outcome"].value_counts()}')
print('Head of cohort data:')
print(cohort_data.head())

records_data = pts_data[['patientunitstayid','offset','description','intubated','non_invasive','hfni','outcome']].drop_duplicates()
print('Shape of records data:', records_data.shape)
print(f'Columns in records data: {records_data.columns.tolist()}')
print('Head of records data:')
print(records_data.head())

cohort_data.to_parquet(all_pts_data_folder + 'pts_with_outcomes.parquet', index=False)
cohort_data.to_csv(all_pts_data_folder + 'pts_with_outcomes.csv', index=False)
records_data.to_parquet(all_pts_data_folder + 'combined_cohort_records_with_outcome.parquet', index=False)

sample_pts = records_data['patientunitstayid'].sample(n=10, random_state=42)
sample_records_data = records_data[records_data['patientunitstayid'].isin(sample_pts)]
sample_records_data.to_csv(all_pts_data_folder + 'sample_cohort_records.csv', index=False)

print('Cohort data and records data saved successfully.')
