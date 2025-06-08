import pandas as pd
import numpy as np
import dask.dataframe as dd
import sys
import os

sys.stdout = open(os.path.join(os.getcwd(), './eICU/extract_descriptions.txt'), 'w')
print('Output will be written to extract_descriptions.txt')

combined_output_filename = 'combined_records_with_outcome.parquet'
description_output_filename = 'all_combined/descriptions.csv'

def read_parquet_file(file_path):
    return pd.read_parquet(file_path)

def get_data():
    ddir = "./pqt/"
    file_paths = {
        "care_plan_general": ddir + "carePlanGeneral.parquet",
        "respiratory_care": ddir + "respiratoryCare.parquet",
        'respiratory_charting': ddir + 'respiratoryCharting.parquet',
        "nurse_care": ddir + "nurseCare.parquet",
        "treatment": ddir + "treatment.parquet",
        "infusion_drug": ddir + "infusionDrug.parquet",
        "medication": ddir + "medication.parquet",
        "nurse_charting": ddir + "nurseCharting.parquet",
        "note": ddir + "note.parquet",
    }

    def process_care_plan_general():
        df = read_parquet_file(file_paths['care_plan_general'])
        df['patientunitstayid'] = df['patientunitstayid'].astype(str)
        df = df[['patientunitstayid', 'cplitemoffset', 'cplgroup', 'cplitemvalue']]
        df.loc[:, 'description'] = 'Source = Care Plan General; Concept = ' + df['cplgroup'] + ": " + df['cplitemvalue']
        df = df[['patientunitstayid', 'cplitemoffset', 'description']].dropna().drop_duplicates().rename(columns={'cplitemoffset': 'offset'})
        df['intubated'] = np.where(df['description'].str.contains(r'Intubated|Intubated/oral ETT', case=True, regex=True), 1, 0)
        df['non_invasive'] = np.where(df['description'].str.contains(r'Non-invasive', case=True, regex=True), 1, 0)
        return df

    def process_respiratory_care():
        df = read_parquet_file(file_paths['respiratory_care'])
        df['patientunitstayid'] = df['patientunitstayid'].astype(str)
        df = df[['patientunitstayid', 'respcarestatusoffset', 'airwaytype']]
        df.loc[:, 'description'] = 'Source = Respiratory Care; Concept = ' + df['airwaytype']
        df = df[['patientunitstayid', 'respcarestatusoffset', 'description']].dropna().drop_duplicates().rename(columns={'respcarestatusoffset': 'offset'})
        df['intubated'] = np.where(df['description'].str.contains(r'Oral ETT|Tracheostomy|Nasal ETT|Double-Lumen Tube|Cricothyrotomy', case=True, regex=True), 1, 0)
        return df
    
    def process_respiratory_charting():
        df = read_parquet_file(file_paths['respiratory_charting'])
        df['patientunitstayid'] = df['patientunitstayid'].astype(str)
        df = df[['patientunitstayid', 'respchartoffset', 'respcharttypecat', 'respchartvaluelabel', 'respchartvalue']]
        df.loc[:, 'description'] = 'Source = Respiratory Charting; Concept = ' + df['respcharttypecat'] + ": " + df['respchartvaluelabel'] + ": " + df['respchartvalue']
        df = df[['patientunitstayid', 'respchartoffset', 'description']].dropna().drop_duplicates().rename(columns={'respchartoffset': 'offset'})
        return df

    def process_nurse_care():
        df = read_parquet_file(file_paths['nurse_care'])
        df['patientunitstayid'] = df['patientunitstayid'].astype(str)
        df = df[['patientunitstayid', 'nursecareoffset', 'cellattributevalue']]
        df.loc[:, 'description'] = 'Source = Nurse Care; Concept = ' + df['cellattributevalue']
        df = df[['patientunitstayid', 'nursecareoffset', 'description']].dropna().drop_duplicates().rename(columns={'nursecareoffset': 'offset'})
        df['intubated'] = np.where(df['description'].str.contains(r'Oral ETT|Tracheostomy|Nasal ETT|Double-Lumen Tube|Cricothyrotomy|Laryngectomy', case=True, regex=True), 1, 0)
        return df

    def process_treatment():
        df = read_parquet_file(file_paths['treatment'])
        df['patientunitstayid'] = df['patientunitstayid'].astype(str)
        df = df[['patientunitstayid', 'treatmentoffset', 'treatmentstring']]
        df.loc[:, 'description'] = 'Source = Treatment; Concept = ' + df['treatmentstring']
        df = df[['patientunitstayid', 'treatmentoffset', 'description']].dropna().drop_duplicates().rename(columns={'treatmentoffset': 'offset'})
        df['intubated'] = np.where(df['description'].str.contains(r'endotracheal tube', case=True, regex=True), 1, 0)
        df['non_invasive'] = np.where(df['description'].str.contains(r'non-invasive ventilation|CPAP', case=True, regex=True), 1, 0)
        return df

    def process_infusion_drug():
        df = read_parquet_file(file_paths['infusion_drug'])
        df['patientunitstayid'] = df['patientunitstayid'].astype(str)
        df = df[['patientunitstayid', 'infusionoffset', 'drugname']]
        df['description'] = 'Source = Infusion Drug; Concept = ' + df['drugname']
        df = df[['patientunitstayid', 'infusionoffset', 'description']].dropna().drop_duplicates().rename(columns={'infusionoffset': 'offset'})
        df['intubated'] = np.where(df['description'].str.contains(r'Propofol|PROPOFOL|Etomidate|ETOMIDATE|Versed|VERSED|Midazolam|MIDAZOLAM|Thiopental|THIOPENTAL|Ketamine|KETAMINE|Succinylcholine|SUCCINYLCHOLINE|Rocuronium|ROCURONIUM|Vecuronium|VERCURONIUM|Fentanyl|FENTANYL|Dexmedetomidine|DEXMEDETOMIDINE|Precedex|PRECEDEX', case=True, regex=True), 1, 0)
        df['meds'] = np.where(df['description'].str.contains(r'Propofol|PROPOFOL|Etomidate|ETOMIDATE|Versed|VERSED|Midazolam|MIDAZOLAM|Thiopental|THIOPENTAL|Ketamine|KETAMINE|Succinylcholine|SUCCINYLCHOLINE|Rocuronium|ROCURONIUM|Vecuronium|VERCURONIUM|Fentanyl|FENTANYL|Dexmedetomidine|DEXMEDETOMIDINE|Precedex|PRECEDEX', case=True, regex=True), 1, 0)
        return df

    def process_medication():
        df = read_parquet_file(file_paths['medication'])
        df['patientunitstayid'] = df['patientunitstayid'].astype(str)
        df = df[['patientunitstayid', 'drugstartoffset', 'drugname']]
        df.loc[:, 'description'] = 'Source = Medication; Concept = ' + df['drugname']
        df = df[['patientunitstayid', 'drugstartoffset', 'description']].dropna().drop_duplicates().rename(columns={'drugstartoffset': 'offset'})
        df['intubated'] = np.where(df['description'].str.contains(r'Propofol|PROPOFOL|Etomidate|ETOMIDATE|Versed|VERSED|Midazolam|MIDAZOLAM|Thiopental|THIOPENTAL|Ketamine|KETAMINE|Succinylcholine|SUCCINYLCHOLINE|Rocuronium|ROCURONIUM|Vecuronium|VERCURONIUM|Fentanyl|FENTANYL|Dexmedetomidine|DEXMEDETOMIDINE|Precedex|PRECEDEX', case=True, regex=True), 1, 0)
        df['meds'] = np.where(df['description'].str.contains(r'Propofol|PROPOFOL|Etomidate|ETOMIDATE|Versed|VERSED|Midazolam|MIDAZOLAM|Thiopental|THIOPENTAL|Ketamine|KETAMINE|Succinylcholine|SUCCINYLCHOLINE|Rocuronium|ROCURONIUM|Vecuronium|VERCURONIUM|Fentanyl|FENTANYL|Dexmedetomidine|DEXMEDETOMIDINE|Precedex|PRECEDEX', case=True, regex=True), 1, 0)
        return df
        
    def process_nurse_charting():
        ddf = dd.read_parquet(file_paths['nurse_charting'])
        
        def process_chunk(chunk):
            chunk['patientunitstayid'] = chunk['patientunitstayid'].astype(str)
            chunk = chunk[['patientunitstayid', 'nursingchartoffset', 'nursingchartcelltypevalname', 'nursingchartvalue']]

            chunk['nursingchartcelltypevalname'] = chunk['nursingchartcelltypevalname'].fillna('')
            chunk['nursingchartvalue'] = chunk['nursingchartvalue'].fillna('')

            chunk['hfni'] = np.where(
                (chunk['nursingchartcelltypevalname'].str.contains(r'O2 Admin Device', case=True, regex=True) & 
                chunk['nursingchartvalue'].str.contains(r'HFNC|NC|nc|high flow|hfnc|HiFlow|HFNC|HNC|other-oximizer|hi flow|NRBM, HFNC|Nasal cannula|High Flow NC|other,vapotherm|HHNF|oximizer|HiFlow NC|High Flow O2|Nasal Canula|Hiflow|optiflow|HHFNC|Hi Flow|HI FLOW N/C|hhfnc|HHF|hi-flo|hi-flow|HI Flow|NC', case=True, regex=True)), 
                1, 0
            )
            
            chunk.loc[:, 'description'] = 'Source = Nurse Charting; Concept = ' + chunk['nursingchartcelltypevalname'] + ": " + chunk['nursingchartvalue']
            chunk = chunk[['patientunitstayid', 'nursingchartoffset', 'description', 'hfni']].dropna().drop_duplicates().rename(columns={'nursingchartoffset': 'offset'})
            return chunk

        meta = pd.DataFrame({
            'patientunitstayid': pd.Series(dtype='str'),
            'offset': pd.Series(dtype='int'),
            'description': pd.Series(dtype='str'),
            'hfni': pd.Series(dtype='int')
        })

        processed_ddf = ddf.map_partitions(process_chunk, meta=meta)
        df = processed_ddf.compute()
        return df
    
    def process_note():
        df = read_parquet_file(file_paths['note'])
        df['patientunitstayid'] = df['patientunitstayid'].astype(str)
        df = df[['patientunitstayid', 'noteoffset', 'notevalue', 'notetext']]
        df['description'] = 'Source = Note; Concept = ' + df['notevalue'] + ": " + df['notetext']
        df = df[['patientunitstayid', 'noteoffset', 'description']].dropna().drop_duplicates().rename(columns={'noteoffset': 'offset'})
        return df

    careplangeneral_df = process_care_plan_general()
    print('\nCARE PLAN GENERAL')
    print('Number of rows:', len(careplangeneral_df))
    print('Unique Patient IDs:', careplangeneral_df['patientunitstayid'].nunique())
    print('Unique descriptions:', careplangeneral_df['description'].nunique())
    print(careplangeneral_df['description'].unique())
    print('Head of the dataframe:', careplangeneral_df.head())

    respiratorycare_df = process_respiratory_care()
    print('\nRESPIRATORY CARE')
    print('Number of rows:', len(respiratorycare_df))
    print('Unique Patient IDs:', respiratorycare_df['patientunitstayid'].nunique())
    print('Unique descriptions:', respiratorycare_df['description'].nunique())
    print(respiratorycare_df['description'].unique())
    print('Head of the dataframe:', respiratorycare_df.head())

    respiratory_charting_df = process_respiratory_charting()
    print('\nRESPIRATORY CHARTING')
    print('Number of rows:', len(respiratory_charting_df))
    print('Unique Patient IDs:', respiratory_charting_df['patientunitstayid'].nunique())
    print('Unique descriptions:', respiratory_charting_df['description'].nunique())
    print(respiratory_charting_df['description'].unique())
    print('Head of the dataframe:', respiratory_charting_df.head())

    nurse_care_df = process_nurse_care()
    print('\nNURSE CARE')
    print('Number of rows:', len(nurse_care_df))
    print('Unique Patient IDs:', nurse_care_df['patientunitstayid'].nunique())
    print('Unique descriptions:', nurse_care_df['description'].nunique())
    print(nurse_care_df['description'].unique())
    print('Head of the dataframe:', nurse_care_df.head())

    treatment_df = process_treatment()
    print('\nTREATMENT')
    print('Number of rows:', len(treatment_df))
    print('Unique Patient IDs:', treatment_df['patientunitstayid'].nunique())
    print('Unique descriptions:', treatment_df['description'].nunique())
    print(treatment_df['description'].unique())
    print('Head of the dataframe:', treatment_df.head())

    infusiondrug_df = process_infusion_drug()
    print('\nINFUSION DRUG')
    print('Number of rows:', len(infusiondrug_df))
    print('Unique Patient IDs:', infusiondrug_df['patientunitstayid'].nunique())
    print('Unique descriptions:', infusiondrug_df['description'].nunique())
    print(infusiondrug_df['description'].unique())
    print('Head of the dataframe:', infusiondrug_df.head())

    medication_df = process_medication()
    print('\nMEDICATION')
    print('Number of rows:', len(medication_df))
    print('Unique Patient IDs:', medication_df['patientunitstayid'].nunique())
    print('Unique descriptions:', medication_df['description'].nunique())
    print(medication_df['description'].unique())
    print('Head of the dataframe:', medication_df.head())

    nurse_charting_df = process_nurse_charting()
    print('\nNURSE CHARTING')
    print('Number of rows:', len(nurse_charting_df))
    print('Unique Patient IDs:', nurse_charting_df['patientunitstayid'].nunique())
    print('Unique descriptions:', nurse_charting_df['description'].nunique())
    print(nurse_care_df['description'].unique())
    print('Head of the dataframe:', nurse_charting_df.head())

    note_df = process_note()
    print('\nNOTE')
    print('Number of rows:', len(note_df))
    print('Unique Patient IDs:', note_df['patientunitstayid'].nunique())
    print('Unique descriptions:', note_df['description'].nunique())
    print(note_df['description'].unique())
    print('Head of the dataframe:', note_df.head())

    combined_df = pd.concat([careplangeneral_df, respiratorycare_df, respiratory_charting_df, nurse_care_df, treatment_df, infusiondrug_df, medication_df, nurse_charting_df, note_df], axis=0)
    combined_df = combined_df.sort_values(by=['patientunitstayid', 'offset']).reset_index(drop=True).drop_duplicates()
    combined_df[['intubated', 'non_invasive', 'hfni','meds']] = combined_df[['intubated', 'non_invasive', 'hfni','meds']].fillna(0)
    combined_df = combined_df.dropna(subset=['description'])

    combined_df['patientunitstayid'] = combined_df['patientunitstayid'].astype(str)
    n_records = combined_df.groupby('patientunitstayid').size().reset_index(name='records')
    combined_df = combined_df.merge(n_records, on='patientunitstayid', how='left')
    combined_df['ground_truth'] = np.where((combined_df['intubated'] == 1) | (combined_df['non_invasive'] == 1) | (combined_df['hfni'] == 1), 1, 0)

    print('\nCOMBINED')
    print('Number of rows:', len(combined_df))
    print('Number of unique descriptions:', combined_df['description'].nunique())
    print('Value counts of intubated:', combined_df['intubated'].value_counts())
    print('Value counts of non_invasive:', combined_df['non_invasive'].value_counts())
    print('Value counts of hfni:', combined_df['hfni'].value_counts())
    print('Value counts of ground_truth:', combined_df['ground_truth'].value_counts())
    print('Head of the dataframe:', combined_df.head())

    return combined_df

ground_truth = pd.read_csv('ventilation_patients_total.csv')
print('Head of the ground truth dataframe:', ground_truth.head())
ground_truth['patientunitstayid'] = ground_truth['patientunitstayid'].astype(str)
print('Number of ground truth patients:', len(ground_truth['patientunitstayid'].unique()))

combined_df = get_data()
combined_df = combined_df.merge(ground_truth[['patientunitstayid','outcome']], on='patientunitstayid', how='left')
combined_df['outcome'] = combined_df['outcome'].fillna(-1).astype(int)
combined_df = combined_df.drop_duplicates()
combined_df.to_parquet(combined_output_filename, index=False)
print('Number of rows:', len(combined_df))
print('Columns:', combined_df.columns)
print('Head of the dataframe:\n', combined_df.head())

description_df = combined_df[['description', 'intubated', 'non_invasive', 'hfni', 'ground_truth']].drop_duplicates()
description_df.to_csv(description_output_filename, index=False)

print('Data saved to combined_records_with_outcome.parquet')
sys.stdout.close()