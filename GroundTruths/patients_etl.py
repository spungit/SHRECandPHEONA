from sqlalchemy import create_engine
import json
import pandas as pd
import numpy as np 

import os
print(os.getcwd())

# database connection
with open('./eicu/etl/config.json','r') as f:
    config = json.load(f)
HOSTNAME = config['HOSTNAME']
DBNAME   = config['DBNAME']
PGUSER   = config['PGUSER']
PGPASS   = config['PGPASS']
PGPORT   = config['PGPORT']
PGSCHEMA = config['PGSCHEMA'] 
PGURI = 'postgresql://{}:{}@{}:{}/{}'.format(
        PGUSER, PGPASS, HOSTNAME, PGPORT, DBNAME)
c = create_engine(PGURI).connect()
# c.execute('set search_path to {};'.format(PGSCHEMA))

# find first admissions only to merge with invasive patients
patient = pd.read_sql('''
SELECT uniquepid, patientunitstayid
FROM eicu_crd.patient
''',c)

x = dict()
for index, row in patient.iterrows():
	if row['uniquepid'] in x:
		if row['patientunitstayid'] < x[row['uniquepid']]:
			x[row['uniquepid']] = row['patientunitstayid']
	else:
		x[row['uniquepid']] = row['patientunitstayid']
print("number of recorded patients: {}".format(len(x)))
first_admissions = pd.DataFrame(x.items(), 
	columns=['uniquepid','patientunitstayid'])
print(first_admissions.head(n=3))


'''invasive vent patients'''
# invasive pts from careplangeneral and item values distribution
cpl_dist = pd.read_sql('''
SELECT cplitemvalue, count(*) AS num_obs
FROM eicu_crd.careplangeneral
WHERE cplitemvalue LIKE '%%Intubated/oral ETT%%'
OR cplitemvalue LIKE '%%Intubated%%'
GROUP BY cplitemvalue
ORDER BY num_obs DESC
''',c)
print(cpl_dist)

cpl = pd.read_sql('''
SELECT patientunitstayid, cplitemoffset
FROM eicu_crd.careplangeneral
WHERE cplitemvalue LIKE '%%Intubated/oral ETT%%'
OR cplitemvalue LIKE '%%Intubated%%'
''',c)
cpl.columns = ['patientunitstayid','invasive_starttime']
print(cpl.head(n=5))
print('careplangeneral: %.0f'%
	cpl.patientunitstayid.nunique())

# invasive patients from respiratory care airway type distribution
resp_care_dist = pd.read_sql('''
SELECT airwaytype, count(*) AS num_obs
FROM eicu_crd.respiratorycare
WHERE airwaytype LIKE '%%Oral ETT%%'
OR airwaytype LIKE '%%Tracheostomy%%'
OR airwaytype LIKE '%%Nasal ETT%%'
OR airwaytype LIKE '%%Double-Lumen Tube%%'
OR airwaytype LIKE '%%Cricothyrotomy%%'
GROUP BY airwaytype
ORDER BY num_obs DESC
''',c)
print(resp_care_dist)

resp_care = pd.read_sql('''
SELECT patientunitstayid, respcarestatusoffset
FROM eicu_crd.respiratorycare
WHERE airwaytype LIKE '%%Oral ETT%%'
OR airwaytype LIKE '%%Tracheostomy%%'
OR airwaytype LIKE '%%Nasal ETT%%'
OR airwaytype LIKE '%%Double-Lumen Tube%%'
OR airwaytype LIKE '%%Cricothyrotomy%%'
''',c)
resp_care.columns = ['patientunitstayid','invasive_starttime']
print(resp_care.head(n=3))
print('respiratory care: %.0f'%
	resp_care.patientunitstayid.nunique())

# invasive patients from nurse care and distribution of values
nurse_care_dist = pd.read_sql('''
SELECT cellattributevalue, count(*) AS num_obs
FROM eicu_crd.nursecare
WHERE cellattributevalue LIKE '%%Oral ETT%%'
OR cellattributevalue LIKE '%%Tracheostomy%%'
OR cellattributevalue LIKE '%%Nasal ETT%%'
OR cellattributevalue LIKE '%%Cricothyrotomy%%'
OR cellattributevalue LIKE '%%Laryngectomy%%'
OR cellattributevalue LIKE '%%Double-Lumen Tube%%'
GROUP BY cellattributevalue
ORDER BY num_obs DESC
''',c)
print(nurse_care_dist)

nurse_care = pd.read_sql('''
SELECT patientunitstayid, nursecareoffset
FROM eicu_crd.nursecare
WHERE cellattributevalue LIKE '%%Oral ETT%%'
OR cellattributevalue LIKE '%%Tracheostomy%%'
OR cellattributevalue LIKE '%%Nasal ETT%%'
OR cellattributevalue LIKE '%%Cricothyrotomy%%'
OR cellattributevalue LIKE '%%Laryngectomy%%'
OR cellattributevalue LIKE '%%Double-Lumen Tube%%'
''',c)
nurse_care.columns = ['patientunitstayid','invasive_starttime']
print(nurse_care.head(n=3))
print('nurse care: %.0f'%
	nurse_care.patientunitstayid.nunique())

# invasive patients from treatment and distribution of values
treatment_dist = pd.read_sql('''
SELECT treatmentstring, count(*) AS num_obs
FROM eicu_crd.treatment
WHERE treatmentstring LIKE '%%endotracheal tube%%'
GROUP BY treatmentstring
ORDER BY num_obs DESC
''',c)
print(treatment_dist)

treatment = pd.read_sql('''
SELECT patientunitstayid, treatmentoffset
FROM eicu_crd.treatment
WHERE treatmentstring LIKE '%%endotracheal tube%%'
''',c)
treatment.columns = ['patientunitstayid','invasive_starttime']
print(treatment.head(n=3))
print('treatment: %.0f'%
	treatment.patientunitstayid.nunique())

# combine all invasive stays
dfs = [resp_care, nurse_care, treatment, cpl]
invasive = pd.concat(dfs, sort=True)
invasive = invasive.merge(first_admissions, 
	on='patientunitstayid', how='inner')

# separate invasive starttime into start and end times
invasive_starttime = invasive.sort_values(
	['patientunitstayid','invasive_starttime']).drop_duplicates(
	subset='patientunitstayid', keep='first')
invasive_starttime.columns = ['invasive_starttime',
	'patientunitstayid','uniquepid']

invasive_endtime = invasive.sort_values(
	['patientunitstayid','invasive_starttime']).drop_duplicates(
	subset='patientunitstayid', keep='last')
invasive_endtime.columns = ['invasive_endtime',
	'patientunitstayid','uniquepid']

invasive = pd.merge(invasive_starttime,invasive_endtime,
	on=['patientunitstayid','uniquepid'],how='inner')

print(invasive.head(n=3))
print(invasive.shape)
print('invasive stays: %.0f'%
	invasive.patientunitstayid.nunique())
print('invasive patients: %.0f'%
	invasive.uniquepid.nunique())

# intubation meds for validation that patients were intubated
infusion_drug = pd.read_sql('''
SELECT patientunitstayid, drugname, infusionoffset
FROM eicu_crd.infusiondrug
WHERE drugname LIKE '%%Propofol%%'
OR drugname LIKE '%%PROPOFOL%%'
OR drugname LIKE '%%Etomidate%%'
OR drugname LIKE '%%ETOMIDATE%%'
OR drugname LIKE '%%Versed%%'
OR drugname LIKE '%%VERSED%%'
OR drugname LIKE '%%Midazolam%%'
OR drugname LIKE '%%MIDAZOLAM%%'
OR drugname LIKE '%%Thiopental%%'
OR drugname LIKE '%%THIOPENTAL%%'
OR drugname LIKE '%%Ketamine%%'
OR drugname LIKE '%%KETAMINE%%'
OR drugname LIKE '%%Succinylcholine%%'
OR drugname LIKE '%%SUCCINYLCHOLINE%%'
OR drugname LIKE '%%Rocuronium%%'
OR drugname LIKE '%%ROCURONIUM%%'
OR drugname LIKE '%%Vecuronium%%'
OR drugname LIKE '%%VERCURONIUM%%'
OR drugname LIKE '%%Fentanyl%%'
OR drugname LIKE '%%FENTANYL%%'
OR drugname LIKE '%%Dexmedetomidine%%'
OR drugname LIKE '%%DEXMEDETOMIDINE%%'
OR drugname LIKE '%%Precedex%%'
OR drugname LIKE '%%PRECEDEX%%'
''',c)
infusion_drug.columns = ['patientunitstayid','drugname',
	'meds_starttime']
print(infusion_drug.head(n=3))
print('infusion table unit stays: %.0f'%
	infusion_drug.patientunitstayid.nunique())

medications = pd.read_sql('''
SELECT patientunitstayid, drugname, drugorderoffset
FROM eicu_crd.medication
WHERE drugname LIKE '%%Propofol%%'
OR drugname LIKE '%%PROPOFOL%%'
OR drugname LIKE '%%Etomidate%%'
OR drugname LIKE '%%ETOMIDATE%%'
OR drugname LIKE '%%Versed%%'
OR drugname LIKE '%%VERSED%%'
OR drugname LIKE '%%Midazolam%%'
OR drugname LIKE '%%MIDAZOLAM%%'
OR drugname LIKE '%%Thiopental%%'
OR drugname LIKE '%%THIOPENTAL%%'
OR drugname LIKE '%%Ketamine%%'
OR drugname LIKE '%%KETAMINE%%'
OR drugname LIKE '%%Succinylcholine%%'
OR drugname LIKE '%%SUCCINYLCHOLINE%%'
OR drugname LIKE '%%Rocuronium%%'
OR drugname LIKE '%%ROCURONIUM%%'
OR drugname LIKE '%%Vecuronium%%'
OR drugname LIKE '%%VERCURONIUM%%'
OR drugname LIKE '%%Fentanyl%%'
OR drugname LIKE '%%FENTANYL%%'
OR drugname LIKE '%%Dexmedetomidine%%'
OR drugname LIKE '%%DEXMEDETOMIDINE%%'
OR drugname LIKE '%%Precedex%%'
OR drugname LIKE '%%PRECEDEX%%'
''',c)
medications.columns = ['patientunitstayid','drugname',
	'meds_starttime']
print(medications.head(n=3))
print('med table unit stays: %.0f'%
	medications.patientunitstayid.nunique())

drug_dfs = [medications, infusion_drug]
intubation_meds = pd.concat(drug_dfs,sort=False)
print(intubation_meds.head(n=3))
print('total meds unit stays: %.0f'%
	intubation_meds.patientunitstayid.nunique())

# combine intubation meds with invasive patients
invasive = invasive.merge(intubation_meds, 
	on='patientunitstayid', how='inner')
print(invasive.head(n=3))

# sum number of records for each unit stay
record_count = invasive.groupby('patientunitstayid').size()
record_count = pd.DataFrame(record_count)
record_count.columns = ['record_count']
invasive = invasive.merge(record_count,
	on='patientunitstayid',how='inner')
# print(invasive.record_count.value_counts())
# filter for number of records
invasive = invasive.loc[invasive['record_count'] > 1]
print(invasive.shape)

# keep latest/earliest invasive start time
invasive = invasive.sort_values(
	['patientunitstayid']).drop_duplicates(
	subset='patientunitstayid')

print(invasive.head(n=3))
print(invasive.shape)
print('total invasive stays: %.0f'%
	invasive.patientunitstayid.nunique())
print('total invasive patients: %.0f'%
	invasive.uniquepid.nunique())

invasive.to_csv(
	'./eicu/data/patients/invasive_patients.csv', index=False)


'''noninvasive vent patients'''
# noninvasive patients from careplangeneral
cpl_dist = pd.read_sql('''
SELECT cplitemvalue, count(*) AS num_obs
FROM eicu_crd.careplangeneral
WHERE cplitemvalue LIKE '%%Non-invasive%%'
GROUP BY cplitemvalue
ORDER BY num_obs DESC
''',c)
print(cpl_dist)

cpl = pd.read_sql('''
SELECT patientunitstayid, cplitemoffset
FROM eicu_crd.careplangeneral
WHERE cplitemvalue LIKE '%%Non-invasive%%'
''',c)
cpl.columns = ['patientunitstayid','niv_endtime']
print(cpl.head(n=3))
print('niv cpl stays: %.0f'%
	cpl.patientunitstayid.nunique())

#  noninvasive patients from treatment table
treatment_dist = pd.read_sql('''
SELECT treatmentstring, count(*) AS num_obs
FROM eicu_crd.treatment
WHERE treatmentstring LIKE '%%non-invasive ventilation%%'
OR treatmentstring LIKE '%%CPAP%%'
GROUP BY treatmentstring
ORDER BY num_obs DESC
''',c)
print(treatment_dist)

treatment = pd.read_sql('''
SELECT patientunitstayid, treatmentoffset
FROM eicu_crd.treatment
WHERE treatmentstring LIKE '%%non-invasive ventilation%%'
OR treatmentstring LIKE '%%CPAP%%'
''',c)
treatment.columns = ['patientunitstayid','niv_endtime']
print(treatment.head(n=3))
print('niv treatment stays: %.0f'%
	treatment.patientunitstayid.nunique())

# niv = pd.merge(cpl, treatment, on='patientunitstayid',how='outer')
niv_dfs = [cpl,treatment]
niv = pd.concat(niv_dfs,sort=True)
niv = niv.merge(first_admissions, 
	on='patientunitstayid',how='inner')

# sum number of records for each unit stay
record_count = niv.groupby('patientunitstayid').size()
record_count = pd.DataFrame(record_count)
record_count.columns = ['record_count']
niv = niv.merge(record_count,on='patientunitstayid',how='inner')
# print(niv.record_count.value_counts())
# filter niv df to keep only stays that have at least a certain number of records
niv = niv.loc[niv['record_count'] > 1]
print(niv.shape)

# dropping duplicates and separating time stamps
niv_starttime = niv.sort_values(
	['patientunitstayid','niv_endtime']).drop_duplicates(
	subset='patientunitstayid', keep='first')
niv_starttime.columns = ['niv_starttime',
	'patientunitstayid','uniquepid','record_count']

niv_endtime = niv.sort_values(
	['patientunitstayid','niv_endtime']).drop_duplicates(
	subset='patientunitstayid', keep='last')
niv_endtime.columns = ['niv_endtime',
	'patientunitstayid','uniquepid','record_count']

niv = pd.merge(niv_starttime,niv_endtime,
	on=['patientunitstayid','uniquepid','record_count'],how='inner')

print(niv.head(n=3))
print(niv.shape)
print('total noninvasive stays: %.0f'%
	niv.patientunitstayid.nunique())
print('total noninvasive patients: %.0f'%
	niv.uniquepid.nunique())

niv.to_csv(
	'./eicu/data/patients/noninvasive_patients.csv',index=False)


'''hfnc patients'''
# nurse chart values, filter treatment table for 'non-invasive' 
# to find hfnc patients
nurse_chart = pd.read_sql('''
SELECT patientunitstayid, nursingchartvalue, nursingchartoffset
FROM eicu_crd.nursecharting
WHERE nursingchartcelltypevalname LIKE '%%O2 Admin Device%%'
''',c)
print('nurse_chart pts: %.0f'%
	nurse_chart.patientunitstayid.nunique())

hfnc = niv.merge(nurse_chart,on='patientunitstayid',how='inner')

# nurse_chart_values = hfnc.nursingchartvalue.value_counts().reset_index()
# nurse_chart_values.to_csv('./eicu/data/hfnc_nurse_chart_values.csv',index=False)

# filter hfnc by nurse_chart value list selected from value_counts above
hfnc = hfnc[hfnc['nursingchartvalue'].isin(['nasal cannula',
	'HFNC','NC','nc','high flow','hfnc','HiFlow','HFNC','HNC',
	'other-oximizer','hi flow','NRBM, HFNC','Nasal cannula',
	'High Flow NC','other,vapotherm','HHNF','oximizer','HiFlow NC',
	'High Flow O2','Nasal Canula','Hiflow','optiflow','HHFNC',
	'Hi Flow','HI FLOW N/C','hhfnc','HHF','hi-flo','hi-flow',
	'HI Flow','NC'])]

# dropping duplicates and separating time stamps
hfnc = hfnc.sort_values(
	['patientunitstayid','niv_endtime']).drop_duplicates(
	subset='patientunitstayid', keep='first')
hfnc.columns = ['hfnc_starttime','patientunitstayid',
	'uniquepid','record_count','hfnc_endtime',
	'nursingchartvalue','nursingchartoffset']
print(hfnc.head(n=3))
print(hfnc.shape)
print('hfnc stays: %.0f'%
	hfnc.patientunitstayid.nunique())
print('hfnc patients: %.0f'%
	hfnc.uniquepid.nunique())

hfnc.to_csv(
	'./eicu/data/patients/hfnc_patients.csv',index=False)


'''create success and failure cohorts'''
# niv, invasive, and hfnc patients and print patients in each
print('dataframe stats...')
niv = niv.drop(
	['record_count'],axis=1)
print('niv df patients: %.0f'%
	niv.uniquepid.nunique())

invasive = invasive.drop(
	['record_count','drugname','meds_starttime'],axis=1)
print('invasive df patients: %.0f'%
	invasive.uniquepid.nunique())

hfnc = hfnc.drop(
	['record_count','nursingchartvalue','nursingchartoffset'],axis=1)
print('hfnc df patients: %.0f'%
	hfnc.uniquepid.nunique())
'''
separate three DISTINCT invasive only, niv only, and hfnc only
'''
print('invasive and noninvasive separation...')
uncommon_niv_hfnc = niv.merge(
	invasive,
	on=['patientunitstayid','uniquepid'],
	how='outer',indicator=True)
inv_only = uncommon_niv_hfnc[uncommon_niv_hfnc['_merge']=='right_only']
print('invasive only patients: %.0f'%
	inv_only.uniquepid.nunique())
inv_only = inv_only.drop(
	['_merge','niv_starttime','niv_endtime'],axis=1)
inv_only['outcome'] = 0

'''niv and hfnc patients that are not included in the invasive df
separate niv + hfnc to just niv to find niv_only must do this 
because hfnc is a record subset of niv in the eicu database'''
niv_hfnc_only = uncommon_niv_hfnc[uncommon_niv_hfnc['_merge']=='left_only']
print('niv+hfnc patients: %.0f'%
	niv_hfnc_only.uniquepid.nunique())
niv_hfnc_only = niv_hfnc_only.drop(['_merge'],axis=1)

uncommon_niv = niv_hfnc_only.merge(hfnc,
	on=['uniquepid','patientunitstayid'],how='outer',indicator=True)

niv_only = uncommon_niv[uncommon_niv['_merge']=='left_only']
niv_only = niv_only.drop(
	['_merge','invasive_starttime','invasive_endtime',
	'hfnc_starttime','hfnc_endtime'],axis=1)
niv_only['outcome'] = 1
print('niv only: %.0f'%
	niv_only.uniquepid.nunique())

# hfnc is a subset of niv patients
uncommon_hfnc = hfnc.merge(
	invasive,on=['patientunitstayid','uniquepid'],
	how='outer',indicator=True)
hfnc_only = uncommon_hfnc[uncommon_hfnc['_merge']=='left_only']
hfnc_only = hfnc_only.drop(
	['_merge','invasive_starttime','invasive_endtime'],axis=1)
hfnc_only['outcome'] = 2
print('hfnc only patints: %.0f'%
	hfnc_only.uniquepid.nunique())

hfnc_both = uncommon_hfnc[uncommon_hfnc['_merge']=='both']
print('both hfnc and inv records patients: %.0f'%
	hfnc_both.uniquepid.nunique())
# both = uncommon_niv_hfnc[uncommon_niv_hfnc['_merge'] == 'both']
# print('both niv+hfnc and inv records patients: %.0f' % both.uniquepid.nunique())

# difference between timestamps to find who failed niv
print('niv failure cohorts...')
inv_and_niv = pd.merge(
	niv,invasive,
	on=['patientunitstayid','uniquepid'],
	how='inner')
# can alter time diff eqn to change how failure is calculated
inv_and_niv['time_difference'] = inv_and_niv[
	'invasive_endtime']-inv_and_niv['niv_starttime']

niv_hfnc_failure = inv_and_niv.loc[inv_and_niv[
	'time_difference']>(-1)]
print('niv+hfnc failure patients: %.0f'%
	niv_hfnc_failure.uniquepid.nunique())

inv_to_niv_hfnc = inv_and_niv.loc[inv_and_niv[
	'time_difference']<0]
print('invasive to niv+hfnc patients: %.0f'%
	inv_to_niv_hfnc.uniquepid.nunique())
hfnc_both = hfnc_both.drop(
	['_merge','invasive_starttime','invasive_endtime'],axis=1)
uncommon_niv_failure = niv_hfnc_failure.merge(
	hfnc_both,
	on=['patientunitstayid','uniquepid'],
	how='outer',indicator=True)
niv_only_failure = uncommon_niv_failure[uncommon_niv_failure[
	'_merge']=='left_only']
niv_only_failure = niv_only_failure.drop(
	['_merge','hfnc_starttime','hfnc_endtime'],axis=1)
niv_only_failure['outcome'] = 3
print('niv only failure: %.0f'%
	niv_only_failure.uniquepid.nunique())

uncommon_inv_to_niv = inv_to_niv_hfnc.merge(
	hfnc_both,
	on=['patientunitstayid','uniquepid'],
	how='outer',indicator=True)
inv_to_niv_only = uncommon_inv_to_niv[uncommon_inv_to_niv[
	'_merge']=='left_only']
inv_to_niv_only = inv_to_niv_only.drop(
	['_merge','hfnc_starttime','hfnc_endtime'],axis=1)
inv_to_niv_only['outcome'] = 4
print('invasive to niv: %.0f' % inv_to_niv_only.uniquepid.nunique())

# difference between timestamps to find who failed hfnc
print('hfnc failure cohorts...')
inv_and_hfnc = pd.merge(
	hfnc,invasive,
	on=['patientunitstayid','uniquepid'],
	how='inner')
# can alter time diff eqn to change how failure is calculated
inv_and_hfnc['time_difference'] = inv_and_hfnc[
	'invasive_endtime']-inv_and_hfnc['hfnc_starttime']

hfnc_failure = inv_and_hfnc.loc[inv_and_hfnc[
	'time_difference']>(-1)].reset_index()
hfnc_failure['outcome'] = 5
print('hfnc failure patients: %.0f'%
	hfnc_failure.uniquepid.nunique())

inv_to_hfnc = inv_and_hfnc.loc[inv_and_hfnc[
	'time_difference']<0].reset_index()
inv_to_hfnc['outcome'] = 6
print('invasive to hfnc patients: %.0f'%
	inv_to_hfnc.uniquepid.nunique())

# print heads
print(inv_only.head(n=3))
print(niv_only.head(n=3))
print(hfnc_only.head(n=3))
print(niv_only_failure.head(n=3))
print(inv_to_niv_only.head(n=3))
print(hfnc_failure.head(n=3))
print(inv_to_hfnc.head(n=3))

# final concatenation 
ventilation_pt_dfs = [inv_only,niv_only,hfnc_only,niv_only_failure,
	inv_to_niv_only,hfnc_failure,inv_to_hfnc]
ventilation_pts = pd.concat(ventilation_pt_dfs,sort=False)
ventilation_pts = ventilation_pts.drop(['index'],axis=1)
print(ventilation_pts.head(n=3))
print('total respoiratory failure patients: %.0f'%
	ventilation_pts.uniquepid.nunique())

ventilation_pts.to_csv(
	'./eicu/data/patients/ventilation_patients_total.csv',index=False)
