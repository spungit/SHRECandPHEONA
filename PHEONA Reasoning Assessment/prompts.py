instructions = """
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
            2) At least TWO INDIVIDUAL records indicating the patient was on invasive mechanical ventilation (IMV) or intubated. **EXCLUDES** records defining ventilation settings. Invasive mechanical ventilation involves a tube in the trachea (either an endotracheal tube placed through the mouth, or rarely the nose, OR a surgically placed tracheostomy tube) connected to a ventilator, delivering mechanical ventilation. Records with the following terms or acronyms should be considered for IMV unless otherwise indicated: ventilator, ETT or ET (endotracheal tube, trach tube), tracheostomy/trach,  PS (pressure support), AC (assist control vent mode), CMV (continuous mandatory ventilation vent mode), SIMV (synchronized intermittent mandatory ventilation vent mode), PRVC (pressure regulated volume control vent mode), APRV or Bi-level (airway pressure release ventilation vent mode).

    - **Treatment 2: Non-Invasive Positive Pressure Ventilation (NIPPV)**
        - **INCLUSION CRITERIA**:
            1) At least TWO INDIVIDUAL records indicating the patient was on non-invasive positive pressure ventilation (NIPPV) **THAT DOES NOT INDICATE** high flow nasal insufflation/cannula or nasal cannula. Also **EXCLUDES** records defining ventilation settings. Non-invasive positive pressure ventilation involves ventilation via a facemask, where the clinician adjusts pressure and oxygen settings. Records with the following terms and acronyms should be considered NIPPV unless otherwise indicated: mask, mask ventilation, NIV (non-invasive ventilation), BiPAP (bilevel positive airway pressure), CPAP (continuous positive airway pressure), IPAP (inspiratory positive airway pressure), EPAP (expiratory positive airway pressure), AVAPS (average volume assured pressure support).

    - **Treatment 3: High-Flow Nasal Insufflation/Nasal Cannula (HFNI/HFNC) or Nasal Cannula**
        - **INCLUSION CRITERIA**:
            1) The criteria for NIPPV is met where the records are **INDEPENDENT** of any records indicating HFNI or nasal cannula.
            AND
            2) At least ONE INDIVIDUAL record indicating the patient was on high flow nasal insufflation/cannula or nasal cannula. HFNI/HFNC involves oxygen delivery through a nasal cannula at a flow rate above 15 L/min, with adjustments for oxygen concentration and flow rate. Records with the following terms and acronyms should be considered HFNI/HFNC unless otherwise indicated: nasal cannula (NC), high flow nasal cannula, high flow nasal oxygen, high flow nasal insufflation, high flow nasal therapy, high flow nasal oxygen therapy, high flow nasal oxygen delivery, high flow nasal oxygen therapy (HFNOT), Optiflow, Vapotherm, Airvo.

"""

examples_prompt = """

{examples}

"""

hint_prompt = """<<I think that the answer is {ground_truth}. If you use this information, please indicate this in your response.>>"""

no_cot_questions = """

OUTPUT:
<output>
Q1) Based on the input information, which category does the patient's records fall under? **ONLY** respond with **ONE** of the following: IMV ONLY, NIPPV ONLY, HFNI ONLY, NIPPV TO IMV, HFNI TO IMV, IMV TO NIPPV, IMV TO HFNI, or NONE (if no records or specific treatments were present).
A1)
</output>
"""

some_cot_questions = """

OUTPUT:
<output>
SUMMARY:
Q1) Summarize the input records in 3-5 sentences.
A1)

TREATMENT TYPES:
Q2) Describe which treatments are present based on the input records.
A2)

TREATMENT ORDERING:
Q3) What is the order of the treatments based on the input records? If NIPPV and HFNI are between IMV records, does removing them affect the classification? If so, how?
A3)

FINAL CLASSIFICATION:
Q4) Based on your answers to the previous questions (Q2-Q3), which category does the patient's records fall under? **ONLY** respond with **ONE** of the following: IMV ONLY, NIPPV ONLY, HFNI ONLY, NIPPV TO IMV, HFNI TO IMV, IMV TO NIPPV, IMV TO HFNI, or NONE (if no records or specific treatments were present).
A4)
</output>
"""

full_cot_questions = """

OUTPUT:
<output>
SUMMARY:
Q1) Summarize the input records in 3-5 sentences.
A1)

TREATMENT TYPES:
Q2) Are any of the required medications present? If so, are there at least TWO INDIVIDUAL records indicating the patient was on invasive mechanical ventilation (IMV) or intubated? Provide a judgment and 'YES' or 'NO' for whether the inclusion criteria for IMV is met.
A2)

Q3) Are there at least TWO INDEPENDENT records indicating the patient was on NIPPV that are ALSO INDEPENDENT of any records indicating HFNI or nasal cannula? Provide a judgment and 'YES' or 'NO' for whether the inclusion criteria for NIPPV is met.
A3)

Q4) Based on the records provided, was the criteria for NIPPV met first? If the criteria for NIPPV was not met, then the criteria for HFNI is also not met. If the criteria for NIPPV was met, is there at least ONE ADDITIONAL record indicating HFNI or nasal cannula? Provide a judgment and 'YES' or 'NO' for whether the inclusion criteria for HFNI is met.
A4)

TREATMENT ORDERING:
Q5) Based on the previous three questions (Q2-Q4), was there MORE THAN ONE treatment present? **REMEMBER**: If the criteria for HFNI is met, then ONLY HFNI applies, **NOT** NIPPV or HFNI and NIPPV. If so, list the treatments and skip to Q6. If not, skip to Q8.
A5)

Q6) What was the start and end record orders for each of the following: 1) IMV, 2) NIPPV (if applicable), and 3) HFNI (if applicable)? Provide a judgment.
A6)

Q7) Based on the start and end record orders, are the NIPPV or HFNI records independent of the IMV records? In other words, were ALL of the QUALIFYING RECORDS for NIPPV or HFNI completely BEFORE or AFTER the IMV records and NOT BETWEEN the IMV records? Provide a judgment. Remember, the QUALIFYING RECORDS for HFNI include the qualifying NIPPV records and an additional record indicating HFNI or nasal cannula so if HFNI was present, ALL THE QUALIFYING RECORDS must be independent of the IMV records.
A7)

FINAL CLASSIFICATION:
Q8) Based on your answers to the previous questions (Q2-Q7), which category does the patient's records fall under? **ONLY** respond with **ONE** of the following: IMV ONLY, NIPPV ONLY, HFNI ONLY, NIPPV TO IMV, HFNI TO IMV, IMV TO NIPPV, IMV TO HFNI, or NONE (if no records or specific treatments were present).
A8)
</output>
"""

random_order_inputs = {
    'mistral-small:24b-instruct-2501-q8_0': """
EXAMPLE 1:
1: 2 ML VIAL : MIDAZOLAM HCL 2 MG/2ML IJ SOLN
2: MORPHINE INJ
3: respFlowSettings: PEEP: 5
4: 1 ML  -  HYDROMORPHONE HCL 1 MG/ML IJ SOLN
5: 2 ML  -  FENTANYL CITRATE 0.05 MG/ML IJ SOLN
6: 100 ML NDC : DEXMEDETOMIDINE HCL IN NACL 400 MCG/100ML IV SOLN
7: Ventilation: Ventilated - rapid wean/extubation
8: Airway: Intubated/oral ETT
9: respFlowSettings: Pressure Support: 8
10: Value: Maximal assist
11: Airway: Not intubated/normal airway
12: Value: Moderate assist
13: Value: Minimal assist
CLASSIFICATION: IMV ONLY

EXAMPLE 2:
1: morphine
2: DILAUDID
3: Airway: Not intubated/normal airway
4: O2 Admin Device: nasal cannula
5: DILAUDID 1MG/ML
6: respFlowSettings: FiO2: 21
CLASSIFICATION: NONE

EXAMPLE 3:
1: 2 ML VIAL : FENTANYL CITRATE 0.05 MG/ML IJ SOLN
2: 1 ML CRTRDG-NDL : MORPHINE SULFATE 2 MG/ML IJ SOLN
3: respFlowCareData: O2 Device: High flow nasal cannula
4: Airway: Not intubated/normal airway
5: respFlowCareData: O2 Device: Nasal cannula
CLASSIFICATION: NONE
""",

    'gemma2:27b-instruct-q8_0': """
EXAMPLE 1:
1: Oral ETT
2: O2 L/%: 15
3: pulmonary|ventilation and oxygenation|oxygen therapy (> 60%)
4: Ventilation: Non-invasive ventilation
5: pulmonary|ventilation and oxygenation|non-invasive ventilation
6: Sedation Scale: RASS
7: respFlowSettings: LPM O2: 15
8: respFlowSettings: LPM O2: 50
9: respFlowSettings: TV/kg IBW: 0.0000
10: respFlowSettings: Vent Rate: 28
11: Ventilation: Ventilated - with daily extubation evaluation
12: Airway: Intubated/oral ETT
13: Ventilation: Ventilated - with no daily extubation trial
14: Sedation: Continuous infusion - with daily holiday
15: pulmonary|ventilation and oxygenation|mechanical ventilation
16: Propofol (ml/hr)
17: respFlowSettings: PEEP: 10
18: respFlowSettings: PEEP: 9
19: Analgesia: Parenteral - continuous
20: Sedation Goal: -1
21: Midazolam (ml/hr)
22: Fentanyl (ml/hr)
23: respFlowSettings: Pressure Control: 18
24: respFlowPtVentData: RR (patient): 20
25: pulmonary|ventilation and oxygenation|mechanical ventilation|pressure controlled
26: pulmonary|medications|sedative|midazolam
27: Sedation: Continuous infusion - no daily holiday
28: respFlowPtVentData: Exhaled MV: 9.7
29: respFlowPtVentData: Mean Airway Pressure: 18
30: respFlowPtVentData: Peak Insp. Pressure: 28
31: respFlowPtVentData: Exhaled TV (patient): 456
32: respFlowSettings: PEEP: 7
33: Dexmedetomidine (ml/hr)
34: respFlowSettings: Vent Rate: 16
35: respFlowPtVentData: RR (patient): 21
36: pulmonary|radiologic procedures / bronchoscopy|reintubation
37: O2 L/%: 3
38: respFlowSettings: LPM O2: 3
39: Sedation: Oral
40: pulmonary|radiologic procedures / bronchoscopy|endotracheal tube removal
CLASSIFICATION: NIPPV TO IMV

EXAMPLE 2:
1: respFlowSettings: LPM O2: 3
2: Care Limitation: No intubation
3: Sedation Scale: RASS
4: respFlowPtVentData: Exhaled TV (patient): 473
5: respFlowSettings: TV/kg IBW: 8.1319
6: respFlowSettings: Tidal Volume (set): 370
7: respFlowSettings: Vent Rate: 12
8: pulmonary|radiologic procedures / bronchoscopy|endotracheal tube|insertion
9: pulmonary|ventilation and oxygenation|mechanical ventilation
10: Airway: Intubated/oral ETT
11: Ventilation: Ventilated - with daily extubation evaluation
12: respFlowSettings: Vent Rate: 10
13: respFlowPtVentData: Exhaled TV (patient): 403
14: respFlowPtVentData: Exhaled TV (patient): 383
15: respFlowSettings: LPM O2: 40
16: respFlowPtVentData: Exhaled TV (patient): 411
17: respFlowPtVentData: Exhaled TV (patient): 401
18: respFlowSettings: Pressure Support: 5
19: respFlowPtVentData: Exhaled TV (patient): 221
CLASSIFICATION: NONE

EXAMPLE 3:
1: MORPHINE 4 MG/1 ML 1 ML SYR
2: MORPHINE 2 MG/1 ML 1 ML SYR
3: HYDROmorphone 1 MG/1 ML SYR
4: fentaNYL (PF) 50 MCG/1 ML 2 ML INJ
5: Propofol (ml/hr)
6: respFlowPtVentData: Exhaled TV (patient): 100
7: respFlowPtVentData: RR (patient): 17
8: respFlowPtVentData: Peak Insp. Pressure: 20
9: respFlowSettings: TV/kg IBW: 7.0776
10: respFlowPtVentData: Plateau Pressure: 17
11: respFlowSettings: Vent Rate: 16
12: respFlowSettings: Tidal Volume (set): 500
13: Airway: Intubated/oral ETT
14: Ventilation: Ventilated - with daily extubation evaluation
15: pulmonary|ventilation and oxygenation|mechanical ventilation
16: PROPOFOL 10 MG/1 ML 100ML SDV INJ
17: respFlowSettings: Vent Rate: 20
18: respFlowPtVentData: RR (patient): 21
19: respFlowPtVentData: Plateau Pressure: 15
20: respFlowPtVentData: Peak Insp. Pressure: 18
21: respFlowPtVentData: Exhaled TV (patient): 600
22: respFlowPtVentData: RR (patient): 20
23: respFlowPtVentData: Plateau Pressure: 16
24: respFlowPtVentData: Peak Insp. Pressure: 19
25: respFlowPtVentData: Plateau Pressure: 14
26: respFlowPtVentData: Mean Airway Pressure: 10
27: fentaNYL 1000 MCG IN 100 ML NS
28: Fentanyl (ml/hr)
29: respFlowSettings: Vent Rate: 18
30: respFlowPtVentData: Exhaled TV (patient): 350
31: respFlowPtVentData: Peak Insp. Pressure: 17
32: respFlowPtVentData: Mean Airway Pressure: 8
33: respFlowSettings: Pressure Support: 10
CLASSIFICATION: IMV ONLY
""",

    'phi4:14b-q8_0': """
EXAMPLE 1:
1: HYDROmorphone 1 MG/1 ML SYR
2: fentaNYL (PF) 50 MCG/1 ML 2 ML INJ
3: respFlowSettings: LPM O2: 15
4: respFlowPtVentData: RR (patient): 13
5: respFlowPtVentData: Peak Insp. Pressure: 12
6: pulmonary|ventilation and oxygenation|non-invasive ventilation
7: Ventilation: Non-invasive ventilation
8: respFlowSettings: Vent Rate: 20
9: respFlowPtVentData: RR (patient): 27
10: Morphine (ml/hr)
CLASSIFICATION: NIPPV ONLY

EXAMPLE 2:
1: MORPHINE 4 MG/ML SYRINGE : 1 ML SYRINGE
2: HYDROmorphone
3: Midazolam
4: DILAUDID
CLASSIFICATION: NONE

EXAMPLE 3:
1: Ventilation: Non-invasive ventilation
2: pulmonary|ventilation and oxygenation|non-invasive ventilation
3: pulmonary|ventilation and oxygenation|non-invasive ventilation|face mask
4: respFlowPtVentData: Mean Airway Pressure: 12
5: respFlowSettings: Pressure Support: 0
6: pulmonary|ventilation and oxygenation|mechanical ventilation
7: respFlowPtVentData: Mean Airway Pressure: 13
8: Airway: Intubated/oral ETT
9: Ventilation: Ventilated - with daily extubation evaluation
10: respFlowPtVentData: Exhaled TV (patient): 662
11: respFlowSettings: PEEP: 7
12: respFlowPtVentData: Exhaled TV (patient): 495
13: respFlowSettings: PEEP: 10
14: respFlowPtVentData: Mean Airway Pressure: 17
15: respFlowPtVentData: Exhaled TV (patient): 545
16: respFlowPtVentData: Mean Airway Pressure: 15
17: respFlowPtVentData: Exhaled TV (patient): 539
18: respFlowPtVentData: Mean Airway Pressure: 19
19: Morphine ()
20: respFlowPtVentData: Mean Airway Pressure: 14
CLASSIFICATION: NIPPV ONLY
"""
}







specific_order_inputs = {
    'mistral-small:24b-instruct-2501-q8_0': """
EXAMPLE 1:
1: Airway: Intubated/oral ETT
2: Ventilation: Ventilated - with daily extubation evaluation
3: respFlowSettings: PEEP: 5
4: pulmonary|medications|sedative|propofol
5: pulmonary|ventilation and oxygenation|CPAP/PEEP therapy
6: Propofol (mcg/kg/min)
7: O2 Admin Device: ventilator
8: respFlowSettings: Pressure Support: 12
9: O2 Admin Device: nasal cannula
10: Airway: Not intubated/normal airway
11: pulmonary|ventilation and oxygenation|oxygen therapy (< 40%)|nasal cannula
CLASSIFICATION: IMV TO HFNI

EXAMPLE 2:
1: Propofol (mcg/kg/min)
2: O2 Admin Device: ventilator
3: Ventilation: Ventilated - with daily extubation evaluation
4: Airway: Intubated/oral ETT
5: respFlowSettings: PEEP: 5
6: pulmonary|ventilation and oxygenation|mechanical ventilation
7: pulmonary|ventilation and oxygenation|CPAP/PEEP therapy
8: pulmonary|medications|sedative|propofol
9: pulmonary|ventilation and oxygenation|mechanical ventilation|volume controlled
10: DESCRIPTION: Source = Note; Concept = mask: mask
11: respFlowSettings: FiO2: 80
12: Fentanyl (mcg/hr)
13: O2 Admin Device: BiPAP/CPAP
14: pulmonary|ventilation and oxygenation|non-invasive ventilation
15: pulmonary|medications|sedative|dexmedetomidine
16: Airway: Not intubated/normal airway
17: Dexmedetomidine (mcg/kg/hr)
18: O2 Admin Device: BIPAP
19: O2 Admin Device: nasal cannula
CLASSIFICATION: IMV TO HFNI

EXAMPLE 3:
1: Airway: Intubated/oral ETT
2: Ventilation: Ventilated - with daily extubation evaluation
3: O2 Admin Device: ventilator
4: Propofol (mcg/kg/min)
5: neurologic|pain / agitation / altered mentation|sedative agent|propofol
6: pulmonary|ventilation and oxygenation|CPAP/PEEP therapy
7: respFlowSettings: PEEP: 5
8: respFlowSettings: Pressure Support: 12
9: Dexmedetomidine (mcg/kg/hr)
10: O2 Admin Device: nasal cannula
11: Airway: Not intubated/normal airway
CLASSIFICATION: IMV TO HFNI
""",




    'gemma2:27b-instruct-q8_0': """
EXAMPLE 1:
1: Airway: Intubated/oral ETT
2: Ventilation: Ventilated - with daily extubation evaluation
3: respFlowSettings: TV/kg IBW: 6.0498
4: respFlowSettings: Tidal Volume (set): 400
5: respFlowSettings: Vent Rate: 12
6: respFlowPtVentData: RR (patient): 21
7: pulmonary|medications|sedative|propofol
8: pulmonary|ventilation and oxygenation|CPAP/PEEP therapy
9: pulmonary|ventilation and oxygenation|mechanical ventilation|assist controlled
10: pulmonary|ventilation and oxygenation|mechanical ventilation|tidal volume 6-10 ml/kg
11: Sedation: Continuous infusion - with daily holiday
12: Propofol (mcg/kg/min)
13: O2 Admin Device: ventilator
14: Sedation Scale: RASS
15: respFlowSettings: Pressure Support: 12
16: O2 Admin Device: nasal cannula
17: respFlowPtVentData: RR (patient): 20
18: pulmonary|ventilation and oxygenation|oxygen therapy (< 40%)|nasal cannula
CLASSIFICATION: IMV TO HFNI

EXAMPLE 2:
1: ATIVAN
2: Propofol (mcg/kg/min)
3: O2 Admin Device: ventilator
4: Airway: Intubated/oral ETT
5: Sedation: Continuous infusion - with daily holiday
6: Ventilation: Ventilated - with daily extubation evaluation
7: respFlowSettings: Tidal Volume (set): 500
8: respFlowSettings: TV/kg IBW: 7.0685
9: respFlowSettings: Vent Rate: 14
10: pulmonary|medications|sedative
11: pulmonary|ventilation and oxygenation|mechanical ventilation
12: pulmonary|ventilation and oxygenation|mechanical ventilation|assist controlled
13: pulmonary|ventilation and oxygenation|CPAP/PEEP therapy
14: pulmonary|medications|sedative|propofol
15: pulmonary|ventilation and oxygenation|mechanical ventilation|volume controlled
16: pulmonary|ventilation and oxygenation|mechanical ventilation|tidal volume 6-10 ml/kg
17: respFlowSettings: Tidal Volume (set): 400
18: respFlowSettings: TV/kg IBW: 5.6548
19: Ordered Protocols: Lung protective ventilation
20: Ordered Protocols: Ventilator bundle
21: Ordered Protocols: Ventilator wean
22: Ordered Protocols: Sedation
23: respFlowPtVentData: RR (patient): 20
24: Sedation Scale: RASS
25: respFlowPtVentData: RR (patient): 21
26: respFlowSettings: FiO2: 45
27: respFlowPtVentData: RR (patient): 17
28: Fentanyl (mcg/hr)
29: O2 Admin Device: BiPAP/CPAP
30: pulmonary|ventilation and oxygenation|non-invasive ventilation
31: pulmonary|medications|sedative|dexmedetomidine
32: Sedation: Continuous infusion - no daily holiday
33: Dexmedetomidine (mcg/kg/hr)
34: O2 Admin Device: BIPAP
35: O2 Admin Device: nasal cannula
CLASSIFICATION: IMV TO HFNI

EXAMPLE 3:
1: Airway: Intubated/oral ETT
2: Ventilation: Ventilated - with daily extubation evaluation
3: respFlowSettings: Tidal Volume (set): 385
4: respFlowSettings: TV/kg IBW: 6.0268
5: respFlowSettings: Vent Rate: 12
6: O2 Admin Device: ventilator
7: Propofol (mcg/kg/min)
8: neurologic|pain / agitation / altered mentation|sedative agent|propofol
9: pulmonary|ventilation and oxygenation|mechanical ventilation|assist controlled
10: DIPRIVAN
11: pulmonary|ventilation and oxygenation|CPAP/PEEP therapy
12: respFlowPtVentData: RR (patient): 17
13: Ordered Protocols: Ventilator wean
14: Sedation: Continuous infusion - with daily holiday
15: Ordered Protocols: Ventilator bundle
16: Ordered Protocols: Sedation
17: Sedation Scale: RASS
18: respFlowPtVentData: RR (patient): 20
19: respFlowSettings: Pressure Support: 12
20: Dexmedetomidine (mcg/kg/hr)
21: Ordered Protocols: Lung protective ventilation
22: O2 Admin Device: nasal cannula
23: XANAX
CLASSIFICATION: IMV TO HFNI
""",




    'phi4:14b-q8_0': """
EXAMPLE 1:
1: Airway: Intubated/oral ETT
2: Ventilation: Ventilated - with daily extubation evaluation
3: pulmonary|ventilation and oxygenation|CPAP/PEEP therapy
4: pulmonary|medications|sedative|propofol
5: pulmonary|ventilation and oxygenation|mechanical ventilation|assist controlled
6: Propofol (mcg/kg/min)
7: O2 Admin Device: ventilator
8: respFlowSettings: Pressure Support: 12
CLASSIFICATION: IMV TO HFNI

EXAMPLE 2:
1: Propofol (mcg/kg/min)
2: O2 Admin Device: ventilator
3: Ventilation: Ventilated - with daily extubation evaluation
4: Airway: Intubated/oral ETT
5: pulmonary|ventilation and oxygenation|mechanical ventilation
6: pulmonary|medications|sedative|propofol
7: pulmonary|ventilation and oxygenation|CPAP/PEEP therapy
8: DESCRIPTION: Source = Note; Concept = mask: mask
9: pulmonary|ventilation and oxygenation|mechanical ventilation|assist controlled
10: pulmonary|ventilation and oxygenation|mechanical ventilation|volume controlled
11: Ordered Protocols: Ventilator wean
12: Fentanyl (mcg/hr)
13: O2 Admin Device: BiPAP/CPAP
14: pulmonary|medications|sedative|dexmedetomidine
15: pulmonary|ventilation and oxygenation|non-invasive ventilation
16: Dexmedetomidine (mcg/kg/hr)
17: O2 Admin Device: BIPAP
18: respFlowPtVentData: RR (patient): 27
CLASSIFICATION: IMV TO HFNI

EXAMPLE 3:
1: Airway: Intubated/oral ETT
2: Ventilation: Ventilated - with daily extubation evaluation
3: O2 Admin Device: ventilator
4: Propofol (mcg/kg/min)
5: pulmonary|ventilation and oxygenation|mechanical ventilation|assist controlled
6: neurologic|pain / agitation / altered mentation|sedative agent|propofol
7: DIPRIVAN
8: pulmonary|ventilation and oxygenation|CPAP/PEEP therapy
9: Ordered Protocols: Ventilator wean
10: respFlowSettings: Pressure Support: 12
11: Dexmedetomidine (mcg/kg/hr)
CLASSIFICATION: IMV TO HFNI
""",
}