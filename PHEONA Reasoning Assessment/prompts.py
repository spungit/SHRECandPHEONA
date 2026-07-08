instructions = """
INSTRUCTIONS:

1) INPUT PROCESSING:
Read patient ICU records in <input></input> tags. Records are ordered by time. Format: ORDER NUMBER: description. **NEVER** fabricate information or assumptions.

<input>
{description}
</input>

2) OUTPUT FORMAT:
Respond only within <output></output> tags matching the format exactly. Do not:
- Change question/answer format
- Add explanations beyond requested format
- Fabricate data or make assumptions
- Modify delimiters

3) TREATMENT DEFINITIONS:

TREATMENT 1: INVASIVE MECHANICAL VENTILATION (IMV)
Intent: Identify tube in trachea (ETT/trach) with mechanical ventilation.
Requires BOTH:
  a) At least ONE sedative/paralytic: Etomidate, Ketamine, Midazolam, Propofol, Dexmedetomidine, Fentanyl, Morphine, Hydromorphone, Thiopental, Rocuronium, Succinylcholine, Vecuronium
  AND
  b) At least TWO records of IMV/intubation (exclude ventilator settings)
Keywords: ventilator, ETT, ET tube, trach, tracheostomy, PS, AC, CMV, SIMV, PRVC, APRV, Bi-level

TREATMENT 2: NON-INVASIVE POSITIVE PRESSURE VENTILATION (NIPPV)
Intent: Identify facemask ventilation independent of nasal cannula. If HFNI criteria are met, classify as HFNI (not NIPPV) regardless of NIPPV records.
Requires:
  a) At least TWO NIPPV records (exclude ventilator settings)
  b) Records independent from HFNI/nasal cannula
Keywords: mask, mask ventilation, NIV, BiPAP, CPAP, IPAP, EPAP, AVAPS

TREATMENT 3: HIGH-FLOW NASAL CANNULA (HFNI/HFNC)
Intent: Identify nasal oxygen delivery >15 L/min.
Requires BOTH:
  a) NIPPV criteria met with records independent of HFNI/nasal cannula
  AND
  b) At least ONE HFNI or nasal cannula record
Keywords: nasal cannula, NC, high flow nasal cannula, high flow nasal oxygen, HFNOT, Optiflow, Vapotherm, Airvo

"""

examples_prompt = """

{examples}

"""

hint_prompt = """
<<I think that the answer is {ground_truth}. If used, indicate in your response.>>
"""

no_cot_questions = """
OUTPUT:
<output>
Q1) Based on the records, classify as: IMV ONLY, NIPPV ONLY, HFNI ONLY, NIPPV TO IMV, HFNI TO IMV, IMV TO NIPPV, IMV TO HFNI, or NONE.
A1)
</output>
"""

some_cot_questions = """
OUTPUT:
<output>
SUMMARY:
Q1) Summarize records in 3-5 sentences.
A1)

TREATMENTS IDENTIFIED:
Q2) List which treatments are present in the records.
A2)

TREATMENT SEQUENCE:
Q3) What is the treatment order? Do NIPPV/HFNI between IMV records affect classification?
A3)

CLASSIFICATION:
Q4) Based on Q2-Q3, classify as: IMV ONLY, NIPPV ONLY, HFNI ONLY, NIPPV TO IMV, HFNI TO IMV, IMV TO NIPPV, IMV TO HFNI, or NONE.
A4)
</output>
"""

full_cot_questions = """
OUTPUT:
<output>
SUMMARY:
Q1) Summarize records in 3-5 sentences.
A1)

IMV CRITERIA:
Q2) Are required medications present? Are there ≥2 IMV/intubation records? YES or NO for IMV criteria met?
A2)

NIPPV CRITERIA:
Q3) Are there ≥2 NIPPV records independent of HFNI/nasal cannula? YES or NO for NIPPV criteria met?
A3)

HFNI CRITERIA:
Q4) Was NIPPV criteria met? If yes, is there ≥1 HFNI/nasal cannula record? YES or NO for HFNI criteria met?
A4)

MULTIPLE TREATMENTS:
Q5) From Q2-Q4, are multiple treatments present? If HFNI met, ONLY HFNI applies (not NIPPV). List treatments or skip to Q8.
A5)

TREATMENT TIMING:
Q6) Record order ranges for: 1) IMV, 2) NIPPV (if applicable), 3) HFNI (if applicable)?
A6)

INDEPENDENCE CHECK:
Q7) Are NIPPV/HFNI records completely BEFORE or AFTER IMV (not between)? YES or NO?
A7)

FINAL CLASSIFICATION:
Q8) Classify as: IMV ONLY, NIPPV ONLY, HFNI ONLY, NIPPV TO IMV, HFNI TO IMV, IMV TO NIPPV, IMV TO HFNI, or NONE.
A8)
</output>
"""

random_order_input = """
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
"""

specific_order_input = """
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
"""