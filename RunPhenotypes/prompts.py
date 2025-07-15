cot_prompt = """
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
