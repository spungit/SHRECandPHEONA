categorization_prompt_wo_meds = """
Instructions:
1) INPUT: The input contains the source table name of the concept and the concept description. The input will be provided to you in the following format: <input>INPUT</input>. **DO NOT FABRICATE AN INPUT**, even if the input is vague or unclear.

2) INSTRUCTIONS: 
2a) Respond to the questions delimited by the <output> tags. Provide your answer **exactly** in the format specified between the <output> tags. Do **NOT** do any of the following:
    - Modify the format of the questions or answers.
    - Provide explanations or additional details beyond the format requested.
    - Fabricate an input description or add information that is not present in the description, even if it is empty or unclear.

2b) When determining if the description matches any of the concept categories, consider the following:
    - **Match Criteria**: The concept description either exactly matches or partially matches the definition, terms, and/or acronyms for each concept. For partial matches, if the **OVERALL** meaning of the description aligns with the concept, it is a match. If the concept relates to removal, absence, or discontinuation of any of the concept categories, it is a match to that concept. Verify that any acronyms are explicitly defined in the concept criteria: **DO NOT ASSUME** the meaning of acronyms.
    - **Unclear or Unmatched Descriptions**: If the concept description is unclear or the content of the description does not match any of the listed terms or acronyms, it is not a match to any concept.
    - **No Assumptions**: Do **not fabricate** additional information or assume meanings for vague, incomplete, or ambiguous descriptions. Only use the **exact terms** provided in the description.

3) CONCEPT CRITERIA: There are four concept categories to consider:

    3a) **Concept 1: Invasive Mechanical Ventilation (IMV)**
        - **Definition**: This involves a tube in the trachea (either an endotracheal tube placed through the mouth, or rarely the nose, OR a surgically placed traheostomy tube) connected to a ventilator, delivering mechanical ventilation.
        - **Terms and Acronyms**:
            - Terms: Endotracheal tube, tracheostomy tube, tracheostomy, trach tube, trach (either unspecified or specific to mechanical ventilation), ventilator, vent, intubated, intubation, extubation, invasive mechanical ventilation, continuous positive airway pressure, pressure support, assist control vent mode, continuous mandatory ventilation vent mode, synchronized intermittent mandatory ventilation vent mode, pressure regulated volume control vent mode, airway pressure release ventilation vent mode.
            - Acronyms: ETT or ET (endotracheal tube), IMV (invasive mechanical ventilation), CPAP (continuous positive airway pressure), PS (pressure support), AC (assist control vent mode), CMV (continuous mandatory ventilation vent mode), SIMV (synchronized intermittent mandatory ventilation vent mode), PRVC (pressure regulated volume control vent mode), APRV or Bi-level (airway pressure release ventilation vent mode)

    3b) **Concept 2: High-Flow Nasal Insufflation/Nasal Cannula (HFNI/HFNC)**
        - **Definition**: Oxygen is delivered through a nasal cannula at a flow rate above 15 L/min, with adjustments for oxygen concentration and flow rate.
        - **Terms and Acronyms**:
            - Terms: Nasal cannula, Vapotherm, Airvo, Optiflow, High flow nasal cannula, Nasal high flow, Heated and humidified high flow nasal oxygen, Heated and humidified high flow nasal cannula, High flow nasal insufflation, High flow nasal oxygen, High flow nasal cannula, High flow cannula, High velocity nasal insufflation (FDA designation for Vapotherm).
            - Acronyms: HHFNO (heated and humidified high flow nasal oxygen), HHFNC (heated and humidified high flow nasal cannula), HFNI (high flow nasal insufflation), HFNO (high flow nasal oxygen), HFNC (high flow nasal cannula), NC (nasal cannula).

    3c) **Concept 3: Non-Invasive Positive Pressure Ventilation (NIPPV)**
        - **Definition**: Non-invasive ventilation via a facemask, where the clinician adjusts inspiratory pressure (the pressure that supports the breath), FiO2, and PEEP (the pressure maintained in the system at the end of a breath to keep the lungs from collapsing) to assist breathing.
        - **Terms and Acronyms**:
            - Terms: Mask, mask ventilation, non-invasive ventilation, bi-level positive airway pressure, continuous positive airway pressure, inspiratory positive airway pressure, expiratory positive airway pressure, pressure support, average volume assured pressure support.
            - Acronyms: BiPAP (bilevel positive airway pressure), CPAP (continuous positive airway pressure), IPAP (inspiratory positive airway pressure), EPAP (expiratory positive airway pressure), PS (pressure support), AVAPS (average volume assured pressure support, an advanced mode that is exactly mechanical ventilation without the endotracheal tube), NIV (non-invasive ventilation).

    3d) **Concept 4**: Convensional Oxygen Therapies
        - **Definition**: Includes trach collar, venturi mask, trach tent, oximizer, oxymizer, oxymask, oximizer nasal cannula, misty ox, oxymask, oxi-mask, partial rebreather, simple mask.
        - **Terms and Acronyms**:
            - Terms: Trach collar, venturi mask, trach tent, oximizer, oxymizer, oxymask, oximizer nasal cannula, misty ox, oxymask, oxi-mask, partial rebreather, simple mask.
        
4) INPUT: <input>{description}</input>

<output>
OUTPUT:
Q1) What was the exact provided input description?
A1)
Q2) Does the description match the criteria for Concept 1 - Invasive Mechanical Ventilation (IMV)? Provide a brief judgment, no more than one sentence.
A2)
Q3) Does the description match the criteria for Concept 2 - High-Flow Nasal Insufflation/Nasal Cannula (HFNI/HFNC)? Provide a brief judgment, no more than one sentence.
A3)
Q4) Does the description match the criteria for Concept 3 - Non-Invasive Positive Pressure Ventilation (NIPPV)? Provide a brief judgment, no more than one sentence.
A4)
Q5) Based on your answers to the **previous three** questions, if the answer to any of the questions is 'YES', respond with 'YES'. Otherwise, the respond with 'NO'.
A5) Final Judgment: [YES/NO]
</output>
"""

categorization_prompt_w_meds = """
Instructions:
1) INPUT: The input contains the source table name of the concept and the concept description. It may be incomplete or ambiguous. The input will be provided to you in the following format: <input>INPUT</input>. **DO NOT FABRICATE AN INPUT**, even if the description is vague or unclear.

2) INSTRUCTIONS: Respond to the questions delimited by the <output> tags. Provide your answer **exactly** in the format specified between the <output> tags. Do **NOT** do any of the following:
    - Modify the format of the questions or answers.
    - Provide explanations or additional details beyond the format requested.
    - Fabricate an input description or add information that is not present in the description, even if it is empty or unclear.
    - Use anything other than the exact terms provided in the description when responding to the questions.

3) INPUT: <input>{description}</input>

<output>
OUTPUT:
Q1) What was the exact provided input description?
A1)
Q2) Does the description contain an **exact match** to one of *ONLY* these medications (do not consider acronyms): **SPECIFIC** sedatives - Etomidate, Ketamine, Midazolam (Versed), Propofol, Dexmedetomidine (Precedex), Fentanyl, Morphine, Hydromorphone (Dilaudid), Thiopental, Cisatracurium; **SPECIFIC** paralytics - Rocuronium, Succinylcholine, Vecuronium? Generic mentions of medication classes or other medications not listed are not valid. Respond with 'YES' or 'NO'.
A2) Final Judgment: [YES/NO]
</output>
"""