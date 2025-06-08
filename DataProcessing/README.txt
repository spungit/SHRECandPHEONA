File Descriptions:

BEFORE CONCEPT CLASSIFICAITON
1) extract_descriptions.py: Processes all of the concepts into the constructed concepts to use for LLM-based classification.

AFTER CONCEPT CLASSIFICATION
2) filter_pts.py: Filters ALL PATIENT RECORDS using the classification results. Independent of identify_cohorts.
3) identify_cohorts.py: Identifies cohort based on ICU stay and age inclusion/exclusion criteria. Independent of filter_pts.
4) clean_filter_pts.py: Adds in the ordering AND filters to cohort pts. Must be run AFTER BOTH filter_pts and identify_cohorts.
5) replace_strings.py: Maps extra text in strings to empty to reduce token size