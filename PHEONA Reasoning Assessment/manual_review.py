import os
import pandas as pd

filepath = ''
models = ['phi414bq80', 'mistralsmall24binstruct2501q80', 'deepseekr132b']

for m in models:
    print('\n\nProcessing model:', m)

    ## Natural DataFrame
    natural_filename = f'natural_results_{m}.xlsx'
    print(f'Natural filename: {natural_filename}')

    if os.path.exists(filepath + natural_filename):
        natural_df = pd.read_excel(filepath + natural_filename)
    else:
        print(f'Files not found for model {m}. Skipping...')
        continue

    print('Natural DataFrame shape:', natural_df.shape)
    print('Natural DataFrame columns:', natural_df.columns.tolist())
    print('Natural DataFrame head:')
    print(natural_df.head())

    for idx, row in natural_df.iterrows():
        print(f"\nProcessing row {idx + 1}/{natural_df.shape[0]}")
        previous_nocot_restoration_response = row['nocot_restoration_error']
        previous_somecot_restoration_response = row['somecot_restoration_error']
        previous_fullcot_restoration_response = row['fullcot_restoration_error']

        previous_nocot_unfaithfulshortcut_response = row['nocot_unfaithfulshortcut_error']
        previous_somecot_unfaithfulshortcut_response = row['somecot_unfaithfulshortcut_error']
        previous_fullcot_unfaithfulshortcut_response = row['fullcot_unfaithfulshortcut_error']

        restoration_filledin = not pd.isna(previous_nocot_restoration_response) and not pd.isna(previous_somecot_restoration_response) and not pd.isna(previous_fullcot_restoration_response)
        unfaithful_shortcut_filledin = not pd.isna(previous_nocot_unfaithfulshortcut_response) and not pd.isna(previous_somecot_unfaithfulshortcut_response) and not pd.isna(previous_fullcot_unfaithfulshortcut_response)

        if restoration_filledin and unfaithful_shortcut_filledin:
            print(f'Row {idx + 1} has already been assessed.')
            continue

        print(f'Row {idx + 1} has not been assessed yet.')
        description = row['description']
        nocot_response = row['nocot_response']
        somecot_response = row['somecot_response']
        fullcot_response = row['fullcot_response']

        print(f"\n\nDescription:\n{description}")
        print(f"\n\nNoCoT Response:\n{nocot_response}")
        nocot_restoration_user_response = input("Assessment of NoCoT response restoration (1/0): ")
        nocot_unfaithfulshortcut_user_response = input("Assessment of NoCoT unfaithful shortcut (1/0): ")
        print(f"\n\nDescription:\n{description}")
        print(f"\n\nSomeCoT Response:\n{somecot_response}")
        somecot_restoration_user_response = input("Assessment of SomeCoT response restoration (1/0): ")
        somecot_unfaithfulshortcut_user_response = input("Assessment of SomeCoT unfaithful shortcut (1/0): ")
        print(f"\n\nDescription:\n{description}")
        print(f"\n\nFullCoT Response:\n{fullcot_response}")
        fullcot_restoration_user_response = input("Assessment of FullCoT response restoration (1/0): ")
        fullcot_unfaithfulshortcut_user_response = input("Assessment of FullCoT unfaithful shortcut (1/0): ")

        natural_df.at[idx, 'nocot_restoration_error'] = int(nocot_restoration_user_response)
        natural_df.at[idx, 'somecot_restoration_error'] = int(somecot_restoration_user_response)
        natural_df.at[idx, 'fullcot_restoration_error'] = int(fullcot_restoration_user_response)
        natural_df.at[idx, 'nocot_unfaithfulshortcut_error'] = int(nocot_unfaithfulshortcut_user_response)
        natural_df.at[idx, 'somecot_unfaithfulshortcut_error'] = int(somecot_unfaithfulshortcut_user_response)
        natural_df.at[idx, 'fullcot_unfaithfulshortcut_error'] = int(fullcot_unfaithfulshortcut_user_response)
        natural_df.to_excel(filepath + natural_filename, index=False)
        print('Responses saved after this description.')
        
    print('Restoration errors added to DataFrame and saved.')

    ## Simulated DataFrame
    simulated_filename = f'simulated_results_{m}.csv'
    explanationcorrectness_filename = f'explanation_correctness_{m}.csv'
    print(f'\n\nSimulated filename: {simulated_filename}')

    if os.path.exists(filepath + simulated_filename) and not os.path.exists(filepath + explanationcorrectness_filename):
        print(f'Saved explanation correctenss results not found for model {m}. Using simulated results.')
        simulated_df = pd.read_csv(filepath + simulated_filename)
        df_to_save = simulated_df[['description', 'no_cot_response', 'some_cot_response', 'full_cot_response']].copy()
        df_to_save['no_cot_explanationcorrectness'] = pd.NA
        df_to_save['some_cot_explanationcorrectness'] = pd.NA
        df_to_save['full_cot_explanationcorrectness'] = pd.NA
        n_to_review = 100
        df_to_save = df_to_save.iloc[:n_to_review] if df_to_save.shape[0] > n_to_review else df_to_save
    elif os.path.exists(filepath + explanationcorrectness_filename):
        print(f'Explanation correctness file found for model {m}. Loading existing results.')
        df_to_save = pd.read_csv(filepath + explanationcorrectness_filename)
    else:
        print(f'Files not found for model {m}. Skipping...')
        continue

    print('Simulated DataFrame shape:', df_to_save.shape)
    print('Simulated DataFrame columns:', df_to_save.columns.tolist())
    print('Simulated DataFrame head:')
    print(df_to_save.head())

    for idx, row in df_to_save.iterrows():
        print(f"\nProcessing row {idx + 1}/{df_to_save.shape[0]}")

        description = row['description']
        no_cot_response = row['no_cot_response']
        some_cot_response = row['some_cot_response']
        full_cot_response = row['full_cot_response']

        response_filledin = not pd.isna(row['no_cot_explanationcorrectness']) and not pd.isna(row['some_cot_explanationcorrectness']) and not pd.isna(row['full_cot_explanationcorrectness'])

        if response_filledin:
            print(f'Row {idx + 1} has already been assessed.')
            continue
    
        print(f'Row {idx + 1} has not been assessed yet.')
        print(f"\n\nDescription:\n{description}")
        print(f"\n\nNoCoT Response:\n{no_cot_response}")
        nocot_explanationcorrectness_user_response = input("Assessment of NoCoT response explanation correctness (1/0): ")

        print(f"\n\nDescription:\n{description}")
        print(f"\n\nSomeCoT Response:\n{some_cot_response}")
        somecot_explanationcorrectness_user_response = input("Assessment of SomeCoT response explanation correctness (1/0): ")

        print(f"\n\nDescription:\n{description}")
        print(f"\n\nFullCoT Response:\n{full_cot_response}")
        fullcot_explanationcorrectness_user_response = input("Assessment of FullCoT response explanation correctness (1/0): ")

        df_to_save.at[idx, 'no_cot_explanationcorrectness'] = int(nocot_explanationcorrectness_user_response)
        df_to_save.at[idx, 'some_cot_explanationcorrectness'] = int(somecot_explanationcorrectness_user_response)
        df_to_save.at[idx, 'full_cot_explanationcorrectness'] = int(fullcot_explanationcorrectness_user_response)

        df_to_save.to_csv(filepath + explanationcorrectness_filename, index=False)
