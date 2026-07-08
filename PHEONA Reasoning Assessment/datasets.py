import pandas as pd
from pathlib import Path


def import_parquet(filepath: str) -> pd.DataFrame:
    """
    Import a parquet file and return as a pandas DataFrame.
    
    Args:
        filepath (str): Path to the parquet file to import
        
    Returns:
        pd.DataFrame: DataFrame containing the data from the parquet file
    """
    return pd.read_parquet(filepath)


def stratified_random_sample(df: pd.DataFrame, outcome_col: str = "outcome", n_per_group: int = 10) -> pd.DataFrame:
    """
    Group by outcome, randomly shuffle, and take n rows per group.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        outcome_col (str): Column name to group by (default: "outcome")
        n_per_group (int): Number of rows to take per group (default: 10)
        
    Returns:
        pd.DataFrame: Final DataFrame with sampled rows from each group
    """
    final_df = df.groupby(outcome_col, group_keys=False).apply(
        lambda x: x.sample(frac=1).head(n_per_group)
    ).reset_index(drop=True)
    
    return final_df


if __name__ == "__main__":
    filename = "phenotyping_processed_mistralsmall24binstruct2501q80_ordered.parquet"
    input_filepath = Path(__file__).parent / "inputs" / filename
    
    df = import_parquet(str(input_filepath))
    print(df.head())
    
    # Get distinct descriptions before sampling
    print(f"DataFrame shape before removing duplicates: {df.shape}")
    df = df.drop_duplicates('ordered_string')
    print(f"DataFrame shape after removing duplicates: {df.shape}")
    
    # perform stratified random sampling for the explanation correctness evaluation
    output_filepath = Path(__file__).parent / "outputs" / 'explanation_correctness_inputdata'
    final_df = stratified_random_sample(df)
    final_df = final_df.drop(columns=["processed_string", "level_1"], errors="ignore").rename(columns={"ordered_string": "description", "patientunitstayid": "record"}).astype({'record': str})
    print(f"\nStratified sample shape: {final_df.shape}")
    print(final_df)
    final_df.to_parquet(f"{output_filepath}.parquet", index=False)
    final_df.to_excel(f"{output_filepath}.xlsx", index=False)

    # perform stratified random sampling for the few shot hint evaluation
    output_filepath_fewshothint = Path(__file__).parent / "outputs" / 'fewshothint_inputdata'
    final_df_fewshothint = stratified_random_sample(df, n_per_group=100)
    final_df_fewshothint = final_df_fewshothint.drop(columns=["processed_string", "level_1"], errors="ignore").rename(columns={"ordered_string": "description", "patientunitstayid": "record"}).astype({'record': str})
    print(f"\nFew shot hint stratified sample shape: {final_df_fewshothint.shape}")
    print(final_df_fewshothint)
    final_df_fewshothint.to_parquet(f"{output_filepath_fewshothint}.parquet", index=False)
    final_df_fewshothint.to_excel(f"{output_filepath_fewshothint}.xlsx", index=False)

    # remove the processed_string column for the restoration and unfaithful shortcut error evaluation
    output_filepath_restoration_unfaithful = Path(__file__).parent / "outputs" / 'restoration_unfaithful_inputdata'
    df_no_processed_string = df.drop(columns=["processed_string"]).rename(columns={"ordered_string": "description", "outcome": "outcome", "patientunitstayid": "record"}).astype({'record': str})
    print(f"\nDataFrame shape without processed_string: {df_no_processed_string.shape}")
    print(df_no_processed_string.head())
    df_no_processed_string.to_parquet(f"{output_filepath_restoration_unfaithful}.parquet", index=False)
    df_no_processed_string.to_excel(f"{output_filepath_restoration_unfaithful}.xlsx", index=False)
