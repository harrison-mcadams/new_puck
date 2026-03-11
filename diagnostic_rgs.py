
import pandas as pd
import numpy as np
from puck import data_pipeline, fit_xgs

def diagnostic():
    # Load 2023-2024 as a sample
    df = pd.read_csv('data/20232024/20232024_df.csv')
    print(f"Sample size: {len(df)}")
    
    # Preprocess
    df_proc = data_pipeline.preprocess_features(df, is_training=True, verbose=True)
    
    print("\nValue Counts for 5v4/4v5:")
    subset = df_proc[df_proc['game_state'].isin(['5v4', '4v5'])]
    print(subset.groupby(['game_state', 'is_home'])['relative_game_state'].value_counts())
    
    print("\nOverall relative_game_state counts:")
    print(df_proc['relative_game_state'].value_counts().head(10))

if __name__ == "__main__":
    diagnostic()
