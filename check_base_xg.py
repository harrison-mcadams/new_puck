
import pandas as pd
import numpy as np
import joblib
from puck import data_pipeline

def check_base_xg():
    df = pd.read_csv('data/20252026.csv')
    df_proc = data_pipeline.preprocess_features(df, apply_filtering=True)
    
    model = joblib.load('analysis/xgs/xg_model_nested_tensor.joblib')
    # Get base xG
    probs = model.predict_proba(df_proc)[:, 1]
    df_proc['xg_base'] = probs
    df_proc['is_goal'] = (df_proc['event'] == 'goal').astype(int)
    
    for state in ['5v5', '5v4', '4v5']:
        subset = df_proc[df_proc['relative_game_state'] == state]
        if len(subset) == 0: continue
        
        mean_xg = subset['xg_base'].mean()
        actual_rate = subset['is_goal'].mean()
        print(f"State {state}:")
        print(f"  Base xG Mean: {mean_xg:.4f}")
        print(f"  Actual Rate:  {actual_rate:.4f}")
        print(f"  Ratio (Act/Pred): {actual_rate/mean_xg:.2f}")

if __name__ == "__main__":
    check_base_xg()
