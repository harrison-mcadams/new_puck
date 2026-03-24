
import pandas as pd
import numpy as np
import os
import sys
import joblib

# Add project root to path
sys.path.append(os.getcwd())

from puck import analyze
from puck import config

def benchmark():
    print("--- Historical Benchmark (2023-2024) ---")
    
    csv_path = 'data/20232024.csv'
    if not os.path.exists(csv_path):
        print(f"Data file {csv_path} not found.")
        return

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows.")

    shot_events = ['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot']
    df_shots = df[df['event'].isin(shot_events)].head(500).copy()
    print(f"Sample size: {len(df_shots)} shots.")
    
    model_path = os.path.join(config.ANALYSIS_DIR, 'xgs', 'xg_model_xgboost_nested_20202021.joblib')
    print(f"Loading model: {model_path}")
    
    # Predict
    print("Running analyze._predict_xgs...")
    try:
        df_pred, clf, meta = analyze._predict_xgs(df_shots, model_path=model_path, behavior='return')
        print(f"Pred shape: {df_pred.shape}")
        
        if 'xgs' in df_pred.columns:
            print(f"SUCCESS: Found xgs column.")
            print(f"Mean xG: {df_pred['xgs'].mean():.4f}")
            print(f"Layer averages:")
            for l in ['prob_block', 'prob_accuracy', 'prob_finish']:
                if l in df_pred.columns:
                    print(f"  {l}: {df_pred[l].mean():.4f}")
        else:
            print("ERROR: xgs column missing in result.")
    except Exception as e:
        print(f"EXCEPTION during prediction: {e}")

if __name__ == "__main__":
    benchmark()
