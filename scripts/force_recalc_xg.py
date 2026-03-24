
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from puck import analyze

def force_recalc():
    print("--- Forcing xG Recalculation (2025-2026) ---")
    csv_path = 'data/20252026.csv'
    model_path = os.path.join('analysis', 'xgs', 'xg_model_xgboost_nested_20202021.joblib')
    
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows.")
    
    # We pass the full df to _predict_xgs with behavior='overwrite'
    # and csv_path to save it.
    print(f"Running _predict_xgs with model: {model_path}")
    df_new, clf, meta = analyze._predict_xgs(df, model_path=model_path, csv_path=csv_path, behavior='overwrite')
    
    print("Recalculation Complete.")
    
    # Verify results
    shots = df_new[df_new.event.isin(['goal','shot-on-goal','missed-shot','blocked-shot'])]
    actual_goals = len(df_new[df_new.event=='goal'])
    sum_xg = shots.xgs.sum()
    print(f"\nFinal Results:")
    print(f"  Total xG: {sum_xg:.2f}")
    print(f"  Total Goals: {actual_goals}")
    print(f"  Ratio xG/Goal: {sum_xg/actual_goals:.4f}")
    print("\nLayer Averages:")
    print(shots[['prob_block','prob_accuracy','prob_finish','xgs']].mean())

if __name__ == "__main__":
    force_recalc()
