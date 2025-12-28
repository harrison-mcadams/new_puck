
import sys
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import fit_xgs
# Ensure NestedXGClassifier is importable for unpickling
from puck import fit_xgs
# Ensure NestedXGClassifier is importable for unpickling
from puck import fit_nested_xgs
from puck import impute

def main():
    print("--- xG Inflation Debugger ---")
    
    # 1. Load Data (2025-2026)
    csv_path = 'data/20252026.csv'
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Season CSV not found.")
        return
    
    # Filter to shot attempts only for fair comparison
    shot_events = ['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot']
    df_shots = df[df['event'].isin(shot_events)].copy()
    print(f"Loaded {len(df_shots)} shot attempts.")

    # 2. Load Model
    model_path = 'analysis/xgs/xg_model_nested.joblib'
    print(f"Loading model from {model_path}...")
    try:
        clf = joblib.load(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 3. Predict xG (using internal predict_proba to replicate analyze.py logic)
    print("Predicting xG...")
    
    # Ensure columns exist (handling the analyze.py prep logic check)
    if 'shot_type' not in df_shots.columns:
        df_shots['shot_type'] = 'Unknown'
    df_shots['shot_type'] = df_shots['shot_type'].fillna('Unknown')
    
    if 'game_state' not in df_shots.columns:
         df_shots['game_state'] = '5v5' # default
         
    # Apply Imputation (Crucial matching analyze.py)
    print("Applying blocked shot imputation...")
    df_shots = impute.impute_blocked_shot_origins(df_shots, method='mean_6')
         
    # Run Prediction
    try:
        probs = clf.predict_proba(df_shots)[:, 1]
        df_shots['xg'] = probs
        # NOTE: NO Manual override for blocked shots here, per new plan.
    except Exception as e:
        print(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Compare Totals
    total_goals = df_shots[df_shots['event'] == 'goal'].shape[0]
    total_xg = df_shots['xg'].sum()
    ratio = total_xg / total_goals if total_goals > 0 else 0
    
    print("\n--- Calibration Summary ---")
    print(f"Total Actual Goals: {total_goals}")
    print(f"Total Predicted xG: {total_xg:.2f}")
    print(f"Ratio (xG / Goals): {ratio:.2f} (Expected ~1.0)")
    
    # 5. Layer Diagnostics
    print("\n--- Layer Diagnostics ---")
    
    if hasattr(clf, 'predict_proba_layer'):
        # A. Block Model
        p_blocked = clf.predict_proba_layer(df_shots, 'block')
        df_shots['p_blocked'] = p_blocked
        
        actual_blocked = (df_shots['event'] == 'blocked-shot').mean()
        pred_blocked = p_blocked.mean()
        print(f"Block Rate: Actual={actual_blocked:.1%}, Predicted={pred_blocked:.1%}")

        # B. Accuracy Model (P(On Net | Unblocked))
        # Evaluate only on unblocked shots
        unblocked_mask = (df_shots['event'] != 'blocked-shot')
        df_unblocked = df_shots[unblocked_mask].copy()
        
        p_on_net = clf.predict_proba_layer(df_unblocked, 'accuracy')
        
        # "On Net" definitions: Goal or Shot-on-Goal (saved). Miss/Block are 0.
        actual_on_net = df_unblocked['event'].isin(['shot-on-goal', 'goal']).mean()
        pred_on_net = p_on_net.mean()
        print(f"Accuracy (On Net | Unblocked): Actual={actual_on_net:.1%}, Predicted={pred_on_net:.1%}")

        # C. Finish Model (P(Goal | On Net))
        on_net_mask = df_shots['event'].isin(['shot-on-goal', 'goal'])
        df_on_net = df_shots[on_net_mask].copy()
        
        p_goal = clf.predict_proba_layer(df_on_net, 'finish')
        
        actual_finish = (df_on_net['event'] == 'goal').mean()
        pred_finish = p_goal.mean()
        print(f"Finish (Goal | On Net): Actual={actual_finish:.1%}, Predicted={pred_finish:.1%}")
        
        # NEW: Breakdown by Event Type
        print("\n--- Breakdown by Event Type ---")
        summary = df_shots.groupby('event').agg(
            Count=('event', 'count'),
            Avg_xG=('xg', 'mean'),
            Total_xG=('xg', 'sum')
        )
        print(summary.to_string())
        print("-" * 30)
    else:
        print("Warning: Classifier does not support 'predict_proba_layer'. Skipping detailed diagnostics.")

    # 6. Feature Sanity Check
    print("\n--- Feature Stats (All Shots) ---")
    print(df_shots[['distance', 'angle_deg']].describe().to_string())
    
    print("\n--- Top High xG Events (Top 10) ---")
    print(df_shots.sort_values('xg', ascending=False)[['event', 'shot_type', 'distance', 'angle_deg', 'xg']].head(10).to_string())

if __name__ == "__main__":
    main()
