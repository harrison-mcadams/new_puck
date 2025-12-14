
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
from puck import fit_nested_xgs

def main():
    print("--- xG Inflation Debugger ---")
    
    # 1. Load Data (2025-2026)
    csv_path = 'data/20252026/20252026_df.csv'
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
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
    # We must prep the data exactly like analyze.py does, or better yet, use the model's internal usage
    # The model expects a raw dataframe and handles encoding internally (if configured right)
    
    # IMPORTANT: The model's predict_proba expects specific columns.
    # verify what columns it uses from its internal config
    if hasattr(clf, 'config_block'):
        print(f"Block Model Features: {clf.config_block.feature_cols}")
        print(f"Finish Model Features: {clf.config_finish.feature_cols}")
    
    # Ensure columns exist (handling the analyze.py prep logic check)
    if 'shot_type' not in df_shots.columns:
        df_shots['shot_type'] = 'Unknown'
    df_shots['shot_type'] = df_shots['shot_type'].fillna('Unknown')
    
    if 'game_state' not in df_shots.columns:
         df_shots['game_state'] = '5v5' # default
         
    # Run Prediction
    try:
        # We call the model directly
        probs = clf.predict_proba(df_shots)[:, 1]
        df_shots['xg'] = probs
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
    
    # 5. Layer Analysis
    # We want to see WHICH layer is inflating.
    # We can access the internal models if we are careful.
    
    print("\n--- Layer Diagnostics ---")
    
    # Recalculate encodings manually to feed internal models (since predict_proba does this internally but doesn't return intermediates)
    # We will "monkey patch" or just copy logic from predict_proba to get intermediates
    
    # Helper to recreate internal state
    temp_df = df_shots.copy()
    
    # Encode Shot Type
    if hasattr(clf, 'le_shot'):
        shot_map = dict(zip(clf.le_shot.classes_, clf.le_shot.transform(clf.le_shot.classes_)))
        fallback = shot_map.get('Unknown', -1)
        temp_df['shot_type_encoded'] = temp_df['shot_type'].astype(str).map(shot_map).fillna(fallback).astype(int)
        
    # Encode Game State
    if hasattr(clf, 'le_state'):
        state_map = dict(zip(clf.le_state.classes_, clf.le_state.transform(clf.le_state.classes_)))
        fallback_state = state_map.get('5v5', 0)
        temp_df['game_state_encoded'] = temp_df['game_state'].astype(str).map(state_map).fillna(fallback_state).astype(int)

    # A. Block Model
    p_blocked = clf.model_block.predict_proba(temp_df[clf.config_block.feature_cols])[:, 1]
    df_shots['p_blocked'] = p_blocked
    df_shots['p_unblocked'] = 1 - p_blocked
    
    actual_blocked = (df_shots['event'] == 'blocked-shot').mean()
    pred_blocked = p_blocked.mean()
    print(f"Block Rate: Actual={actual_blocked:.1%}, Predicted={pred_blocked:.1%}")

    # B. Accuracy Model (P(On Net | Unblocked))
    # Evaluate only on unblocked shots for "fair" comparison of what happened
    unblocked_mask = (df_shots['event'] != 'blocked-shot')
    df_unblocked = df_shots[unblocked_mask].copy()
    temp_unblocked = temp_df[unblocked_mask]
    
    p_on_net = clf.model_accuracy.predict_proba(temp_unblocked[clf.config_accuracy.feature_cols])[:, 1]
    
    # "On Net" definitions: Goal or Shot-on-Goal (saved). Miss/Block are 0.
    actual_on_net = df_unblocked['event'].isin(['shot-on-goal', 'goal']).mean()
    pred_on_net = p_on_net.mean()
    print(f"Accuracy (On Net | Unblocked): Actual={actual_on_net:.1%}, Predicted={pred_on_net:.1%}")

    # C. Finish Model (P(Goal | On Net))
    on_net_mask = df_shots['event'].isin(['shot-on-goal', 'goal'])
    df_on_net = df_shots[on_net_mask].copy()
    temp_on_net = temp_df[on_net_mask]
    
    p_goal = clf.model_finish.predict_proba(temp_on_net[clf.config_finish.feature_cols])[:, 1]
    
    actual_finish = (df_on_net['event'] == 'goal').mean()
    pred_finish = p_goal.mean()
    print(f"Finish (Goal | On Net): Actual={actual_finish:.1%}, Predicted={pred_finish:.1%}")

    # 6. Feature Sanity Check
    print("\n--- Feature Stats (All Shots) ---")
    print(df_shots[['distance', 'angle_deg']].describe().to_string())
    
    print("\n--- Top High xG Events (Top 10) ---")
    print(df_shots.sort_values('xg', ascending=False)[['event', 'shot_type', 'distance', 'angle_deg', 'xg']].head(10).to_string())


if __name__ == "__main__":
    main()
