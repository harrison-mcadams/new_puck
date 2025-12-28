
import sys
import os
import pandas as pd
import numpy as np
import joblib

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import fit_nested_xgs
from puck import impute

def main():
    print("--- Calculating League xG/60 ---")
    
    # 1. Load Data
    csv_path = 'data/20252026.csv'
    if not os.path.exists(csv_path):
        print("Data file not found.")
        return
        
    df = pd.read_csv(csv_path)
    n_games = df['game_id'].nunique()
    print(f"Loaded {len(df)} rows from {n_games} games.")
    
    # 2. Filter to Shots
    events = ['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot']
    df_shots = df[df['event'].isin(events)].copy()
    
    # 3. Load Model
    model_path = 'analysis/xgs/xg_model_nested.joblib'
    clf = joblib.load(model_path)
    
    # 4. Impute & Predict
    if 'game_state' not in df_shots.columns:
        df_shots['game_state'] = '5v5'
        
    df_shots = impute.impute_blocked_shot_origins(df_shots, method='point_pull')
    
    try:
        probs = clf.predict_proba(df_shots)[:, 1]
        df_shots['xg'] = probs
    except Exception as e:
        print(f"Prediction error: {e}")
        return
        
    # 5. Calculate Rates
    total_xg = df_shots['xg'].sum()
    total_goals = df_shots[df_shots['event'] == 'goal'].shape[0]
    
    # Estimate Time
    # We can perform a rough estimate: N_Games * 60 mins * 2 teams? 
    # Or just "Per Game" metric.
    # To be more precise, we could check max time per game, but standard "xG/60" usually assumes 60 min games for normalization unless we have exact TOI.
    # Let's use the number of games.
    
    # Total League Minutes = n_games * 60 (approx duration of play)
    # Total Team Minutes = n_games * 60 * 2
    
    avg_xg_per_game_combined = total_xg / n_games
    avg_xg_per_60_team = (total_xg / n_games) / 2
    
    avg_goals_per_60_team = (total_goals / n_games) / 2
    
    print("\nResults:")
    print(f"Total xG: {total_xg:.2f}")
    print(f"Total Goals: {total_goals}")
    print(f"Unique Games: {n_games}")
    print("-" * 20)
    print(f"League Average xG/60 (per team): {avg_xg_per_60_team:.3f}")
    print(f"League Average Goals/60 (per team): {avg_goals_per_60_team:.3f}")
    print(f"Ratio: {total_xg/total_goals:.3f}")

if __name__ == "__main__":
    main()
