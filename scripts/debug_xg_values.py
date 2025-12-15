
import os
import sys
import pandas as pd
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import timing
from puck import analyze

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def main():
    season = '20252026'
    print(f"Loading data for {season}...")
    df = timing.load_season_df(season)
    
    print(f"Loaded {len(df)} rows.")
    
    # Check for duplicates
    print(f"Unique Game IDs: {df['game_id'].nunique()}")
    
    # Run Prediction
    print("Predicting xG...")
    df_pred, clf, meta = analyze._predict_xgs(df)
    
    # Filter to attempts
    attempts = df_pred[df_pred['event'].isin(['goal', 'shot-on-goal', 'missed-shot', 'blocked-shot'])].copy()
    
    if attempts.empty:
        print("No attempts found.")
        return

    print("\n--- Summary by Event Type ---")
    summary = attempts.groupby('event')['xgs'].agg(['count', 'mean', 'sum', 'min', 'max'])
    print(summary)
    
    print("\n--- Summary by Shot Type ---")
    if 'shot_type' in attempts.columns:
        summary_st = attempts.groupby('shot_type')['xgs'].agg(['count', 'mean'])
        print(summary_st)
        
    print("\n--- Total Season xG ---")
    total_xg = attempts['xgs'].sum()
    total_goals = attempts[attempts['event'] == 'goal'].shape[0]
    print(f"Total xG: {total_xg:.2f}")
    print(f"Total Goals: {total_goals}")
    print(f"Ratio xG/Goals: {total_xg/total_goals:.2f}")
    
    # Check high xG blocked shots
    blocked = attempts[attempts['event'] == 'blocked-shot']
    if not blocked.empty:
        print("\n--- Top 5 xG Blocked Shots ---")
        print(blocked.sort_values('xgs', ascending=False)[['game_id', 'event', 'shot_type', 'distance', 'angle_deg', 'xgs']].head(5))

if __name__ == "__main__":
    main()
