
import sys
import os
import pandas as pd
import json

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import analyze

def main():
    print("--- Debug League Summary ---")
    
    # Run league analysis for one team (e.g. PHI) and all teams to see stats
    # We use mode='compute' to force calculation
    print("Running analyze.league(mode='compute')...")
    
    # We'll just run for one team to save time if possible, but league() iterates all.
    # But league() has a 'teams' arg.
    # Let's run for 'PHI' first.
    
    res = analyze.league(season='20252026', mode='compute', teams=['PHI', 'TOR', 'MTL']) # Sample teams
    
    if not res:
        print("No result from analyze.league")
        return

    print("\n--- Summary Stats ---")
    summary_data = res.get('summary', [])
    
    print(f"{'Team':<5} {'Goals':<10} {'xG':<10} {'Ratio':<10}")
    print("-" * 40)
    
    for row in summary_data:
        team = row.get('team')
        goals = row.get('goals') or 0
        xg = row.get('xgs') or 0.0
        
        ratio = xg / goals if goals > 0 else 0
        print(f"{team:<5} {goals:<10} {xg:<10.2f} {ratio:<10.2f}")
        
    # Deep Dive into One Team (MTL)
    print("\n--- Deep Dive: MTL ---")
    df_seas = analyze.timing.load_season_df('20252026')
    print(f"Season DF Raw: {len(df_seas)} rows")
    
    # Check key columns
    if 'shot_type' in df_seas.columns:
        print(f"shot_type NaNs: {df_seas['shot_type'].isna().sum()} / {len(df_seas)}")
        print(f"shot_type value counts:\n{df_seas['shot_type'].value_counts().head()}")
    else:
        print("shot_type column MISSING")

    # Manually call _predict_xgs
    print("Running _predict_xgs manually...")
    df_pred, clf, meta = analyze._predict_xgs(df_seas.copy())
    
    if 'xgs' in df_pred.columns:
        print(f"xgs NaNs: {df_pred['xgs'].isna().sum()}")
        print(f"xgs > 0: {(df_pred['xgs'] > 0).sum()}")
        print(f"xgs Mean: {df_pred['xgs'].mean()}")
        
        # Check sum for valid shots
        shots = df_pred[df_pred['event'].isin(['goal', 'shot-on-goal', 'missed-shot', 'blocked-shot'])]
        print(f"Total xG (All Shots): {shots['xgs'].sum():.2f}")
    else:
        print("xgs column MISSING after prediction")

