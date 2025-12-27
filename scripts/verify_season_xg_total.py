import sys
import os
import pandas as pd
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import analyze
from puck import fit_xgs

def main():
    season = '20252026'
    print(f"--- Verifying Season Totals for {season} ---")
    
    # 1. Load Data
    # We can use fit_xgs loader or parse
    # fit_xgs.load_all_seasons_data loads EVERYTHING.
    # We want just 20252026.
    # Let's verify file existence first.
    
    csv_path = os.path.join('data', season, f"{season}_df.csv")
    if not os.path.exists(csv_path):
        # Fallback
        csv_path = os.path.join('data', f"{season}.csv")
        
    print(f"Loading {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to load: {e}")
        # Try fit_xgs loader as fallback
        print("Trying fit_xgs.load_all_seasons_data...")
        df = fit_xgs.load_all_seasons_data()
        df = df[df['season'] == int(season)].copy()
        
    print(f"Loaded {len(df)} rows.")
    
    # 1b. Filter for regular season
    if 'game_id' in df.columns:
        df['game_id_str'] = df['game_id'].astype(str)
        # Game IDs are YYYYTTNNNN, where TT=02 is regular season.
        mask_regular = (df['game_id_str'].str.len() >= 6) & (df['game_id_str'].str[4:6] == '02')
        if not mask_regular.all():
            print(f"Filtering {len(df) - mask_regular.sum()} non-regular season rows.")
            df = df[mask_regular].copy()
        df.drop(columns=['game_id_str'], inplace=True)

    # 2. Predict
    print("Running Nested xG Prediction (this may take a minute)...")
    df_pred, _, _ = analyze._predict_xgs(df, behavior='overwrite')
    
    # 3. Aggregates
    if 'xgs' not in df_pred.columns:
        print("Error: xgs column missing.")
        return
        
    # Valid events only just in case
    valid = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
    df_valid = df_pred[df_pred['event'].isin(valid)].copy()
    
    total_goals = (df_valid['event'] == 'goal').sum()
    total_xg = df_valid['xgs'].sum()
    
    print("\n" + "="*40)
    print(f"SEASON TOTALS ({season})")
    print("="*40)
    print(f"Total Goals:      {total_goals}")
    print(f"Total xG:         {total_xg:.2f}")
    print(f"Difference:       {total_xg - total_goals:.2f}")
    if total_goals > 0:
        print(f"Ratio (xG/Goals): {total_xg / total_goals:.3f}")
    else:
        print("Ratio: N/A")
        
    print("\n--- Breakdown by Event Type ---")
    stats = df_valid.groupby('event').agg(
        count=('xgs', 'count'),
        total_xg=('xgs', 'sum'),
        avg_xg=('xgs', 'mean')
    ).sort_values('total_xg', ascending=False)
    print(stats)
    
if __name__ == "__main__":
    main()
