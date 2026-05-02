import pandas as pd
import joblib
import os
import sys
import numpy as np
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puck import analyze

def calculate_team_rates():
    model_path = 'analysis/xgs/xg_model_xgboost_tensor_modern_era.joblib'
    season = '20252026'
    teams = ['CAR', 'PHI']
    
    path = analyze.locate_season_csv(season)
    df_raw = pd.read_csv(path)
    
    # Predict xGs with NEW model
    print(f"Predicting xGs for {season} with NEW model...")
    df, _, _ = analyze._predict_xgs(df_raw, model_path=model_path)
    
    print(f"Columns after prediction: {df.columns.tolist()[:10]}...")
    
    results = {}
    # Use robust is_home derivation if missing
    if 'is_home' not in df.columns:
        print("Deriving is_home manually...")
        tid_s = df['team_id'].astype(str).str.replace(r'\.0$', '', regex=True)
        hid_s = df['home_id'].astype(str).str.replace(r'\.0$', '', regex=True)
        df['is_home'] = (tid_s == hid_s).astype(int)

    df_5v5 = df[(df['game_state'] == '5v5')].copy()
    
    # Identify shooting team
    df_5v5['shooting_team'] = np.where(df_5v5['is_home'] == 1, df_5v5['home_abb'], df_5v5['away_abb'])

    for team in teams:
        # Team Offense
        df_off = df_5v5[df_5v5['shooting_team'] == team]
        
        # Team Defense
        df_def = df_5v5[((df_5v5['home_abb'] == team) | (df_5v5['away_abb'] == team)) & (df_5v5['shooting_team'] != team)]
        
        # Seconds (CAR: 189092s, PHI: 174959s)
        seconds = {'CAR': 189092, 'PHI': 174959}
        
        goals_for = df_off[df_off['event'] == 'goal'].shape[0]
        xg_for = df_off['xgs'].sum()
        
        goals_against = df_def[df_def['event'] == 'goal'].shape[0]
        xg_against = df_def['xgs'].sum()
        
        hrs = seconds[team] / 3600.0
        
        results[team] = {
            'GF60': round(goals_for / hrs, 3),
            'GA60': round(goals_against / hrs, 3),
            'xGF60': round(xg_for / hrs, 3),
            'xGA60': round(xg_against / hrs, 3)
        }

    print("\n5v5 Rates (per 60) - 2025-2026 Season (NEW MODEL):")
    for team, stats in results.items():
        print(f"{team}:")
        print(f"  Goals: For={stats['GF60']}, Against={stats['GA60']}")
        print(f"  xG:    For={stats['xGF60']}, Against={stats['xGA60']}")
        print(f"  Ratio (G/xG) Off: {stats['GF60']/stats['xGF60']:.3f}")
        print(f"  Ratio (G/xG) Def: {stats['GA60']/stats['xGA60']:.3f}")

if __name__ == "__main__":
    calculate_team_rates()
