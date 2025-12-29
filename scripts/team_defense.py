"""scripts/team_defense.py

Script to isolate team-level DEFENSIVE talent (and goaltending) using Ridge Regression.
We model: Goal ~ logit(xG) + Defending_Team_Effect
Team_Effect is regularized (shrunk) to separate signal from noise.

INTERPRETATION:
A NEGATIVE coefficient means the team allows FEWER goals than expected given the xG.
This combines structural defense (forcing poor shots) and goaltending (saving good shots).
Note: Since we control for xG (which accounts for shot quality), this metric is HEAVILY influenced by Goaltending.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from scipy.special import logit
import requests

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from puck import fit_xgs, analyze

def main():
    print("--- Team Defense Analysis ---")
    
    # 1. Load Data
    season = "20252026"
    print(f"Loading data for season {season}...")
    try:
        data_path = Path(f"data/{season}/{season}_df.csv")
        if not data_path.exists():
            data_path = Path(f"data/{season}.csv")
            
        if data_path.exists():
            df = fit_xgs.load_data(str(data_path))
        else:
            print("Loading all data...")
            df = fit_xgs.load_data()
            if 'game_id' in df.columns:
                 df['season_start'] = df['game_id'].astype(str).str[:4]
                 df = df[df['season_start'] == season[:4]].copy()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Check for team identifier
    team_col = 'team_id' if 'team_id' in df.columns else 'event_team_id'
    
    # 2. Load xG Model & Predict
    print("Loading xG model...")
    model_path = Path('analysis/xgs/xg_model_nested_all.joblib')
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
        
    print("Generating xG predictions...")
    try:
        df_pred, clf, _ = analyze._predict_xgs(df, model_path=str(model_path))
    except Exception as e:
        print(f"Prediction failed: {e}")
        return
        
    # 3. Prepare Data
    target_df = df_pred.copy()
    if 'is_net_empty' in target_df.columns:
        target_df = target_df[target_df['is_net_empty'] != 1]
    if 'is_goal' not in target_df.columns:
        target_df['is_goal'] = (target_df['event'] == 'goal').astype(int)
    
    target_df = target_df.dropna(subset=['xgs', team_col, 'is_goal'])
    
    # 4. Feature Engineering
    target_df['xgs_clipped'] = target_df['xgs'].clip(1e-5, 1 - 1e-5)
    target_df['logit_xgs'] = logit(target_df['xgs_clipped'])
    
    # --- DEFENSIVE TEAM ATTRIBUTION ---
    # We want to identify the DEFENDING TEAM for each event.
    
    print("Determining Defending Team...")
    if 'home_id' in target_df.columns and 'away_id' in target_df.columns:
        target_df['defending_team_id'] = np.nan
        
        # Case A: Blocked Shot (Event Owner = Blocker = Defense)
        # So for blocked shots, defending_team_id = team_col
        mask_blocked = target_df['event'] == 'blocked-shot'
        target_df.loc[mask_blocked, 'defending_team_id'] = target_df.loc[mask_blocked, team_col]
        
        # Case B: Shot/Goal/Miss (Event Owner = Shooter = Offense)
        # So defending_team is the Opponent.
        mask_offense = ~mask_blocked
        
        # Where Offense is Home -> Defense is Away
        mask_home_off = mask_offense & (target_df[team_col] == target_df['home_id'])
        target_df.loc[mask_home_off, 'defending_team_id'] = target_df.loc[mask_home_off, 'away_id']
        
        # Where Offense is Away -> Defense is Home
        mask_away_off = mask_offense & (target_df[team_col] == target_df['away_id'])
        target_df.loc[mask_away_off, 'defending_team_id'] = target_df.loc[mask_away_off, 'home_id']
        
        # Fill any remaining NaNs (if team_id matches neither home nor away?)
        
        target_df['defending_team_id'] = target_df['defending_team_id'].fillna(-1)
        analysis_team_col = 'defending_team_id'
    else:
        print("Error: detailed home/away info missing. Cannot determine opponents.")
        return

    # One-Hot Encode DEFENDING Teams
    print("Encoding team features...")
    team_dummies = pd.get_dummies(target_df[analysis_team_col], prefix='team')
    
    # 5. Ridge Regression
    X = pd.concat([target_df[['logit_xgs']], team_dummies], axis=1)
    y = target_df['is_goal']
    
    valid_idx = X.notna().all(axis=1) & y.notna() & (target_df[analysis_team_col] != -1)
    X = X[valid_idx]
    y = y[valid_idx]
    target_df = target_df[valid_idx].copy()
    
    print(f"Fitting Ridge Regression on {len(X)} shots...")
    lr = LogisticRegression(penalty='l2', C=0.1, solver='lbfgs', max_iter=2000)
    lr.fit(X, y)
    
    # 6. Extract Results
    coefs = dict(zip(X.columns, lr.coef_[0]))
    intercept = lr.intercept_[0]
    
    # Extract Team Effects
    team_effects = {k.replace('team_', ''): v for k, v in coefs.items() if k.startswith('team_')}
    res_df = pd.DataFrame(list(team_effects.items()), columns=['TeamID', 'TeamEffect_LogOdds'])
    
    # Stats (Group by Defending Team)
    stats = target_df.groupby(analysis_team_col).agg({
        'is_goal': ['sum', 'count'],
        'xgs': ['sum']
    })
    stats.columns = ['Goals_Allowed', 'Shots_Against', 'xG_Against']
    
    # "Goals Saved" = xG - Goals (Positive is Good)
    stats['Goals_Saved_Above_xG'] = stats['xG_Against'] - stats['Goals_Allowed']
    stats['GSAx_per_100'] = (stats['Goals_Saved_Above_xG'] / stats['Shots_Against']) * 100
    
    # Ensure merge keys match types
    # res_df['TeamID'] comes from string column names "team_12" -> "12"
    # stats index is likely float or Int from dataframe
    
    res_df['TeamID'] = res_df['TeamID'].astype(str).str.replace(r'\.0$', '', regex=True)
    stats.index = stats.index.astype(str).str.replace(r'\.0$', '', regex=True)
    
    final_df = stats.merge(res_df, left_index=True, right_on='TeamID')
    
    # FETCH MAPPING
    try:
        resp = requests.get('https://api.nhle.com/stats/rest/en/team')
        teams_data = resp.json().get('data', [])
        # TeamID in final_df is STRING (from stats.index cleanup)
        team_map = {str(t['id']): t['triCode'] for t in teams_data if 'triCode' in t}
    except:
        team_map = {}
        
    final_df['TeamAbbrev'] = final_df['TeamID'].map(team_map).fillna(final_df['TeamID'].astype(str))
    
    # Sorting:
    # For Defense, we want "Goals Saved" (Positive) or "Effect" (Negative).
    # Effect is "Impact on Goal Prob". Negative = Good.
    # Let's sort by Effect Ascending (Most Negative = Best Defense first)
    final_df = final_df.sort_values('TeamEffect_LogOdds', ascending=True)
    
    print("\nTop 5 Defensive Teams (Best Suppression/Saves):")
    print(final_df[['TeamAbbrev', 'Goals_Allowed', 'xG_Against', 'GSAx_per_100', 'TeamEffect_LogOdds']].head())
    
    print("\nBottom 5 Defensive Teams (Worst Suppression/Saves):")
    print(final_df[['TeamAbbrev', 'Goals_Allowed', 'xG_Against', 'GSAx_per_100', 'TeamEffect_LogOdds']].tail())
    
    # 7. Visualize
    plt.figure(figsize=(12, 8))
    
    # X: GSAx per 100 (Positive = Good)
    # Y: Effect (Negative = Good)
    # To make the plot intuitive (Top Right = Good), let's invert Y axis? 
    # Or plot -Effect on Y.
    
    final_df['Defensive_Talent_Score'] = -1 * final_df['TeamEffect_LogOdds']
    
    sns.scatterplot(data=final_df, x='GSAx_per_100', y='Defensive_Talent_Score', size='Shots_Against', sizes=(50, 400), alpha=0.7)
    
    for i, row in final_df.iterrows():
        plt.text(row['GSAx_per_100'], row['Defensive_Talent_Score'], str(row['TeamAbbrev']), fontsize=9, fontweight='bold')
        
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.title('Team Defensive Talent vs Raw Results (Shrinkage Analysis)')
    plt.xlabel('Goals Saved Above Expected per 100 Shots (Raw)')
    plt.ylabel('Adjusted Defensive Talent (Log Odds, Inverted)')
    plt.grid(True, alpha=0.3)
    
    out_path = Path('analysis/team_defense_ridge.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")
    
    try:
        csv_path = Path('analysis/team_defense_results.csv')
        final_df.to_csv(csv_path, index=False)
    except:
        csv_path = Path('analysis/team_defense_results_safe.csv')
        final_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    main()
