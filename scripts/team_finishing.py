"""scripts/team_finishing.py

Script to isolate team-level finishing talent using Ridge Regression.
We model: Goal ~ logit(xG) + Team_Effect
Team_Effect is regularized (shrunk) to separate signal from noise.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from scipy.special import logit, expit
import joblib

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from puck import fit_xgs, analyze

def main():
    print("--- Team Finishing Analysis ---")
    
    # 1. Load Data
    # Load specific season (2025-2026)
    season = "20252026"
    print(f"Loading data for season {season}...")
    try:
        # Construct path to specific season file
        data_path = Path(f"data/{season}/{season}_df.csv")
        if not data_path.exists():
            # Fallback to loose file
            data_path = Path(f"data/{season}.csv")
            
        if data_path.exists():
            df = fit_xgs.load_data(str(data_path))
        else:
            print(f"Analysis limited to {season}, but file not found at {data_path}. Loading all and filtering...")
            df = fit_xgs.load_data()
            # Filter by GameID if possible, or just proceed if we can't
            if 'game_id' in df.columns:
                 # GameID starts with YYYY...
                 df['season_start'] = df['game_id'].astype(str).str[:4]
                 target_start = season[:4]
                 df = df[df['season_start'] == target_start].copy()
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Check for team identifier
    # Usually 'team_id' or 'event_team_id'. Let's find it.
    team_col = None
    if 'team_id' in df.columns:
        team_col = 'team_id'
    elif 'event_team_id' in df.columns:
        team_col = 'event_team_id'
    
    if not team_col:
        print("Error: Could not find 'team_id' column in dataframe.")
        print("Columns:", df.columns.tolist())
        return
        
    print(f"Using team column: {team_col}")
    
    # 2. Load xG Model & Predict
    print("Loading xG model...")
    model_path = Path('analysis/xgs/xg_model_nested_all.joblib')
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}. Please train it first.")
        return
        
    # We use analyze._predict_xgs properly
    # It handles preprocessing internally
    print("Generating xG predictions...")
    try:
        # returns df_pred, model, meta
        df_pred, clf, _ = analyze._predict_xgs(df, model_path=str(model_path))
    except Exception as e:
        print(f"Prediction failed: {e}")
        return
        
    # 3. Prepare Data for Team Effect Model
    # Filter for standard play (no empty net) - analyze._predict_xgs might not filter empty net 
    # but cleaning usually does.
    
    target_df = df_pred.copy()
    
    # Filter: Regular Season 5v5/5v4 etc? Or All?
    # Let's stick to standard shots (no empty net)
    if 'is_net_empty' in target_df.columns:
        target_df = target_df[target_df['is_net_empty'] != 1]
        
    # Ensure is_goal exists
    if 'is_goal' not in target_df.columns and 'event' in target_df.columns:
        target_df['is_goal'] = (target_df['event'] == 'goal').astype(int)
    
    # Drop rows without xG or Team
    target_df = target_df.dropna(subset=['xgs', team_col, 'is_goal'])
    
    # 4. Feature Engineering
    # Logit of xG for the offset
    # Clip to avoid inf
    target_df['xgs_clipped'] = target_df['xgs'].clip(1e-5, 1 - 1e-5)
    target_df['logit_xgs'] = logit(target_df['xgs_clipped'])
    
    # --- FIX TEAM ATTRIBUTION ---
    # For blocked shots, 'team_id' is the BLOCKING team (defense).
    # We want Offensive Finishing, so we must attribute these to the SHOOTING (opposing) team.
    
    print("Adjusting team attribution for blocked shots...")
    # Check if we have home/away columns to derive opponent
    if 'home_id' in target_df.columns and 'away_id' in target_df.columns:
        # Create a 'shooting_team_id' column
        # Default to team_id
        target_df['shooting_team_id'] = target_df[team_col]
        
        # Identify blocked shots
        mask_blocked = target_df['event'] == 'blocked-shot'
        
        # For blocked shots: if team_id == home_id, shooter is away_id, and vice versa
        # We can vectorise this:
        # Where (Blocked & Team==Home) -> Away
        # Where (Blocked & Team==Away) -> Home
        
        # Ensure IDs are comparable (float/int issues)
        # Convert all to numeric errors='coerce' first if needed, but assuming ID consistency
        
        mask_home_block = mask_blocked & (target_df[team_col] == target_df['home_id'])
        mask_away_block = mask_blocked & (target_df[team_col] == target_df['away_id'])
        
        target_df.loc[mask_home_block, 'shooting_team_id'] = target_df.loc[mask_home_block, 'away_id']
        target_df.loc[mask_away_block, 'shooting_team_id'] = target_df.loc[mask_away_block, 'home_id']
        
        print(f"  Swapped {mask_blocked.sum()} blocked shots to shooting team.")
        
        # Use new column for analysis
        analysis_team_col = 'shooting_team_id'
    else:
        print("Warning: 'home_id'/'away_id' not found. attributing blocked shots to recorded team (likely blocker). Analysis may be skewed.")
        analysis_team_col = team_col

    # One-Hot Encode Teams
    # We want strict control, so let's use sklearn OneHotEncoder or simple pd.get_dummies
    # pd.get_dummies is easier for inspection
    print("Encoding team features...")
    team_dummies = pd.get_dummies(target_df[analysis_team_col], prefix='team')
    
    # 5. Ridge Regression (L2 Logistic)
    # Target: is_goal
    # Features: logit_xgs (we want this coef to be ~1) + Team Dummies (regularized)
    
    X = pd.concat([target_df[['logit_xgs']], team_dummies], axis=1)
    y = target_df['is_goal']
    
    # Filter out NaNs if any created
    valid_idx = X.notna().all(axis=1) & y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    target_df = target_df[valid_idx].copy() # keep aligned for stats
    
    print(f"Fitting Ridge Regression on {len(X)} shots...")
    # C=0.1 means specific strength of regularization. 
    # We let sklearn penalize all, including logit_xgs. 
    # Given the sample size (thousands), logit_xgs won't be shrunk much if it's strong signal.
    # Team effects (smaller N per team) will be shrunk.
    
    lr = LogisticRegression(penalty='l2', C=0.1, solver='lbfgs', max_iter=2000)
    lr.fit(X, y)
    
    # 6. Extract Results
    coefs = dict(zip(X.columns, lr.coef_[0]))
    intercept = lr.intercept_[0]
    
    print("\n--- Model Results ---")
    print(f"Base Intercept: {intercept:.4f}")
    if 'logit_xgs' in coefs:
        print(f"logit(xG) Coefficient: {coefs['logit_xgs']:.4f} (Should be close to 1.0)")
        
    # Extract Team Effects
    team_effects = {k.replace('team_', ''): v for k, v in coefs.items() if k.startswith('team_')}
    
    # Convert to easier format
    res_df = pd.DataFrame(list(team_effects.items()), columns=['TeamID', 'TeamEffect_LogOdds'])
    
    # Add simple raw stats for context
    # Group by team - use the analysis column
    stats = target_df.groupby(analysis_team_col).agg({
        'is_goal': ['sum', 'count', 'mean'],
        'xgs': ['sum', 'mean']
    })
    stats.columns = ['Goals', 'Shots', 'Sh%', 'xG_Sum', 'xG_Avg']
    stats['Goals_Above_xG'] = stats['Goals'] - stats['xG_Sum']
    stats['GAx_per_100'] = (stats['Goals_Above_xG'] / stats['Shots']) * 100
    
    # Merge
    # TeamID in res_df might be int/str, ensure match
    if pd.api.types.is_numeric_dtype(target_df[team_col]):
        try:
            res_df['TeamID'] = res_df['TeamID'].astype(target_df[team_col].dtype)
        except:
             # if team id had missing values or weird types
             pass
        
    import requests
    
    # --- FETCH TEAM MAPPING ---
    print("Fetching team mapping...")
    try:
        resp = requests.get('https://api.nhle.com/stats/rest/en/team')
        resp.raise_for_status()
        teams_data = resp.json().get('data', [])
        # Map ID (int) -> triCode (str)
        # Handle cases where ID might be string in JSON but int in DF
        team_map = {int(t['id']): t['triCode'] for t in teams_data if 'triCode' in t}
        print(f"Loaded {len(team_map)} teams.")
    except Exception as e:
        print(f"Warning: Could not fetch team mapping: {e}")
        team_map = {}
    
    final_df = stats.merge(res_df, left_index=True, right_on='TeamID')
    
    # Apply Mapping
    final_df['TeamAbbrev'] = final_df['TeamID'].map(team_map)
    # Fill missing with ID
    final_df['TeamAbbrev'] = final_df['TeamAbbrev'].fillna(final_df['TeamID'].astype(str))
    
    # Sort by Team Effect (Talent)
    final_df = final_df.sort_values('TeamEffect_LogOdds', ascending=False)
    
    print("\nTop 5 Finishing Teams (Adjusted):")
    print(final_df[['TeamID', 'TeamAbbrev', 'Goals', 'xG_Sum', 'GAx_per_100', 'TeamEffect_LogOdds']].head())
    
    print("\nBottom 5 Finishing Teams (Adjusted):")
    print(final_df[['TeamID', 'TeamAbbrev', 'Goals', 'xG_Sum', 'GAx_per_100', 'TeamEffect_LogOdds']].tail())
    
    # 7. Visualize
    plt.figure(figsize=(12, 8))
    
    sns.scatterplot(data=final_df, x='GAx_per_100', y='TeamEffect_LogOdds', size='Shots', sizes=(50, 400), alpha=0.7)
    
    # Annotate top/bottom
    for i, row in final_df.head(5).iterrows():
        plt.text(row['GAx_per_100'], row['TeamEffect_LogOdds'], str(row['TeamAbbrev']), fontsize=9, fontweight='bold')
    for i, row in final_df.tail(5).iterrows():
        plt.text(row['GAx_per_100'], row['TeamEffect_LogOdds'], str(row['TeamAbbrev']), fontsize=9, fontweight='bold')
    # Also annotate outliers? 
    # Let's annotate ALL if reasonable (32 teams is fine) or just outliers.
    # User might want to see specific teams. Let's do all for now but small font?
    # Or just top/bottom 5 as before but with Abbrev.
        
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.title('Team Finishing Talent vs Raw Results (Shrinkage Analysis) - ' + str(season))
    plt.xlabel('Raw Goals Above Average per 100 Shots')
    plt.ylabel('Adjusted Team Finishing Effect (Log Odds)')
    
    out_path = Path('analysis/team_finishing_ridge.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")
    
    # Save CSV
    try:
        csv_path = Path('analysis/team_finishing_results.csv')
        final_df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
    except PermissionError:
        csv_path = Path('analysis/team_finishing_results_offense.csv')
        final_df.to_csv(csv_path, index=False)
        print(f"Locked file. Saved results to {csv_path}")

if __name__ == "__main__":
    main()
