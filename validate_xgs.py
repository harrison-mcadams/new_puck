
import pandas as pd
import numpy as np
import analyze
import fit_xgs
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import requests
import json
import os
import time

def get_moneypuck_data():
    """Download and prepare MoneyPuck team data."""
    url = "https://moneypuck.com/moneypuck/playerData/seasonSummary/2024/regular/teams.csv"
    print(f"Downloading MoneyPuck data from {url}...")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        return df
    except Exception as e:
        print(f"Error downloading MoneyPuck data: {e}")
        return None

def run_validation(season='20242025', limit_teams=None):
    # 1. Get MoneyPuck Data
    mp_df = get_moneypuck_data()
    if mp_df is None:
        return

    # Filter for 5v5
    if 'situation' in mp_df.columns:
        mp_5v5 = mp_df[mp_df['situation'] == '5on5'].copy()
    else:
        print("Could not find 'situation' column in MoneyPuck data. Columns:", mp_df.columns)
        return

    # Map MoneyPuck team names
    team_map = {
        'S.J': 'SJS', 'T.B': 'TBL', 'N.J': 'NJD', 'L.A': 'LAK', 'MTL': 'MTL'
    }
    mp_5v5['team'] = mp_5v5['team'].replace(team_map)
    
    # 2. Prepare Local Analysis
    print(f"Preparing local analysis for season {season}...")
    
    # Load/Train model once
    print("Loading/Training xG model...")
    fit_xgs.get_or_train_clf(csv_path=f'data/{season}/{season}_df.csv')
    
    # Get team list
    with open('static/teams.json', 'r') as f:
        teams_data = json.load(f)
    all_teams = [t['abbr'] for t in teams_data]
    
    if limit_teams:
        all_teams = all_teams[:limit_teams]
        print(f"Limiting analysis to first {limit_teams} teams: {all_teams}")

    comparison_rows = []
    
    # 3. Loop through teams
    condition = {'game_state': ['5v5'], 'is_net_empty': [0]}
    
    for i, team in enumerate(all_teams):
        print(f"Processing team {i+1}/{len(all_teams)}: {team}...")
        t0 = time.time()
        
        try:
            # Set team condition
            team_cond = condition.copy()
            team_cond['team'] = team
            
            # Call xgs_map
            # We use return_heatmaps=False to speed up if possible, but we need summary_stats.
            # analyze.xgs_map returns (out_path, ret_heat, ret_df, summary_stats)
            # We can set heatmap_only=True maybe? No, that returns arrays.
            # We'll just run it.
            _, _, _, summary_stats = analyze.xgs_map(
                season=season,
                condition=team_cond,
                show=False,
                return_heatmaps=True, # Need this to get stats?
                return_filtered_df=False
            )
            
            local_xg = float(summary_stats.get('team_xgs', 0.0))
            local_sec = float(summary_stats.get('team_seconds', 0.0))
            
            # Get MoneyPuck stats
            mp_team_row = mp_5v5[mp_5v5['team'] == team]
            if not mp_team_row.empty:
                mp_xg = mp_team_row['xGoalsFor'].values[0]
                mp_sec = mp_team_row['iceTime'].values[0]
                
                comparison_rows.append({
                    'team': team,
                    'local_xg': local_xg,
                    'mp_xg': mp_xg,
                    'local_sec': local_sec,
                    'mp_sec': mp_sec,
                    'local_xg_rate': (local_xg / local_sec) * 3600 if local_sec > 0 else 0,
                    'mp_xg_rate': (mp_xg / mp_sec) * 3600 if mp_sec > 0 else 0
                })
                
            print(f"  Done in {time.time()-t0:.1f}s. Local xG: {local_xg:.1f}, MP xG: {mp_xg:.1f}")
            
        except Exception as e:
            print(f"  Error processing {team}: {e}")
            import traceback
            traceback.print_exc()

    # 4. Analyze Results
    comp_df = pd.DataFrame(comparison_rows)
    if comp_df.empty:
        print("No results to compare.")
        return

    print("\n--- Comparison Summary (5v5) ---")
    print(comp_df[['team', 'local_xg', 'mp_xg', 'local_sec', 'mp_sec']].to_string())
    
    # Correlations
    corr_xg = comp_df['local_xg'].corr(comp_df['mp_xg'])
    corr_sec = comp_df['local_sec'].corr(comp_df['mp_sec'])
    corr_rate = comp_df['local_xg_rate'].corr(comp_df['mp_xg_rate'])
    
    print(f"\nCorrelations:")
    print(f"  Total xG: {corr_xg:.4f}")
    print(f"  Total Seconds: {corr_sec:.4f}")
    print(f"  xG Rate (per 60): {corr_rate:.4f}")
    
    # Save
    comp_df.to_csv('static/validation_comparison.csv', index=False)
    
    # Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=comp_df, x='mp_xg', y='local_xg')
    # Add diagonal line
    max_val = max(comp_df['mp_xg'].max(), comp_df['local_xg'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
    plt.title(f'Total xG (r={corr_xg:.3f})')
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=comp_df, x='mp_xg_rate', y='local_xg_rate')
    max_rate = max(comp_df['mp_xg_rate'].max(), comp_df['local_xg_rate'].max())
    plt.plot([0, max_rate], [0, max_rate], 'r--', alpha=0.5)
    plt.title(f'xG Rate/60 (r={corr_rate:.3f})')
    
    plt.tight_layout()
    plt.savefig('static/validation_plot.png')
    print("\nSaved plot to static/validation_plot.png")

if __name__ == "__main__":
    # Run for a few teams to demonstrate
    print("Running validation for first 5 teams (demo mode)...")
    run_validation(limit_teams=5)
