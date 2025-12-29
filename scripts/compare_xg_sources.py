
import pandas as pd
import numpy as np
import os
import sys
import requests
import io
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from puck import config

def main():
    season = "20252026"  # Hardcoded for now, or match existing
    
    # 1. Load Local Data (Mine)
    # -------------------------
    local_path = os.path.join(config.ANALYSIS_DIR, 'league', season, '5v5', f'{season}_team_summary.csv')
    if not os.path.exists(local_path):
        print(f"Error: Local summary not found at {local_path}")
        print("Please run scripts/run_league_stats.py --season 20252026 --condition 5v5 first.")
        return

    df_local = pd.read_csv(local_path)
    # Expected columns: team, team_xg_per60, other_xg_per60, team_xgs, team_seconds ...
    
    print(f"Loaded Local Data: {len(df_local)} teams.")

    # 2. Load MoneyPuck Data
    # ----------------------
    mp_url = "https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/teams.csv"
    print(f"Downloading MoneyPuck data from {mp_url}...")
    
    try:
        r = requests.get(mp_url)
        r.raise_for_status()
        df_mp = pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        print(f"Failed to download MoneyPuck data: {e}")
        return

    print(f"Loaded MoneyPuck Data: {len(df_mp)} rows.")
    
    # Filter MoneyPuck for 5v5
    # MoneyPuck columns: team, situation, xGoalsFor, xGoalsAgainst, iceTime ...
    # situation == '5on5'
    
    if 'situation' in df_mp.columns:
        df_mp_5v5 = df_mp[df_mp['situation'] == '5on5'].copy()
    else:
        print("Warning: 'situation' column not found in MoneyPuck data. Available columns:")
        print(df_mp.columns.tolist())
        return

    # Calculate Rates for MoneyPuck (per 60)
    # iceTime is usually in seconds?
    # Let's verify. usually it's correct.
    # MoneyPuck xGoalsPercentage is also available.
    
    # Normalize Team Names
    # MoneyPuck uses ARI/UTA? 
    # Let's check unique teams
    # print(df_mp_5v5['team'].unique())
    
    # Rename columns for merge
    df_mp_5v5['mp_xg_for_per_60'] = (df_mp_5v5['xGoalsFor'] / df_mp_5v5['iceTime']) * 3600
    df_mp_5v5['mp_xg_against_per_60'] = (df_mp_5v5['xGoalsAgainst'] / df_mp_5v5['iceTime']) * 3600
    
    df_mp_clean = df_mp_5v5[['team', 'mp_xg_for_per_60', 'mp_xg_against_per_60', 'xGoalsFor', 'xGoalsAgainst', 'goalsFor', 'goalsAgainst']].copy()
    df_mp_clean.rename(columns={'team': 'team_abbr'}, inplace=True)
    
    # Local data has 'team' as abbr
    df_local.rename(columns={'team': 'team_abbr'}, inplace=True)

    # Merge
    merged = pd.merge(df_local, df_mp_clean, on='team_abbr', how='inner', suffixes=('_my', '_mp'))
    
    if merged.empty:
        print("Error: Merge resulted in empty DataFrame. Check team abbreviations.")
        print(f"Local Teams: {df_local['team_abbr'].unique()}")
        print(f"MP Teams: {df_mp_clean['team_abbr'].unique()}")
        return

    print(f"Merged {len(merged)} teams.")

    # 3. Compare
    # ----------
    
    # Correlations
    corr_for = merged['team_xg_per60'].corr(merged['mp_xg_for_per_60'])
    corr_against = merged['other_xg_per60'].corr(merged['mp_xg_against_per_60'])
    
    print("\n--- Correlation Analysis (5v5 xG Rate) ---")
    print(f"xG For Correlation:      {corr_for:.4f}")
    print(f"xG Against Correlation:  {corr_against:.4f}")
    
    # High Level Stats
    print("\n--- League Totals (5v5) ---")
    print(f"My Total xG For:   {merged['team_xgs'].sum():.1f}")
    print(f"MP Total xG For:   {merged['xGoalsFor'].sum():.1f}")
    print(f"My Total Goals:    {merged['team_goals'].sum()}")
    print(f"MP Total Goals:    {merged['goalsFor'].sum()}")
    
    ratio_my = merged['team_xgs'].sum() / merged['team_goals'].sum()
    ratio_mp = merged['xGoalsFor'].sum() / merged['goalsFor'].sum()
    print(f"My xG/Goal Ratio:  {ratio_my:.3f}")
    print(f"MP xG/Goal Ratio:  {ratio_mp:.3f}")

    # Differences Table
    merged['diff_xg_for'] = merged['team_xg_per60'] - merged['mp_xg_for_per_60']
    merged['diff_xg_against'] = merged['other_xg_per60'] - merged['mp_xg_against_per_60']
    
    print("\n--- Largest Differences (per 60) ---")
    print("Positive = My Model is Higher")
    
    cols_show = ['team_abbr', 'team_xg_per60', 'mp_xg_for_per_60', 'diff_xg_for']
    print("\nTop 5 Overestimates (xG For):")
    print(merged.sort_values('diff_xg_for', ascending=False).head(5)[cols_show].to_string(index=False))
    
    print("\nTop 5 Underestimates (xG For):")
    print(merged.sort_values('diff_xg_for', ascending=True).head(5)[cols_show].to_string(index=False))
    
    # Scatter Plot
    out_dir = os.path.dirname(local_path)
    plot_path = os.path.join(out_dir, 'comparison_moneypuck_5v5.png')
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=merged, x='mp_xg_for_per_60', y='team_xg_per60')
    
    # x=y line
    min_val = min(merged['mp_xg_for_per_60'].min(), merged['team_xg_per60'].min())
    max_val = max(merged['mp_xg_for_per_60'].max(), merged['team_xg_per60'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    for i in range(merged.shape[0]):
        plt.text(merged.mp_xg_for_per_60.iloc[i]+0.01, merged.team_xg_per60.iloc[i], 
                 merged.team_abbr.iloc[i], fontsize=9)
                 
    plt.title(f"xG For/60: My Model vs MoneyPuck (5v5) | r={corr_for:.3f}")
    plt.xlabel("MoneyPuck xG/60")
    plt.ylabel("My Model xG/60")
    plt.grid(True, alpha=0.3)
    plt.savefig(plot_path)
    print(f"\nSaved Comparison Plot to: {plot_path}")

if __name__ == "__main__":
    main()
