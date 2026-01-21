
import pandas as pd
import json
import os
import numpy as np

# Load CSV
csv_path = 'data/20252026/20252026.csv'
if not os.path.exists(csv_path):
    csv_path = 'data/20252026.csv' 
if not os.path.exists(csv_path):
    import glob
    files = glob.glob('data/**/*20252026*.csv', recursive=True)
    if files:
        csv_path = files[0]

print(f"Loading CSV from {csv_path}")
try:
    df = pd.read_csv(csv_path, low_memory=False)
except Exception as e:
    print(f"Failed to load CSV: {e}")
    exit()

# Load Cached Summary (Shift-based)
json_path = 'analysis/league/20252026/5v5/20252026_team_summary.json'
if not os.path.exists(json_path):
    json_path = 'analysis/league/20252026/5v5/team_summary.json'

print(f"Loading JSON from {json_path}")
try:
    with open(json_path, 'r') as f:
        summary = json.load(f)
except:
    print("Could not load JSON")
    summary = []

# Find ANA in Summary
ana_summary = next((t for t in summary if t.get('team') == 'ANA'), None)

print("\n--- Shift-based Stats (from JSON) ---")
if ana_summary:
    print("Keys found in summary:", list(ana_summary.keys()))
    # Try multiple keys for attempts/goals
    s_goals = ana_summary.get('team_goals', ana_summary.get('goals_for', 'N/A'))
    s_attempts = ana_summary.get('team_attempts', ana_summary.get('fenwick_for', ana_summary.get('corsi_for', 'N/A')))
    s_xg = ana_summary.get('team_xgs', ana_summary.get('xgs_for', 'N/A'))
    
    print(f"Goals: {s_goals}")
    print(f"Attempts: {s_attempts}")
    print(f"xG: {s_xg}")
else:
    print("ANA not found in summary!")
    s_goals, s_attempts, s_xg = 'N/A', 'N/A', 'N/A'

# Calculate CSV-based Stats (API-based)
mask = (df['game_state'] == '5v5') & ((df['is_net_empty'] == 0) | (df['is_net_empty'].astype(str) == '0'))
df_5v5 = df[mask]

# Define attempt types
attempt_events = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
df_attempts = df_5v5[df_5v5['event'].isin(attempt_events)]

# Filter for ANA 
ana_attempts = df_attempts[((df_attempts['team_id'] == df_attempts['home_id']) & (df_attempts['home_abb'] == 'ANA')) | 
                           ((df_attempts['team_id'] == df_attempts['away_id']) & (df_attempts['away_abb'] == 'ANA'))]

ana_goals = ana_attempts[ana_attempts['event'] == 'goal']

print("\n--- API-based Stats (from CSV) ---")
print(f"ANA Attempts: {len(ana_attempts)}")
print(f"ANA Goals: {len(ana_goals)}")

# --- Game by Game Analysis (API) ---
print("\n--- Game-by-Game Breakdown (API) ---")
ana_games = ana_attempts.groupby('game_id').size().reset_index(name='attempts')
ana_goals_games = ana_goals.groupby('game_id').size().reset_index(name='goals')
ana_games = pd.merge(ana_games, ana_goals_games, on='game_id', how='left').fillna(0)

# Sort by attempts to see outliers
print("Top 5 Games (Most Attempts):")
print(ana_games.sort_values('attempts', ascending=False).head(5))

print("\nBottom 5 Games (Fewest Attempts):")
print(ana_games.sort_values('attempts', ascending=True).head(5))

# Check for zero event games
all_ana_games_in_csv = df_5v5[((df_5v5['home_abb'] == 'ANA') | (df_5v5['away_abb'] == 'ANA'))]['game_id'].unique()
missing_games = set(all_ana_games_in_csv) - set(ana_games['game_id'].unique())
if missing_games:
    print(f"\nGames with 0 attempts for ANA in 5v5: {missing_games}")
else:
    print("\nNo games with 0 attempts found.")

