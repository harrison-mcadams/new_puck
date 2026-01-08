
import json
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "data"
season = "20182019"
arena = "Lightning" # Key from JSON

def load_season_data(season_dir):
    shots = []
    files = glob.glob(os.path.join(season_dir, "game_*.json"))
    for f_path in files:
        try:
            with open(f_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            game_data = data.get('gameData', {})
            home_team = game_data.get('teams', {}).get('home', {}).get('name', 'Unknown')
            if home_team == 'Unknown':
                 home_team_obj = data.get('homeTeam', {})
                 home_team = home_team_obj.get('name')
                 if not home_team:
                      home_team = home_team_obj.get('commonName', {}).get('default', 'Unknown')

            plays = data.get('plays', [])
            if not plays:
                plays = data.get('liveData', {},).get('plays', {}).get('allPlays', [])
                
            for play in plays:
                event = play.get('typeDescKey') or (play.get('type') or {}).get('description')
                if event not in ['shot-on-goal', 'missed-shot', 'goal']:
                    continue
                
                details = play.get('details', {})
                x = details.get('xCoord')
                if x is not None:
                    shots.append({'x': abs(x), 'arena': home_team})
        except:
            continue
    return pd.DataFrame(shots)

df = load_season_data(os.path.join(DATA_DIR, season))
if df.empty:
    print("No data found")
    exit()

tampa_shots = df[df['arena'].str.contains('Tampa|Lightning', case=False)]
league_shots = df

print(f"Total Season Shots: {len(league_shots)}")
print(f"Tampa Shots: {len(tampa_shots)}")

# Plot CDFs
def plot_cdf(arena_vals, league_vals, title, filename):
    plt.figure(figsize=(10, 6))
    
    # League
    sorted_league = np.sort(league_vals)
    y_league = np.arange(1, len(sorted_league)+1) / len(sorted_league)
    plt.plot(sorted_league, y_league, label='League', color='red', linewidth=2)
    
    # Arena
    sorted_arena = np.sort(arena_vals)
    y_arena = np.arange(1, len(sorted_arena)+1) / len(sorted_arena)
    plt.plot(sorted_arena, y_arena, label='Tampa Bay', color='blue', linewidth=2)
    
    plt.title(title)
    plt.xlabel("Absolute X Coordinate")
    plt.ylabel("Cumulative Probability")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(filename)
    print(f"Saved {filename}")

plot_cdf(tampa_shots['x'].values, league_shots['x'].values, 
         "CDF Comparison: X-Coordinates (2018-2019)", 
         "tampa_vs_league_cdf.png")

# Report some percentiles
for p in [25, 50, 75, 90, 95]:
    l_v = np.percentile(league_shots['x'].values, p)
    t_v = np.percentile(tampa_shots['x'].values, p)
    print(f"{p}th Percentile | League: {l_v:.1f}ft | Tampa: {t_v:.1f}ft | Delta: {l_v - t_v:.1f}ft")
