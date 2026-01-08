
import json
import os
import glob
import pandas as pd
import numpy as np

DATA_DIR = "data"
season = "20182019"

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

# CLEANED: Only shots in front of goal line
df_clean = df[df['x'] <= 89].copy()

tampa_clean = df_clean[df_clean['arena'].str.contains('Tampa|Lightning', case=False)]
league_clean = df_clean

print(f"CLEANED BASELINE (X <= 89):")
print(f"Total Clean League Shots: {len(league_clean)}")
print(f"Total Clean Tampa Shots: {len(tampa_clean)}")

for p in [25, 50, 75, 90, 95, 99]:
    l_v = np.percentile(league_clean['x'].values, p)
    t_v = np.percentile(tampa_clean['x'].values, p)
    print(f"{p}th Percentile | League: {l_v:.1f}ft | Tampa: {t_v:.1f}ft | Delta: {l_v - t_v:.1f}ft")
