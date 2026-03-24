import pandas as pd
import numpy as np
import requests
import os

def audit_teams(season_csv):
    if not os.path.exists(season_csv):
        return
    
    df = pd.read_csv(season_csv)
    blocks = df[df['event'].str.lower() == 'blocked-shot'].sample(10, random_state=42)
    
    results = []
    print(f"\n--- Auditing {season_csv} ---")
    for idx, row in blocks.iterrows():
        game_id = str(int(row['game_id']))
        url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
        
        try:
            resp = requests.get(url, timeout=5)
            data = resp.json()
            
            # Find the exact event by time/period/coord if possible
            # Or just find any block with the same player
            matching_play = None
            for p in data.get('plays', []):
                if p.get('typeDescKey') == 'blocked-shot':
                   d = p.get('details', {})
                   # Check coordinates (allowing for some pipeline drift like arena adj)
                   if abs(d.get('xCoord', 0) - row['x']) < 10:
                        matching_play = p
                        break
            
            if matching_play:
                d = matching_play.get('details', {})
                api_owner = d.get('eventOwnerTeamId')
                shooter_id = d.get('shootingPlayerId')
                blocker_id = d.get('blockingPlayerId')
                
                # Check which team the shooter is on
                # We need roster for this
                home_id = data.get('homeTeam', {}).get('id')
                away_id = data.get('awayTeam', {}).get('id')
                
                # Assume home/away for now based on owner if id not found
                # Or just report what we see
                results.append({
                    'EventOwnersTeam': api_owner,
                    'CSV_TeamID': row['team_id'],
                    'ShooterID': shooter_id,
                    'BlockerID': blocker_id,
                    'X': row['x'],
                    'Zone': d.get('zoneCode')
                })
                print(f"Game {game_id}: API Owner={api_owner}, CSV Team={row['team_id']}, Shooter={shooter_id}, Blocker={blocker_id}, X={row['x']:.1f}")
        except Exception as e:
            print(f"Error checking game {game_id}: {e}")

audit_teams('data/20202021/20202021_df.csv')
