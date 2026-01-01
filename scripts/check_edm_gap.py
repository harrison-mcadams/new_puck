
import pandas as pd
import requests
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_edm_gap():
    # 1. Get Analyzed Games
    try:
        df = pd.read_csv('data/edge_goals/metadata.csv')
        analyzed_ids = set(df['game_id'].astype(str))
    except:
        analyzed_ids = set()

    # 2. Get EDM Schedule
    print("Fetching EDM schedule...")
    try:
        # Schedule endpoint
        url = "https://api-web.nhle.com/v1/club-schedule-season/EDM/20252026"
        resp = requests.get(url)
        data = resp.json()
        
        games = data.get('games', [])
        regular_season = [g for g in games if str(g.get('id')).startswith('202502')]
        
        # Filter for completed games (score exists or status not scheduled)
        completed = [g for g in regular_season if g.get('gameState') in ['FINAL', 'OFF', 'CRIT']] 
        # Note: gameState names vary, but 'FINAL'/'OFF' usually mostly what we want.
        # Actually safer: check if 'score' exists? Or assume past dates.
        # Let's just grab all and check which ones match.
        
        print(f"EDM has {len(completed)} completed/active games in 2025-26 schedule.")
        
        missing = []
        found = 0
        for g in completed:
            gid = str(g['id'])
            if gid in analyzed_ids:
                found += 1
            else:
                missing.append(gid)
        
        print(f"We have analyzed {found} of them.")
        print(f"Missing {len(missing)} games: {missing[:5]}...")
        
        return missing

    except Exception as e:
        print(f"Error checking schedule: {e}")
        return []

if __name__ == "__main__":
    check_edm_gap()
