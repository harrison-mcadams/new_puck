
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from puck.nhl_api import get_season

def check_league_gap():
    print("Loading local metadata...")
    try:
        df = pd.read_csv('data/edge_goals/metadata.csv')
        analyzed_ids = set(df['game_id'].astype(str))
    except:
        analyzed_ids = set()
    print(f"Local collection has {len(analyzed_ids)} unique games.")

    print("Fetching global 2025-26 schedule (this may take a moment)...")
    # get_season(team='all') pages through the weeks
    all_games = get_season(team='all', season='20252026', game_types=['02'])
    
    # Filter for completed
    completed = [g for g in all_games if g.get('gameState') in ['FINAL', 'OFF', 'CRIT']]
    
    print(f"NHL API reports {len(completed)} completed games for 2025-26.")
    
    missing = []
    for g in completed:
        gid = str(g['id'])
        if gid not in analyzed_ids:
            missing.append(gid)
            
    print(f"MISSING GAMES: {len(missing)}")
    
    # Save missing list for scraper
    if missing:
        with open('missing_games_list.txt', 'w') as f:
            for gid in missing:
                f.write(f"{gid}\n")
        print("Missing Game IDs saved to 'missing_games_list.txt'")

if __name__ == "__main__":
    check_league_gap()
