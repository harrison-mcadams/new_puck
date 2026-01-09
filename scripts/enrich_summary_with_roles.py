
import os
import sys
import pandas as pd
import requests

# Add project root to path
sys.path.append(os.getcwd())
from puck import fit_xgs

SUMMARY_FILE = "analysis/blocked_shots/blocked_shots_summary_batch.csv"

def main():
    if not os.path.exists(SUMMARY_FILE):
        print("Summary file not found.")
        return

    df = pd.read_csv(SUMMARY_FILE)
    print(f"Enriching {len(df)} records with shooter roles...")

    player_ids = []
    
    # We'll batch API requests by GAME to save time
    current_game = None
    game_plays = []
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"  {idx}/{len(df)}...")
            
        gid = str(row['game_id'])
        bid = str(row['block_id'])
        
        if gid != current_game:
            url = f"https://api-web.nhle.com/v1/gamecenter/{gid}/play-by-play"
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    game_plays = resp.json().get('plays', [])
                else:
                    game_plays = []
            except:
                game_plays = []
            current_game = gid
            
        pid = None
        for p in game_plays:
            if str(p.get('eventId')) == bid:
                pid = p.get('details', {}).get('shootingPlayerId')
                break
        player_ids.append(pid)

    df['player_id'] = player_ids
    
    # Now map Player IDs to Roles
    print("Mapping Player IDs to positions...")
    # Map back
    df = fit_xgs.enrich_data_with_bios(df)
    
    df.to_csv(SUMMARY_FILE, index=False)
    print(f"Done. Saved to {SUMMARY_FILE}")

if __name__ == "__main__":
    main()
