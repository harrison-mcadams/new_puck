
import pandas as pd
import sys
import os
import json
import csv
import time

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from puck.nhl_api import get_game_feed
from puck.edge import fetch_tracking_data, transform_coordinates

OUTPUT_DIR = os.path.join("data", "edge_goals")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.csv")
SEASON = "20252026"
MISSING_LIST = "missing_games_list.txt"

def scrape_list():
    if not os.path.exists(MISSING_LIST):
        print("No missing list found.")
        return

    with open(MISSING_LIST, 'r') as f:
        missing_ids = [line.strip() for line in f if line.strip()]
        
    print(f"Scraping {len(missing_ids)} missing games from list...")
    
    season_dir = os.path.join(OUTPUT_DIR, SEASON)
    os.makedirs(season_dir, exist_ok=True)
    
    metadata_f = open(METADATA_FILE, 'a', newline='', encoding='utf-8')
    writer = csv.writer(metadata_f)
    
    # Load existing to avoid dupes if re-run
    try:
        df = pd.read_csv(METADATA_FILE)
        existing = set(df['game_id'].astype(str))
    except:
        existing = set()

    processed_count = 0
    
    for i, game_id in enumerate(missing_ids):
        if game_id in existing:
            print(f"[{i+1}/{len(missing_ids)}] Game {game_id} already in metadata. Skipping.")
            continue
            
        print(f"[{i+1}/{len(missing_ids)}] Processing Game {game_id}...")
        
        try:
            feed = get_game_feed(game_id)
        except Exception as e:
            print(f"  Failed feed: {e}")
            continue
            
        game_date = feed.get('gameDate', 'Unknown')
        
        plays = feed.get('plays', [])
        if not plays: plays = feed.get('liveData', {}).get('plays', {}).get('allPlays', [])
        
        goals_found = 0
        for play in plays:
            is_goal = False
            event_id = None
            if play.get('typeDescKey') == 'goal':
                is_goal = True
                event_id = str(play.get('eventId'))
            elif play.get('type') == 'GOAL':
                 is_goal = True
                 event_id = str(play.get('eventId'))
                 
            if not is_goal or not event_id: continue
            
            # 5v5 Filter
            sc = play.get('situationCode', '0000')
            if not (len(sc) == 4 and sc[1] == '5' and sc[2] == '5'): continue
            
            # Paths
            json_path = os.path.join(season_dir, f"game_{game_id}_goal_{event_id}_edge.json")
            csv_filename = os.path.join(season_dir, f"game_{game_id}_goal_{event_id}_positions.csv")
            
            # Fetch
            data = None
            if os.path.exists(json_path):
                 with open(json_path, 'r') as jf: data = json.load(jf)
            else:
                 data = fetch_tracking_data(game_id, event_id, SEASON)
                 if data:
                     with open(json_path, 'w') as jf: json.dump(data, jf, indent=2)
            
            if data:
                # CSV Convert
                csv_rows = [['frame_idx', 'timestamp', 'entity_type', 'entity_id', 'team_id', 'x', 'y', 'sweater_number']]
                if isinstance(data, list):
                    for fi, frame in enumerate(data):
                        ts = frame.get('timeStamp')
                        on_ice = frame.get('onIce', {})
                        for k, v in on_ice.items():
                             xr, yr = v.get('x'), v.get('y')
                             if xr is None or yr is None: continue
                             x, y = transform_coordinates(xr, yr)
                             if k == "1":
                                 csv_rows.append([fi, ts, 'puck', 'puck', '', x, y, ''])
                             else:
                                 csv_rows.append([fi, ts, 'player', v.get('playerId', k), v.get('teamId', ''), x, y, v.get('sweaterNumber', '')])

                with open(csv_filename, 'w', newline='') as cf:
                    csv.writer(cf).writerows(csv_rows)
                
                # Metadata
                period = play.get('periodDescriptor', {}).get('number', 0)
                time_in_period = play.get('timeInPeriod', '00:00')
                scorer_id = play.get('details', {}).get('scoringPlayerId')
                strength = play.get('strength', 'EV')
                
                writer.writerow([SEASON, game_id, event_id, game_date, period, time_in_period, scorer_id, "Unknown", strength])
                metadata_f.flush()
                goals_found += 1
                
        print(f"  Saved {goals_found} 5v5 goals.")
        processed_count += 1
        time.sleep(1.0) # Rate limit

    metadata_f.close()
    print(f"Scrape complete. Processed {processed_count} games.")

if __name__ == "__main__":
    scrape_list()
