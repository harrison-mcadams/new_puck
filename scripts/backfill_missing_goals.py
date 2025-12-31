import pandas as pd
import os
import sys
import csv
import time
import logging

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck.edge import fetch_tracking_data, transform_coordinates

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

DATA_DIR = r"c:\Users\harri\Desktop\new_puck\data\edge_goals"
METADATA_FILE = os.path.join(DATA_DIR, "metadata.csv")

def backfill():
    if not os.path.exists(METADATA_FILE):
        logging.error(f"Metadata file not found: {METADATA_FILE}")
        return

    df_meta = pd.read_csv(METADATA_FILE)
    logging.info(f"Targeting {len(df_meta)} goals from metadata.")
    
    downloaded = 0
    skipped = 0
    failures = 0

    for idx, row in df_meta.iterrows():
        try:
            game_id = str(int(row['game_id']))
            event_id = str(int(row['event_id']))
            season = str(row['season'])
        except:
            continue

        csv_filename = os.path.join(DATA_DIR, f"game_{game_id}_goal_{event_id}_positions.csv")
        
        if os.path.exists(csv_filename):
            skipped += 1
            if skipped % 50 == 0:
                print(f"Skipped {skipped} existing files...", end='\r')
            continue

        logging.info(f"Downloading missing: Game {game_id} Event {event_id} ({season})")
        
        try:
            data = fetch_tracking_data(game_id, event_id, season)
        except Exception as e:
            logging.error(f"Fetch error: {e}")
            data = None
            
        if not data:
            logging.warning(f"  -> No data found.")
            failures += 1
            continue

        # Convert to CSV
        csv_rows = []
        csv_rows.append(['frame_idx', 'timestamp', 'entity_type', 'entity_id', 'team_id', 'x', 'y', 'sweater_number'])
        
        if isinstance(data, list):
            for i, frame in enumerate(data):
                ts = frame.get('timeStamp')
                on_ice = frame.get('onIce', {})
                if not on_ice: continue
                
                for key, info in on_ice.items():
                    x_raw = info.get('x')
                    y_raw = info.get('y')
                    if x_raw is None or y_raw is None: continue
                    
                    x, y = transform_coordinates(x_raw, y_raw)
                    
                    if key == "1": # Puck
                        csv_rows.append([i, ts, 'puck', 'puck', '', x, y, ''])
                    else: # Player
                        csv_rows.append([
                            i, ts, 'player',
                            info.get('playerId', key),
                            info.get('teamId', ''),
                            x, y,
                            info.get('sweaterNumber', '')
                        ])
        
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
            
        downloaded += 1
        time.sleep(0.5) # Rate limiting

    print(f"\nBackfill Complete.")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped (Exists): {skipped}")
    print(f"Failures: {failures}")

if __name__ == "__main__":
    backfill()
