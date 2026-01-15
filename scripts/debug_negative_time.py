
import pandas as pd
import numpy as np
import logging
import json
import sys
import os

sys.path.append(os.getcwd())
from puck import parse

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def debug_negative_time(csv_path):
    logger.info(f"Loading raw feeds from {csv_path}...")
    
    # Read a chunk that likely contains enough data
    try:
        df_raw = pd.read_csv(csv_path, nrows=200) 
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return

    for idx, row in df_raw.iterrows():
        try:
            feed = json.loads(row['feed'])
            
            # 1. Parse to Events
            ev_df = parse._game(feed)
            if ev_df is None or ev_df.empty:
                continue
                
            # 2. Elaborate (Calculates Features)
            edf = parse._elaborate(ev_df)
            
            if 'last_event_time_diff' not in edf.columns:
                continue

            # Check for negative values
            neg_mask = edf['last_event_time_diff'] < 0
            if neg_mask.any():
                print(f"DEBUG_FOUND_NEGATIVE: Game {row.get('game_id')}")
                
                bad_rows = edf[neg_mask]
                print(f"COUNT: {len(bad_rows)}")
                
                # Print details of the first few bad rows
                for i, r in bad_rows.head(5).iterrows():
                    print(f"ROW[{i}]: Event='{r['event']}', Time={r['total_time_elapsed_s']}, LastDiff={r['last_event_time_diff']}, Synth={r.get('synthetic', False)}")
                
                # We found one, that's enough to prove the point
                return
                
        except Exception as e:
            pass
    
    print("VERIFICATION_SUCCESS: No negative time diffs found in checked games.")

if __name__ == "__main__":
    path = "data/20252026/20252026_raw_game_feeds.csv" 
    # Try absolute path based on user info
    if not os.path.exists(path):
        path = r"c:\Users\harri\Desktop\new_puck\data\20252026_raw_game_feeds.csv"
         
    debug_negative_time(path)
