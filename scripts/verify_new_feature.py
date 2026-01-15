
import pandas as pd
import numpy as np
import logging
import json
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from puck import parse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def verify_new_feature(csv_path):
    logger.info(f"Loading raw feeds from {csv_path}...")
    
    records = []
    
    if not os.path.exists(csv_path):
        logger.error(f"File not found: {csv_path}")
        # Try listing dir to help
        try:
             logger.info(f"Files in data/: {os.listdir('data')}")
        except:
             pass
        return

    try:
        # Use chunksize to just get a few without loading everything
        chunk = pd.read_csv(csv_path, nrows=50)
        
        for idx, row in chunk.iterrows():
            try:
                feed_str = row['feed']
                if isinstance(feed_str, str):
                    feed = json.loads(feed_str)
                else:
                    continue
                    
                # Parse
                ev_df = parse._game(feed)
                if ev_df is None or ev_df.empty:
                    continue
                    
                # Elaborate (THIS IS WHERE FEATURES ARE MADE)
                edf = parse._elaborate(ev_df)
                if edf is not None and not edf.empty:
                     records.extend(edf.to_dict('records'))
                     
            except Exception as e:
                pass
                
    except Exception as e:
        logger.error(f"Failed to load raw data: {e}")
        return

    df = pd.DataFrame.from_records(records)
    logger.info(f"Generated DataFrame with {len(df)} rows.")
    
    # Check for new feature
    col = 'angle_change_last_event'
    if col in df.columns:
        logger.info(f"\n--- Verification: {col} ---")
        series = df[col].dropna()
        
        if len(series) == 0:
             logger.warning("Column exists but is all NaN/None (expected if missing previous event?).")
        else:
            # Stats
            desc = series.describe()
            logger.info(f"\nDistribution:\n{desc}")
            
            # Check range 0-180
            min_val = series.min()
            max_val = series.max()
            logger.info(f"Range: [{min_val}, {max_val}]")
            
            if max_val > 180.01:
                 logger.error("FAIL: Angle Change exceeds 180 degrees.")
            else:
                 logger.info("PASS: Range within 0-180.")

            # Check Random Samples
            sample = df[['event', 'x', 'y', 'angle_deg', 'angle_change_last_event']].dropna().sample(min(10, len(series)))
            logger.info(f"\nSample Rows:\n{sample}")
        
    else:
        logger.error(f"FAIL: '{col}' column not found in elaborated DataFrame.")

if __name__ == "__main__":
    # Use 2025 raw feeds
    # Look for likely paths
    possible_paths = [
        "data/20252026/20252026_raw_game_feeds.csv",
        "data/20252026_raw_game_feeds.csv",
        r"c:\Users\harri\Desktop\new_puck\data\20252026_raw_game_feeds.csv"
    ]
    
    path = None
    for p in possible_paths:
        if os.path.exists(p):
            path = p
            break
            
    if path:
        verify_new_feature(path)
    else:
        print("Could not find raw game feeds csv. Checking data dir...")
        try:
             print(os.listdir("data"))
        except:
             print("No data dir.")
