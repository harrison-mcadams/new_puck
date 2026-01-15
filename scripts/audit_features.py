
import pandas as pd
import numpy as np
import logging
import json
import ast
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from puck import parse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def audit_features_dynamic(csv_path):
    logger.info(f"Loading raw feeds from {csv_path}...")
    
    records = []
    
    # Read first 50 games from CSV
    try:
        # Use chunksize to just get a few without loading everything
        chunk = pd.read_csv(csv_path, nrows=50)
        
        for idx, row in chunk.iterrows():
            try:
                feed_str = row['feed']
                # It might be double encoded or just json
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
                # logger.warning(f"Error parsing game {idx}: {e}")
                pass
                
    except Exception as e:
        logger.error(f"Failed to load raw data: {e}")
        return

    df = pd.DataFrame.from_records(records)
    logger.info(f"Generated DataFrame with {len(df)} rows and {len(df.columns)} columns.")
    
    # 1. Inspect 'last_event_type'
    if 'last_event_type' in df.columns:
        logger.info("\n--- Audit: last_event_type ---")
        
        # Fill NaN
        filled = df['last_event_type'].fillna('Unknown')
        
        # Value Counts
        vc = filled.value_counts()
        logger.info(f"Unique values: {len(vc)}")
        logger.info(f"\nTop 20 values:\n{vc.head(20)}")
        
        logger.info(f"\nTail 20 values (Potential Noise?):\n{vc.tail(20)}")
            
        # Check if any look like "Shot from 25ft" or similar high-cardinality strings
        rare_count = (vc < 5).sum()
        logger.info(f"\nNumber of types with < 5 occurrences: {rare_count}")
        
    else:
        logger.warning("'last_event_type' column not found.")

    # 2. Inspect 'dist_from_last_event' and 'speed_from_last_event'
    numeric_cols = ['dist_from_last_event', 'speed_from_last_event', 'last_event_time_diff']
    
    for col in numeric_cols:
        if col in df.columns:
            logger.info(f"\n--- Audit: {col} ---")
            series = df[col].dropna()
            
            # Stats
            desc = series.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
            logger.info(f"\nDistribution:\n{desc}")
            
            # Check for zeros or weird spikes
            zeros = (series == 0).sum()
            logger.info(f"Count of Zeros: {zeros} ({zeros/len(series):.1%})")
            
            # Check for extreme values (potential outliers)
            if col == 'speed_from_last_event':
                high_speed = (series > 50).sum()
                logger.info(f"High Speed (>50 ft/s) count: {high_speed}")
                
        else:
            logger.warning(f"'{col}' column not found.")

    # 3. Simple Correlation (if target exists)
    if 'event' in df.columns:
        df['is_goal'] = (df['event'] == 'goal').astype(int)
        
        # Correlation for numerics
        corr_cols = [c for c in numeric_cols if c in df.columns]
        if corr_cols:
            logger.info("\n--- Correlations with is_goal ---")
            corr = df[corr_cols + ['is_goal']].corr()['is_goal']
            logger.info(f"\n{corr}")
    
    # 4. Check for 'Unknown' skew in last_event
    # Check frequency of 'Unknown' or None
    logger.info("\n--- Missing Data Checks ---")
    if 'dist_from_last_event' in df.columns:
        missing_dist = df['dist_from_last_event'].isna().sum()
        logger.info(f"Missing dist_from_last_event: {missing_dist} ({missing_dist/len(df):.1%})")

if __name__ == "__main__":
    # Use 2025 raw feeds
    # Found at data/20252026_raw_game_feeds.csv based on find_by_name
    path = "data/20252026_raw_game_feeds.csv"
    if not os.path.exists(path):
         # Try absolute path based on user info
         path = r"c:\Users\harri\Desktop\new_puck\data\20252026_raw_game_feeds.csv"
         
    audit_features_dynamic(path)
