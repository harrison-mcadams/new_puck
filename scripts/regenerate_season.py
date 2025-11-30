import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import parse

def regenerate(season='20252026'):
    print(f"Regenerating data for season {season}...")
    out_dir = os.path.join('data', season)
    os.makedirs(out_dir, exist_ok=True)
    
    # parse._season will save to {out_path}/{season}.csv
    df = parse._season(season=season, out_path=out_dir, use_cache=True, verbose=True)
    
    if not df.empty:
        print(f"Successfully regenerated season data: {len(df)} rows.")
        # Check if file exists
        expected_file = os.path.join(out_dir, f"{season}.csv")
        if os.path.exists(expected_file):
            print(f"File saved to {expected_file}")
        else:
            print(f"Warning: Expected output file {expected_file} not found.")
    else:
        print("Failed to regenerate season data (empty dataframe).")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    regenerate()
