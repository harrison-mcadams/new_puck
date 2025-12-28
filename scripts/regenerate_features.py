
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from puck import parse

def regenerate(season):
    print(f"Regenerating features for {season}...")
    # use_cache=True ensures we don't hit the API, just re-process local raw data
    df = parse._scrape(season=season, 
                       team='all', 
                       out_dir='data', 
                       use_cache=True, 
                       process_elaborated=True, 
                       save_elaborated=True, 
                       return_elaborated_df=True,
                       save_csv=True) # Ensure we save the CSV
    
    if 'is_rebound' in df.columns:
        print(f"SUCCESS: 'is_rebound' present. Counts:\n{df['is_rebound'].value_counts(dropna=False)}")
    
    if 'is_rush' in df.columns:
        print(f"SUCCESS: 'is_rush' present. Counts:\n{df['is_rush'].value_counts(dropna=False)}")
    else:
        print("FAILURE: 'is_rush' NOT found in returned DataFrame.")

if __name__ == "__main__":
    # Regenerate for all seasons from 2014-2015 to 2025-2026
    start_year = 2014
    end_year = 2025 # current season start
    
    for year in range(start_year, end_year + 1):
        season_str = f"{year}{year+1}"
        try:
            regenerate(season_str)
        except Exception as e:
            print(f"Failed to regenerate {season_str}: {e}")
