
"""reprocess_data.py

Re-runs the elaboration step for all seasons in data/ to backfill new features
(dist_from_last_event, speed_from_last_event) using cached raw feeds.
"""

import sys
from pathlib import Path
import re

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from puck import parse

def main():
    print("--- Reprocessing Data ---")
    data_dir = Path('data')
    if not data_dir.exists():
        print("Data directory not found.")
        return

    # Find season folders
    seasons = []
    for item in data_dir.iterdir():
        if item.is_dir() and re.match(r'^\d{8}$', item.name):
            seasons.append(item.name)
            
    seasons = sorted(seasons)
    print(f"Found {len(seasons)} seasons: {seasons}")
    
    for season in seasons:
        print(f"\nProcessing {season}...")
        try:
            # Re-scrapes (using cache) -> processes elaborated -> saves elaborated CSV
            # use_cache=True ensures we don't hit the API
            # save_csv=False to avoid re-writing raw CSVs if we already have them
            parse._scrape(
                season=season,
                out_dir=str(data_dir),
                use_cache=True,
                process_elaborated=True,
                save_elaborated=True,
                verbose=False
            )
            print(f"Completed {season}.")
        except Exception as e:
            print(f"Failed {season}: {e}")

if __name__ == "__main__":
    main()
