
import sys
import os
import gc

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puck import parse

# Define seasons to process (Reverse Chronological)
# Matching backfill_seasons.py range
SEASONS = [f"{y}{y+1}" for y in range(2025, 2013, -1)]

def reparse_all():
    print(f"Starting force re-parse for {len(SEASONS)} seasons...")
    
    for season in SEASONS:
        print(f"\nProcessing season {season}...")
        try:
            # Force re-processing by calling _scrape with process_elaborated=True and save_elaborated=True
            # use_cache=True ensures we don't hit the API if optional, but scrape logic usually checks cache.
            # We are NOT checking for existing CSVs here, effectively overwriting them.
            result = parse._scrape(
                season=season, 
                out_dir='data', 
                use_cache=True,  # Use existing raw JSONs
                verbose=True,
                max_workers=4,   # Parallel processing for speed
                process_elaborated=True,
                save_elaborated=True,
                save_raw=False,  # Don't re-save raw JSONs over themselves unnecessarily
                save_json=False,
                save_csv=True
            )
            print(f"Completed {season}.")
            gc.collect()
        except Exception as e:
            print(f"Error parsing season {season}: {e}")
            continue

    print("\nAll seasons re-parsed.")

if __name__ == "__main__":
    reparse_all()
