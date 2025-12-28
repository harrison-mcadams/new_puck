
import sys
import os
import gc

# Add project root to path
# Add project root to path (FIRST)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puck import parse
print(f"Loaded parse from: {parse.__file__}")

def reparse_2025():
    season = "20252026"
    print(f"Starting force re-parse for season {season}...")
    
    try:
        # Force re-processing by calling _scrape with process_elaborated=True and save_elaborated=True
        result = parse._scrape(
            season=season, 
            out_dir='data', 
            use_cache=True,  # Use existing raw JSONs
            verbose=True,
            max_workers=4,   # Parallel processing
            process_elaborated=True,
            save_elaborated=True,
            save_raw=False,
            save_json=False,
            save_csv=True
        )
        print(f"Completed {season}.")
    except Exception as e:
        print(f"Error parsing season {season}: {e}")

if __name__ == "__main__":
    reparse_2025()
