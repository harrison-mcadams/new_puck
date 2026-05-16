"""download_all_seasons.py

Downloads, reprocesses, and saves out season event data for all seasons 
for which we have data access to, using the revamped pipeline.
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puck import parse, verify
from puck.features import get_features

# Play-by-play data with coordinates is generally available back to 2010-2011.
SEASONS = [f"{y}{y+1}" for y in range(2025, 2009, -1)]

def main():
    parser = argparse.ArgumentParser(description="Download and process all accessible NHL seasons with verification.")
    parser.add_argument('--out-dir', type=str, default='data', help="Output directory")
    parser.add_argument('--max-workers', type=int, default=8, help="Number of concurrent workers for fetching")
    parser.add_argument('--skip-raw', action='store_true', help="Skip saving raw JSON/CSV feeds")
    parser.add_argument('--latest-only', action='store_true', help="Only download the most recent season (for testing)")
    parser.add_argument('--verify-blocked', action='store_true', default=True, help="Perform specific sanity checks on blocked shots")
    args = parser.parse_args()

    seasons_to_process = SEASONS
    if args.latest_only:
        seasons_to_process = [SEASONS[0]]

    print(f"Target Seasons: {seasons_to_process}")
    
    data_dir = Path(args.out_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Define features to verify against (all_inclusive ensures we catch everything)
    verify_features = get_features('all_inclusive')

    for season in seasons_to_process:
        print(f"\n==========================================")
        print(f" Processing Season: {season}")
        print(f"==========================================\n")
        
        try:
            # Using the revamped pipeline in parse._scrape
            # We set return_elaborated_df=True to perform in-memory verification
            res = parse._scrape(
                season=season,
                team='all',
                out_dir=str(data_dir),
                use_cache=True,
                max_workers=args.max_workers,
                verbose=True,
                save_raw=not args.skip_raw,
                save_csv=not args.skip_raw,
                save_json=not args.skip_raw,
                process_elaborated=True,
                save_elaborated=True,
                return_elaborated_df=True
            )
            
            df = res if isinstance(res, pd.DataFrame) else res.get('elaborated_df')

            if df is not None and not df.empty:
                print(f"\n--- Verifying {season} Data Integrity ---")
                verify.verify_df(
                    df, 
                    features=verify_features, 
                    verify_blocked=args.verify_blocked,
                    mode='train'
                )
            else:
                print(f"Warning: No data returned for season {season}, skipping verification.")

            print(f"Successfully completed pipeline and verification for {season}")
        except Exception as e:
            print(f"Error processing season {season}: {e}")

if __name__ == "__main__":
    main()
