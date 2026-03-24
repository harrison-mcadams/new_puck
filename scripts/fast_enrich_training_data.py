import pandas as pd
import numpy as np
import os
import sys
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.getcwd())
from puck import html_enrichment, correction

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def enrich_season(season_str: str, overwrite: bool = False, max_workers: int = 4):
    csv_path = f"data/{season_str}/{season_str}_df.csv"
    if not os.path.exists(csv_path):
        csv_path = f"data/{season_str}.csv"
        if not os.path.exists(csv_path):
            logging.error(f"Season CSV NOT FOUND: {csv_path}")
            return

    logging.info(f"--- Processing Season {season_str} (Multi-threaded, workers={max_workers}) ---")
    df = pd.read_csv(csv_path, low_memory=False)
    
    # 1. Apply Attribution Fix
    logging.info(f"Applying attribution fix to {season_str}...")
    df = correction.fix_blocked_shot_attribution(df)

    # 2. Identify candidates
    mask_unknown = (df['event'] == 'blocked-shot') & (
        (df['shot_type'].isna()) | 
        (df['shot_type'].astype(str).str.lower().isin(['unknown', 'nan', '', 'none']))
    )
    
    if not mask_unknown.any():
        logging.info(f"No 'Unknown' blocks in {season_str}. Saving corrected file.")
        out_path = csv_path if overwrite else csv_path.replace(".csv", "_enriched.csv")
        df.to_csv(out_path, index=False)
        return

    target_game_ids = df.loc[mask_unknown, 'game_id'].unique()
    logging.info(f"Found {mask_unknown.sum()} unknown blocks across {len(target_game_ids)} games.")

    updated_total = 0
    results = {}

    def process_game(gid):
        try:
            game_mask = (df['game_id'] == gid)
            game_df = df[game_mask].copy()
            enriched_game_df = html_enrichment.enrich_blocks_with_html(game_df, str(int(gid)))
            return gid, enriched_game_df['shot_type']
        except Exception as e:
            return gid, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_game = {executor.submit(process_game, gid): gid for gid in target_game_ids}
        for i, future in enumerate(as_completed(future_to_game)):
            gid, shot_types = future.result()
            if shot_types is not None:
                game_mask = (df['game_id'] == gid)
                # Count changes for logging
                old_types = df.loc[game_mask, 'shot_type']
                df.loc[game_mask, 'shot_type'] = shot_types
                updated_total += (shot_types != old_types).sum()
            
            if i % 50 == 0:
                logging.info(f"Progress: {i}/{len(target_game_ids)} games enriched...")

    logging.info(f"Completed {season_str}. Updated {updated_total} shot types.")
    
    out_path = csv_path if overwrite else csv_path.replace(".csv", "_enriched.csv")
    df.to_csv(out_path, index=False)
    logging.info(f"Saved to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", default=["20222023", "20232024", "20242025", "20252026"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    for season in args.seasons:
        enrich_season(season, overwrite=args.overwrite, max_workers=args.workers)
