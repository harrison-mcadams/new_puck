import pandas as pd
import numpy as np
import os
import sys
import logging

sys.path.append(os.getcwd())
from puck import html_enrichment, config as puck_config

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(puck_config.DATA_DIR, "html_enrichment_batch.log")),
        logging.StreamHandler()
    ]
)

def enrich_season(season_str: str, overwrite: bool = False):
    csv_path = os.path.join(puck_config.DATA_DIR, season_str, f"{season_str}_df.csv")
    if not os.path.exists(csv_path):
        # Try root data dir as fallback
        csv_path = os.path.join(puck_config.DATA_DIR, f"{season_str}.csv")
        if not os.path.exists(csv_path):
            logging.error(f"Season CSV NOT FOUND: {csv_path}")
            return

    logging.info(f"--- Processing Season {season_str} ---")
    df = pd.read_csv(csv_path, low_memory=False)
    
    # 1. Apply Attribution Fix (Blocker ID -> Shooter)
    # This ensures team_id is corrected and coordinates are shooter-centric
    logging.info(f"Applying attribution fix to all blocked shots in {season_str}...")
    from puck import correction
    df = correction.fix_blocked_shot_attribution(df)

    # 2. Identify candidates for enrichment
    # We look for blocked shots where shot_type is missing, NaN, empty, or 'Unknown'
    mask_unknown = (df['event'] == 'blocked-shot') & (
        (df['shot_type'].isna()) | 
        (df['shot_type'].astype(str).str.lower().isin(['unknown', 'nan', '', 'none']))
    )
    
    if not mask_unknown.any():
        logging.info(f"No 'Unknown' blocked shots found in {season_str} after attribution fix. Saving corrected file.")
        # Still save because attribution changed
        out_path = csv_path if overwrite else csv_path.replace(".csv", "_enriched.csv")
        df.to_csv(out_path, index=False)
        return

    target_game_ids = df.loc[mask_unknown, 'game_id'].unique()
    logging.info(f"Found {mask_unknown.sum()} unknown blocks across {len(target_game_ids)} games.")

    updated_total = 0
    # Process game by game to minimize HTML fetches
    for i, gid in enumerate(target_game_ids):
        if i % 10 == 0:
            logging.info(f"Processing game {i}/{len(target_game_ids)}: {gid}")
        try:
            # We use the existing function which handles the per-game grouping and parsing
            game_mask = (df['game_id'] == gid)
            game_df = df[game_mask].copy()
            
            # Enrich
            enriched_game_df = html_enrichment.enrich_blocks_with_html(game_df, str(int(gid)))
            
            # Update shot_type
            df.loc[game_mask, 'shot_type'] = enriched_game_df['shot_type']
            # Count changes relative to original for logging
            # (Note: we already assigned corrected team etc above)
            updated_total += (enriched_game_df['shot_type'] != game_df['shot_type']).sum()
                
        except Exception as e:
            logging.error(f"Failed to enrich game {gid}: {e}")

    logging.info(f"Completed {season_str}. Updated {updated_total} shot types.")
    
    if updated_total > 0:
        out_path = csv_path if overwrite else csv_path.replace(".csv", "_enriched.csv")
        df.to_csv(out_path, index=False)
        logging.info(f"Saved enriched data to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", default=["20202021", "20212022", "20222023", "20232024", "20242025"])
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing CSV files")
    args = parser.parse_args()

    for season in args.seasons:
        enrich_season(season, overwrite=args.overwrite)
