"""
regenerate_season_csvs.py
========================
Regenerates the canonical 'Gold Standard' {season}_df.csv files for all modern era seasons.
Uses the Canonical Processing Chain (CPC):
  1. parse._game(feed)
  2. html_enrichment.enrich_blocks_with_html(df)
  3. parse._elaborate(df)

This ensures correct blocked shot attribution, enrichment, and spatial features.
"""
import os
import sys
import json
import pandas as pd
import logging
from pathlib import Path
from joblib import Parallel, delayed
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puck import parse, html_enrichment, config, nhl_api

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SEASONS = [
    "20202021",
    "20212022",
    "20222023",
    "20232024",
    "20242025",
    "20252026"
]

def process_game(game_id, feed_json):
    """Canonical Processing Chain for a single game."""
    try:
        # 1. Parse raw feed
        # (puck.parse._game is now fixed to use API shooter attribution)
        df_game = parse._game(feed_json)
        if df_game is None or df_game.empty:
            return None
        
        # 2. HTML Enrichment (Shot types for blocks)
        df_enriched = html_enrichment.enrich_blocks_with_html(df_game, str(game_id))
        
        # 3. Elaboration (Spatial features)
        # (puck.parse._elaborate calculates distance/angle)
        df_final = parse._elaborate(df_enriched)
        
        return df_final
    except Exception as e:
        logger.error(f"Error processing game {game_id}: {e}")
        return None

def regenerate_season(season):
    logger.info(f"=== Regenerating Season {season} ===")
    
    season_dir = Path(config.DATA_DIR) / season
    raw_feeds_path = season_dir / f"{season}_game_feeds.csv"
    output_path = season_dir / f"{season}_df.csv"
    
    if not raw_feeds_path.exists():
        logger.warning(f"Raw feeds not found for {season} at {raw_feeds_path}. Skipping.")
        return
    
    # Load raw feeds
    logger.info(f"Loading raw feeds from {raw_feeds_path}...")
    df_feeds = pd.read_csv(raw_feeds_path)
    
    if 'game_id' not in df_feeds.columns or 'feed' not in df_feeds.columns:
        logger.error(f"Malformed game feeds CSV for {season}. Expected 'game_id' and 'feed' columns.")
        return

    # Backup existing df.csv if it exists
    if output_path.exists():
        backup_path = season_dir / f"{season}_df_backup_{int(time.time())}.csv"
        logger.info(f"Backing up existing {season}_df.csv to {backup_path.name}")
        os.rename(output_path, backup_path)

    # Process games in parallel
    logger.info(f"Processing {len(df_feeds)} games for {season} in parallel...")
    
    game_data = zip(df_feeds['game_id'], df_feeds['feed'])
    
    def worker(gid, feed_str):
        try:
            feed = json.loads(feed_str)
            return process_game(gid, feed)
        except Exception as e:
            logger.error(f"Failed to load JSON for game {gid}: {e}")
            return None

    # Use -1 for all cores, but maybe lower if memory is an issue
    processed_dfs = Parallel(n_jobs=-1)(delayed(worker)(gid, f) for gid, f in game_data)
    
    # Filter out Nones and concatenate
    final_dfs = [d for d in processed_dfs if d is not None and not d.empty]
    if not final_dfs:
        logger.warning(f"No games successfully processed for {season}.")
        return

    logger.info(f"Concatenating {len(final_dfs)} games...")
    df_season = pd.concat(final_dfs, ignore_index=True)
    
    # Save Gold Standard CSV
    logger.info(f"Saving {len(df_season)} events to {output_path}...")
    df_season.to_csv(output_path, index=False)
    
    # Also remove any legacy _enriched.csv to enforce the single-file convention
    enriched_path = season_dir / f"{season}_df_enriched.csv"
    if enriched_path.exists():
        logger.info(f"Removing legacy enriched file: {enriched_path.name}")
        os.remove(enriched_path)

    logger.info(f"Season {season} regeneration complete.")

def main():
    start_time = time.time()
    for season in SEASONS:
        regenerate_season(season)
    
    elapsed = time.time() - start_time
    logger.info(f"All seasons regenerated in {elapsed/60:.1f} minutes.")

if __name__ == "__main__":
    main()
