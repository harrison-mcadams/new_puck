import os
import pandas as pd
from pathlib import Path
import logging
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puck import data_pipeline, config

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

def standardize_season(season):
    logger.info(f"=== Standardizing Season {season} ===")
    
    season_dir = Path(config.DATA_DIR) / season
    csv_path = season_dir / f"{season}_df.csv"
    
    if not csv_path.exists():
        logger.warning(f"CSV not found for {season} at {csv_path}. Skipping.")
        return
    
    # Load
    logger.info(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} events.")

    # Standardize orientation and features for ALL events
    # We set apply_filtering=False to keep hits, faceoffs, etc.
    # We set is_training=False to avoid dithering (we want exact coordinates in the CSV).
    logger.info("Applying orientation standardization and feature recalculation to full dataset...")
    df_std = data_pipeline.preprocess_features(
        df, 
        is_training=False, 
        apply_filtering=False, 
        apply_imputation=True, # Impute blocked shots if not already done
        apply_arena_adjustments=True,
        apply_html_enrichment=False, # ALREADY DONE IN REGEN
        verbose=True
    )
    
    # Save back
    logger.info(f"Saving standardized dataset to {csv_path}...")
    df_std.to_csv(csv_path, index=False)
    logger.info(f"Season {season} standardization complete.")

def main():
    for season in SEASONS:
        standardize_season(season)
    logger.info("All seasons standardized.")

if __name__ == "__main__":
    main()
