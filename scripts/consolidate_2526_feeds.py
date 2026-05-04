import os
import json
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def consolidate_2526_feeds():
    season = "20252026"
    data_dir = Path(f"data/{season}")
    output_path = data_dir / f"{season}_game_feeds.csv"
    
    if not data_dir.exists():
        logger.error(f"Directory {data_dir} does not exist.")
        return

    logger.info(f"Consolidating individual game JSONs in {data_dir}...")
    
    records = []
    # Match game_20250xxxxx.json
    for p in data_dir.glob("game_20250*.json"):
        try:
            game_id = p.stem.replace("game_", "")
            with open(p, 'r', encoding='utf-8') as f:
                feed = json.load(f)
                records.append({
                    'game_id': game_id,
                    'feed': json.dumps(feed, ensure_ascii=False)
                })
        except Exception as e:
            logger.warning(f"Failed to read {p}: {e}")

    if not records:
        logger.warning("No game JSONs found.")
        return

    df = pd.DataFrame(records)
    logger.info(f"Saving {len(df)} games to {output_path}...")
    df.to_csv(output_path, index=False)
    logger.info("Done.")

if __name__ == "__main__":
    consolidate_2526_feeds()
