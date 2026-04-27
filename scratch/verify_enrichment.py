import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from puck import parse
import logging

logging.basicConfig(level=logging.INFO)

def verify():
    # Test with a single game known to have blocked shots
    # Game ID 2023020001 (Opening night 2023)
    game_id = 2023020001
    
    print(f"Fetching and parsing game {game_id}...")
    # We need to mock a 'games' list for _season or just use nhl_api directly
    from puck import nhl_api
    feed = nhl_api.get_game_feed(game_id)
    
    # We'll call _game then manually call the enrichment to simulate what _season does
    df = parse._game(feed)
    print(f"Parsed {len(df)} events.")
    
    blocks = df[df['event'].str.lower() == 'blocked-shot']
    print(f"Found {len(blocks)} blocked shots before enrichment.")
    print("Sample shot types before enrichment:")
    print(blocks['shot_type'].value_counts())
    
    from puck import html_enrichment
    df_enriched = html_enrichment.enrich_blocks_with_html(df, str(game_id))
    
    blocks_enriched = df_enriched[df_enriched['event'].str.lower() == 'blocked-shot']
    print("\nSample shot types after enrichment:")
    print(blocks_enriched['shot_type'].value_counts())
    
    # Check if we have any non-Unknown
    non_unknown = blocks_enriched[~blocks_enriched['shot_type'].str.lower().isin(['unknown', 'nan', 'none'])]
    if not non_unknown.empty:
        print(f"\nSUCCESS: Found {len(non_unknown)} enriched blocked shot types!")
        print(non_unknown[['period', 'period_time', 'shot_type']].head())
    else:
        print("\nFAILURE: No blocked shot types were enriched.")

if __name__ == "__main__":
    verify()
