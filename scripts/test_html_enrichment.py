import pandas as pd
import sys
import os

sys.path.append(os.getcwd())

from puck import nhl_api, parse, html_enrichment

def test_enrichment(game_id):
    print(f"\n--- Testing HTML Enrichment for {game_id} ---")
    
    # 1. Get API Data
    feed = nhl_api.get_game_feed(game_id)
    df = parse._game(feed)
    # _elaborate adds periodTime_seconds_elapsed
    df = parse._elaborate(df)
    
    # 2. Check blocks before
    blocks_before = df[df['event'].str.lower() == 'blocked-shot'].copy()
    print(f"Blocks found: {len(blocks_before)}")
    print("Shot types before enrichment:")
    print(blocks_before['shot_type'].value_counts() if 'shot_type' in blocks_before.columns else "No shot_type column")
    
    # 3. Enrich
    df_enriched = html_enrichment.enrich_blocks_with_html(df, game_id)
    
    # 4. Check blocks after
    blocks_after = df_enriched[df_enriched['event'].str.lower() == 'blocked-shot'].copy()
    print("\nShot types after enrichment:")
    print(blocks_after['shot_type'].value_counts())
    
    # Show a few examples
    print("\nExamples:")
    print(blocks_after[['period', 'periodTime_seconds_elapsed', 'shot_type']].head(10))

# Test with various games
test_enrichment("2024020151") # WSH vs NYR
