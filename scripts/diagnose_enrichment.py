import pandas as pd
import os
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puck import analyze, html_enrichment

def diagnose_unenriched():
    season = '20252026'
    path = analyze.locate_season_csv(season)
    df = pd.read_csv(path)
    
    # Filter for blocked shots
    blocks = df[df['event'].str.lower() == 'blocked-shot'].copy()
    total_blocks = len(blocks)
    
    if total_blocks == 0:
        print("No blocked shots found.")
        return
        
    unknowns = blocks[blocks['shot_type'].isna() | (blocks['shot_type'].str.lower().isin(['unknown', 'none', '']))]
    num_unknown = len(unknowns)
    
    print(f"Season: {season}")
    print(f"Total Blocked Shots: {total_blocks}")
    print(f"Unenriched (Unknown): {num_unknown} ({100.0 * num_unknown / total_blocks:.1f}%)")
    
    if num_unknown == 0:
        return
        
    # Check distribution of games with unknowns
    game_counts = unknowns['game_id'].value_counts()
    print("\nTop games with unenriched blocks:")
    print(game_counts.head(10))
    
    # Take a sample game and try to enrich it manually with verbose output
    sample_game = str(int(game_counts.index[0]))
    print(f"\n--- Debugging Sample Game: {sample_game} ---")
    
    # Fetch HTML and parse manually to see what's in there
    from puck import nhl_api
    try:
        html_text = nhl_api.get_pbp_from_nhl_html(sample_game)
        if not html_text:
            print(f"FAILED: No HTML report found for game {sample_game}")
        else:
            events = html_enrichment.parse_html_pbp(html_text)
            print(f"Parsed {len(events)} events from HTML.")
            
            html_blocks = [e for e in events if e['event_code'] == 'BLOCK']
            print(f"Found {len(html_blocks)} BLOCKS in HTML.")
            
            # Show descriptions of first 5 blocks
            for i, b in enumerate(html_blocks[:5]):
                print(f"  HTML Block {i}: {b['description']} -> Extracted: {b['shot_type']}")
                
            # Check for "Unknown" extraction
            unknown_extractions = [b for b in html_blocks if b['shot_type'] == 'Unknown']
            if unknown_extractions:
                print(f"  WARNING: {len(unknown_extractions)} blocks had 'Unknown' shot type in HTML descriptions.")
                print(f"  Example: {unknown_extractions[0]['description']}")
    except Exception as e:
        print(f"Error debugging game {sample_game}: {e}")

if __name__ == "__main__":
    diagnose_unenriched()
