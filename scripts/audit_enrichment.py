import pandas as pd
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puck import analyze, data_pipeline

def audit_enrichment():
    season = '20252026'
    path = analyze.locate_season_csv(season)
    df = pd.read_csv(path)
    
    print(f"Auditing enrichment for {season}...")
    # Call preprocess_features with enrichment enabled
    df_enriched = data_pipeline.preprocess_features(df, apply_html_enrichment=True, verbose=True)
    
    blocks = df_enriched[df_enriched['event'].str.lower() == 'blocked-shot']
    total = len(blocks)
    unknowns = blocks[blocks['shot_type'].str.lower().isin(['unknown', 'none', ''])]
    
    print(f"\nFinal Enrichment Audit for {season}:")
    print(f"Total Blocked Shots: {total}")
    print(f"Unenriched (Unknown): {len(unknowns)} ({100.0 * len(unknowns) / total:.1f}%)")
    
    # Check games with many unknowns
    if len(unknowns) > 0:
        bad_games = unknowns['game_id'].value_counts()
        print("\nGames with highest UNENRICHED blocks:")
        print(bad_games.head(10))
        
        # Check if these games have ANY enriched blocks
        sample_bad_game = bad_games.index[0]
        enriched_in_bad = blocks[(blocks['game_id'] == sample_bad_game) & (~blocks['shot_type'].str.lower().isin(['unknown', 'none', '']))]
        print(f"\nGame {sample_bad_game}: {len(enriched_in_bad)} enriched, {bad_games.iloc[0]} unknowns.")

if __name__ == "__main__":
    audit_enrichment()
