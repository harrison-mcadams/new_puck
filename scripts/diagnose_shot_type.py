import sys
import os
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puck import data_pipeline, config

def main():
    print("Loading 2023-2024 season data...")
    df = pd.read_csv(Path(config.DATA_DIR) / '20232024.csv')
    
    print("\n--- Raw Shot Type Counts ---")
    print(df['shot_type'].value_counts(dropna=False))
    
    print("\n--- Preprocessing (No HTML Enrichment) ---")
    df_p = data_pipeline.preprocess_features(df, apply_html_enrichment=False, apply_filtering=True)

    print("\n--- Shot Type by Event (Proportions) ---")
    cross = pd.crosstab(df_p['event'], df_p['shot_type'], normalize='index')
    print(cross)

    print("\n--- Blocked Shot Counts (Absolute) ---")
    blocks = df_p[df_p['event'] == 'blocked-shot']
    print(blocks['shot_type'].value_counts(dropna=False))

    print("\n--- Shot Type Feature Importance (Modern Era Artifact) ---")
    # I already saw it, but let's confirm what the model sees.

if __name__ == "__main__":
    main()
