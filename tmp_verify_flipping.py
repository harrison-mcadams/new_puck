import pandas as pd
import os
import sys

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from puck import analyze, data_pipeline

def verify_flipping(season):
    print(f"Verifying {season}...")
    try:
        csv_path = analyze.locate_season_csv(season)
        df_raw = pd.read_csv(csv_path, nrows=100)
    except:
        print(f"File {season} not found.")
        return

    df_raw = df_raw[df_raw['event'].astype(str).str.lower().str.contains('shot|goal')].copy()
    if 'home_id' in df_raw.columns and 'team_id' in df_raw.columns:
        df_raw['is_home'] = (df_raw['team_id'] == df_raw['home_id']).astype(int)
    
    # Run through pipeline
    df_processed = data_pipeline.preprocess_features(df_raw.head(10), apply_filtering=False)
    
    cols = ['event', 'is_home', 'x', 'y', 'home_team_defending_side']
    print("RAW (from CSV):")
    print(df_raw.head(3)[cols])
    print("\nPROCESSED (thru pipeline):")
    print(df_processed.head(3)[cols])
    print("-" * 40)

if __name__ == "__main__":
    verify_flipping('20232024')
    verify_flipping('20252026')
