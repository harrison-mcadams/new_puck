import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import nhl_api
from puck import fit_xgs
from puck import features

def verify():
    print("--- 1. Testing NHL API Bio Fetching ---")
    season = "20232024"
    bios = nhl_api.get_season_player_bios(season)
    if not bios:
        print("ERROR: No bios returned.")
        return
    
    # Check sample
    sample_id = list(bios.keys())[0]
    sample_entry = bios[sample_id]
    print(f"Sample Entry for {sample_id}: {sample_entry}")
    
    if 'positionCode' not in sample_entry:
        print("ERROR: 'positionCode' missing from bio entry.")
    else:
        print("SUCCESS: 'positionCode' found.")

    print("\n--- 2. Testing Data Enrichment ---")
    # Create dummy DF with known IDs
    # 8479318 = Auston Matthews (Center -> F)
    # 8480069 = Cale Makar (Defense -> D)
    df = pd.DataFrame({
        'game_id': [2023020001, 2023020001],
        'player_id': [8479318, 8480069]
    })
    
    # Mock the API return to ensure deterministic test without network if need be, 
    # but we just tested the real API above, so let's trust the real integration or mock if risky.
    # Let's rely on the real call we just made which cached data.
    
    df_enriched = fit_xgs.enrich_data_with_bios(df)
    
    print("Enriched Columns:", df_enriched.columns.tolist())
    print("\nData:")
    print(df_enriched[['player_id', 'shoots_catches', 'shooter_role']])
    
    row_matthews = df_enriched[df_enriched['player_id'] == 8479318].iloc[0]
    row_makar = df_enriched[df_enriched['player_id'] == 8480069].iloc[0]
    
    if row_matthews['shooter_role'] == 'F':
        print("SUCCESS: Matthews identified as F")
    else:
        print(f"FAILURE: Matthews identified as {row_matthews['shooter_role']}")

    if row_makar['shooter_role'] == 'D':
        print("SUCCESS: Makar identified as D")
    else:
        print(f"FAILURE: Makar identified as {row_makar['shooter_role']}")

    print("\n--- 3. Testing Feature Configuration ---")
    all_features = features.get_features('all_inclusive')
    if 'shooter_role' in all_features:
        print("SUCCESS: 'shooter_role' found in all_inclusive features.")
    else:
        print("FAILURE: 'shooter_role' NOT found in all_inclusive features.")

if __name__ == "__main__":
    verify()
