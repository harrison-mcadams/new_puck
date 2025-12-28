"""validate_training_data.py

Validates the training data for the xG models.
Checks for:
1. Feature presence (especially handedness 'shoots_catches').
2. Filtering logic (Empty Net, 1v0).
3. Data consistency across seasons.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from puck import fit_nested_xgs, nhl_api, fit_xgs

def validate_data():
    print("--- VALIDATING TRAINING DATA ---")
    
    # 1. Load Data using fit_nested_xgs method (which uses the new simple load_data)
    # Note: fit_nested_xgs.load_data currently does NOT enrich handedness.
    # fit_xgs.load_all_seasons_data DOES enrich handedness.
    # This comparison is key.
    
    print("\n1. Loading Data via fit_nested_xgs.load_data()...")
    df_nested = fit_nested_xgs.load_data()
    print(f"   Loaded {len(df_nested)} rows.")
    
    print("   Checking for 'shoots_catches'...")
    if 'shoots_catches' in df_nested.columns:
        filled = df_nested['shoots_catches'].notna().mean()
        print(f"   [OK] 'shoots_catches' present. Coverage: {filled:.2%}")
    else:
        print("   [FAIL] 'shoots_catches' MISSING from fit_nested_xgs data load.")

    print("\n2. Loading Data via fit_xgs.load_all_seasons_data() (Reference)...")
    # This might take longer as it fetches API data
    try:
        df_xgs = fit_xgs.load_all_seasons_data()
        print(f"   Loaded {len(df_xgs)} rows.")
        
        if 'shoots_catches' in df_xgs.columns:
            filled = df_xgs['shoots_catches'].notna().mean()
            print(f"   [OK] 'shoots_catches' present. Coverage: {filled:.2%}")
        else:
            print("   [FAIL] 'shoots_catches' MISSING from fit_xgs data load.")
            
    except Exception as e:
        print(f"   [ERROR] Failed to load data via fit_xgs: {e}")

    # 3. Validate Filtering Logic (using df_nested as base, assuming we fix enrichment)
    print("\n3. Validating Filtering Logic (feature validation)...")
    
    # Check Empty Net
    if 'is_net_empty' in df_nested.columns:
        en_count = (df_nested['is_net_empty'] == 1).sum()
        print(f"   Empty Net Events in Raw Data: {en_count}")
        
    # Check Game State outliers
    if 'game_state' in df_nested.columns:
        outliers = df_nested['game_state'].isin(['1v0', '0v1']).sum()
        print(f"   1v0/0v1 Events in Raw Data: {outliers}")

    # Simulate Preprocessing
    print("\n4. Simulating Preprocessing...")
    df_clean = fit_nested_xgs.preprocess_features(df_nested.copy())
    print(f"   Cleaned Rows: {len(df_clean)} (Removed {len(df_nested) - len(df_clean)})")
    
    # Verify removals
    if 'is_net_empty' in df_clean.columns:
        en_rem = (df_clean['is_net_empty'] == 1).sum()
        print(f"   Empty Net Events after Clean: {en_rem} (Should be 0)")
        
    if 'game_state' in df_clean.columns:
        out_rem = df_clean['game_state'].isin(['1v0', '0v1']).sum()
        print(f"   1v0/0v1 Events after Clean: {out_rem} (Should be 0)")
        
    # Verify critical columns for new model
    required_cols = ['distance', 'angle_deg', 'game_state', 'is_goal_layer', 'is_blocked', 'is_on_net']
    missing = [c for c in required_cols if c not in df_clean.columns]
    if missing:
        print(f"   [FAIL] Missing required columns after preprocessing: {missing}")
    else:
        print("   [OK] All target/feature columns present.")

if __name__ == "__main__":
    validate_data()
