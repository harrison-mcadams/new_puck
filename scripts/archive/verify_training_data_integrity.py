
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import fit_xgs, features, fit_nested_xgs

def verify_data():
    print("Loading data...")
    df_raw = fit_xgs.load_all_seasons_data()
    print(f"Loaded {len(df_raw)} rows.")

    # 1. Standard Filtering (Matching train_and_compare_models.py)
    print("Applying standard filters...")
    df_raw['is_goal'] = (df_raw['event'] == 'goal').astype(int)
    
    valid_events = ['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot']
    mask_valid = df_raw['event'].isin(valid_events)
    if 'is_net_empty' in df_raw.columns:
        mask_valid &= (df_raw['is_net_empty'] != 1)
    if 'game_state' in df_raw.columns:
        mask_valid &= ~df_raw['game_state'].isin(['1v0', '0v1'])
    
    df_filtered = df_raw[mask_valid].copy()
    print(f"Filtered to {len(df_filtered)} rows.")

    # 2. Train/Test Split
    df_train, _ = train_test_split(
        df_filtered, test_size=0.2, random_state=42, stratify=df_filtered['is_goal']
    )
    print(f"Training set size: {len(df_train)}")

    # 3. Feature Check
    feature_list = features.get_features('all_inclusive')
    print(f"\nChecking Features: {feature_list}")

    # Preprocess (similar to fit_nested_xgs logic for raw values)
    # We want to check the data state exactly as it goes into imputation/training
    # fit_nested_xgs.preprocess_features mostly ensures columns exist and types match
    df_check = fit_nested_xgs.preprocess_features(df_train.copy())

    # Columns to verify
    # score_diff might need explicit check if it's calculated
    
    issues_found = False

    for col in feature_list:
        if col not in df_check.columns:
            print(f"[ERROR] Column missing cleanly: {col}")
            issues_found = True
            continue
            
        series = df_check[col]
        
        # A. NaNs
        n_nans = series.isna().sum()
        if n_nans > 0:
            print(f"[FAIL] {col}: {n_nans} NaNs found ({n_nans/len(df_check):.2%})")
            # Show a sample
            bad_rows = df_check[series.isna()]
            print(bad_rows[['game_id', 'event', 'x', 'y', 'period']].head())
            if 'period' in bad_rows.columns:
                print(f"Period counts for NaNs in {col}:")
                print(bad_rows['period'].value_counts(dropna=False))
            issues_found = True
        
        # B. Infinite
        if pd.api.types.is_numeric_dtype(series):
            n_inf = np.isinf(series).sum()
            if n_inf > 0:
                print(f"[FAIL] {col}: {n_inf} Infinite values found")
                issues_found = True
                
        # C. Specific Range Checks
        if col == 'distance':
            if (series < 0).any():
                print(f"[WARN] {col}: standard range check (>=0) failed.")
        elif col == 'angle_deg':
            # Updated geometry uses 0-360 or -180/180? Let's check min/max
            print(f"       {col} range: [{series.min():.2f}, {series.max():.2f}]")
            
    with open("verification_report.txt", "w") as f:
        if not issues_found:
            msg = "[SUCCESS] No Nulls or Infinite values found in training features."
            print(f"\n{msg}")
            f.write(msg + "\n")
        else:
            msg = "[FAIL] Data integrity issues detected."
            print(f"\n{msg}")
            f.write(msg + "\n")

if __name__ == "__main__":
    verify_data()
