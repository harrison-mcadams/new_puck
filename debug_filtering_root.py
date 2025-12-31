
import pandas as pd
from puck import fit_nested_xgs, impute
from pathlib import Path
import sys
import os

def main():
    print("Loading data...")
    df = fit_nested_xgs.load_data()
    
    # Filter Regular Season
    df['game_id_str'] = df['game_id'].astype(str)
    df = df[df['game_id_str'].str.contains(r'^\d{4}02\d{4}$')]
    
    print(f"Total Events (Reg Season): {len(df):,}")
    
    goals = df[df['event'] == 'goal']
    print(f"Total Goals: {len(goals):,}")
    
    # 1. Shootouts
    so_goals = goals[goals['period_number'] == 5]
    print(f"Shootout Goals (P5): {len(so_goals):,}")
    
    # 2. Empty Nets
    if 'is_net_empty' in df.columns:
        en_goals = goals[(goals['is_net_empty'] == 1) | (goals['is_net_empty'] == True)]
        print(f"Empty Net Goals: {len(en_goals):,}")
    else:
        print("is_net_empty column missing")
        
    # 3. Extreme Game States
    if 'game_state' in df.columns:
        extreme_goals = goals[goals['game_state'].isin(['1v0', '0v1'])]
        print(f"Extreme State Goals (1v0/0v1): {len(extreme_goals):,}")

    # 4. Total "Excluded" Goals from xG model perspective
    mask_exclude = (df['period_number'] == 5)
    if 'is_net_empty' in df.columns:
        mask_exclude |= (df['is_net_empty'] == 1) | (df['is_net_empty'] == True)
    if 'game_state' in df.columns:
        mask_exclude |= df['game_state'].isin(['1v0', '0v1'])
        
    excluded_goals = df[mask_exclude & (df['event'] == 'goal')]
    print(f"\nTotal Goals in excluded categories: {len(excluded_goals):,}")
    
    # xG Model Baseline
    df_clean = fit_nested_xgs.preprocess_features(df)
    clean_goals = df_clean[df_clean['event'] == 'goal']
    print(f"Goals in cleaned dataset (xG Universe): {len(clean_goals):,}")
    
    # Difference
    print(f"Discrepancy: {len(goals) - len(clean_goals):,}")

if __name__ == "__main__":
    main()
