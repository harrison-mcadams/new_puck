
import pandas as pd
import numpy as np
import os

def check_high_slot_ratio():
    print("--- Verifying Blocked/Unblocked Ratio in High Slot (2025-2026) ---")
    
    # Path found in listing: data/20252026/20252026_df.csv (Likely the main dataset)
    csv_path = os.path.join('data', '20252026', '20252026_df.csv')
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    print(f"Loading {csv_path}...")
    try:
        # Only load necessary columns for speed
        cols = ['event', 'x', 'y', 'is_net_empty', 'game_state']
        df = pd.read_csv(csv_path, usecols=lambda c: c in cols)
    except Exception as e:
        print(f"Failed to load specific columns, trying full load: {e}")
        df = pd.read_csv(csv_path)

    print(f"Total Rows: {len(df)}")
    
    # Standardize events
    valid_events = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
    df = df[df['event'].isin(valid_events)].copy()
    print(f"Valid Shot Events: {len(df)}")
    
    # Coordinates
    # High Slot Definition:
    # X: 60 to 75 (High Slot) - adjusting for 89ft rink length
    # Y: -10 to 10 (Central)
    # Using Absolute X to handle both sides if data is not standardized to one side
    # But usually 'x' in _df.csv is raw. 
    # Let's assume standard "Attacking Right" (positive X) for simplicity or check both.
    
    # Filter for standard game state if possible (5v5)
    if 'game_state' in df.columns:
        df = df[df['game_state'] == '5v5']
        print(f"5v5 Shot Events: {len(df)}")

    # Define Point / High Zone (X < 53)
    # User specified x < 53. Blue line is 25.
    mask_high_slot = (df['x'].between(25, 53)) & (df['y'].between(-20, 20))
    high_slot = df[mask_high_slot]
    
    n_total = len(high_slot)
    if n_total == 0:
        print("No events found in Point Zone (X:25-53).")
        return

    n_blocked = len(high_slot[high_slot['event'] == 'blocked-shot'])
    n_unblocked = len(high_slot[high_slot['event'].isin(['shot-on-goal', 'goal', 'missed-shot'])])
    
    ratio = n_blocked / n_total if n_total > 0 else 0
    
    print("\n--- POINT ZONE STATISTICS (X=25-53) ---")
    print(f"Total Events: {n_total}")
    print(f"Blocked Shots: {n_blocked}")
    print(f"Unblocked (SOG+Miss+Goal): {n_unblocked}")
    print(f"Blocked %: {ratio:.1%}")
    
    if n_unblocked > n_blocked:
        print("RESULT: Unblocked > Blocked (User is CORRECT about raw data)")
    else:
        print("RESULT: Blocked > Unblocked (Avalanche confirmed in raw data?)")

    # Check Deep Slot for Contrast
    mask_deep = (df['x'].between(75, 85)) & (df['y'].between(-5, 5))
    deep = df[mask_deep]
    n_deep_blocked = len(deep[deep['event'] == 'blocked-shot'])
    print(f"\nDeep Slot Blocked Count: {n_deep_blocked}")
    
    print("----------------------------------------------------------------")

if __name__ == "__main__":
    check_high_slot_ratio()
