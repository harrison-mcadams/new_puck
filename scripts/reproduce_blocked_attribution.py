
import sys
import os
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import analyze

def test_blocked_attribution():
    print("--- Testing Blocked Shot Attribution ---")
    
    # Mock Data: Team 100 blocks a shot from Team 200
    # In NHL API, 'team_id' for blocked-shot is the blocker (100).
    df = pd.DataFrame([{
        'event': 'blocked-shot',
        'x': 50.0,
        'y': 0.0,
        'team_id': 100, # Blocker
        'home_id': 100,
        'away_id': 200, # Shooter
        'home_abb': 'BLK',
        'away_abb': 'SHT',
        'home_team_defending_side': 'left', 
        'period': 1,
        'period_time': '10:00',
        'xgs': 0.5 # Mock xG
    }])
    
    # Analyze for Team 100 (The Blocker)
    # We expect this to be xG AGAINST them (since they blocked it, it was a shot at them).
    # But current logic might assign it as xG FOR them (team_xgs).
    
    print("\nAnalyzing for Team 100 (Blocker):")
    stats_blk = {'team_xgs': 0.0, 'other_xgs': 0.0}
    
    # --- TEST DISTANCE CALCULATION LOGIC ---
    print("\n--- Testing Coordinate Logic (correction.py) ---")
    
    from puck import correction
    
    # Mock Dataframe for correction test
    # Scenario:
    # Home Team (100) Defending LEFT (-89)
    # Away Team (200) Shooting at LEFT NET (-89)
    # Block happens at x = -80 (Defensive Zone of Home)
    
    # Original Data (Pre-Swap)
    # team_id = 100 (Blocker)
    # home_id = 100
    # away_id = 200
    # event = 'blocked-shot'
    # home_team_defending_side = 'left'
    # x = -80, y = 0
    
    df_mock = pd.DataFrame([{
        'game_id': 2025020001,
        'event': 'blocked-shot',
        'team_id': 100,
        'home_id': 100,
        'away_id': 200,
        'home_team_defending_side': 'left',
        'x': -80.0,
        'y': 0.0,
        'distance': 999.0, # Bad original distance
        'angle_deg': 0.0
    }])
    
    print("Mock Input:")
    print(df_mock[['team_id', 'x', 'distance']])
    
    # Apply Correction
    df_fixed = correction.fix_blocked_shot_attribution(df_mock)
    
    print("\nCorrected Output:")
    print(df_fixed[['team_id', 'x', 'distance']])
    
    # Verification
    # New team_id should be 200 (Shooter)
    # target net should be -89 (Left)
    # distance should be |-80 - (-89)| = 9.0
    
    new_team = df_fixed.iloc[0]['team_id']
    new_dist = df_fixed.iloc[0]['distance']
    
    print(f"\nExpected Team: 200, Actual: {new_team}")
    print(f"Expected Dist: 9.0, Actual: {new_dist}")
    
    if new_team == 200 and abs(new_dist - 9.0) < 1.0:
        print("RESULT: SUCCESS. Coordinate logic is correct.")
    else:
        print("RESULT: FAILURE. Coordinate logic is flawed.")
        if new_dist > 100:
            print("  -> ERROR: Distance is huge! Likely targeting WRONG NET (Right, 89).")
        
    print("\n--- Invoking analyze.xgs_map for Team 200 ---")

    try:
        out_path, heat, df_res, stats = analyze.xgs_map(
            season='test',
            data_df=df.copy(),
            stats_only=True,
            condition={'team': 200},
            show=False
        )
        
        team_xgs_2 = stats['team_xgs']
        other_xgs_2 = stats['other_xgs']
        
        print(f"Team 200 (Shooter) Stats from xgs_map: Team xG = {team_xgs_2}, Other xG = {other_xgs_2}")
        
        if team_xgs_2 > 0:
             print("RESULT: SUCCESS. Blocked shot attributed to Shooter (200).")
        else:
             print("RESULT: FAILURE. Blocked shot NOT attributed to Shooter (200).")

    except Exception as e:
        print(f"Error calling xgs_map: {e}")


if __name__ == "__main__":
    test_blocked_attribution()
