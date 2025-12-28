
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from puck.parse import _elaborate

def test_rush_logic():
    # Setup test scenarios
    # Assumes standard rink: goals at -89 and +89. Blue lines at +/- 25.
    
    rows = [
        # Scenario 1: Standard Rush (Right)
        # Event 1: Faceoff at Center (0,0)
        {
            'event': 'faceoff',
            'team_id': 1, 'period': 1, 'periodTime': '00:00', 'periodTimeType': 'elapsed',
            'x': 0.0, 'y': 0.0,
            'home_id': 1, 'away_id': 2, 'home_team_defending_side': 'left' 
            # Home(1) defends Left (-89). Away(2) defends Right (+89).
            # If Team 1 shoots, they attack Right (+89).
        },
        # Event 2: Shot by Team 1 (Attacking Right)
        # Last event X=0. Attacking Right implies Defending Zone X > 25.
        # 0 <= 25 is TRUE. Time 3s <= 5s. -> RUSH.
        {
            'event': 'shot-on-goal',
            'team_id': 1, 'period': 1, 'periodTime': '00:03', 'periodTimeType': 'elapsed',
            'x': 80.0, 'y': 0.0,
            'home_id': 1, 'away_id': 2, 'home_team_defending_side': 'left'
        },
        
        # Scenario 2: No Rush (Inside Zone)
        # Event 3: Block in defending zone (relative to Team 1 attacking)
        # Team 1 attacking Right (+89). Defending Zone is X > 25.
        # Event at X=80.
        {
            'event': 'blocked-shot',
            'team_id': 2, 'period': 1, 'periodTime': '00:10', 'periodTimeType': 'elapsed',
            'x': 80.0, 'y': 0.0,
            'home_id': 1, 'away_id': 2, 'home_team_defending_side': 'left'
        },
        # Event 4: Shot by Team 1
        # Last event 80. 80 is NOT <= 25. -> NO RUSH.
        {
            'event': 'shot-on-goal',
            'team_id': 1, 'period': 1, 'periodTime': '00:12', 'periodTimeType': 'elapsed',
            'x': 85.0, 'y': 0.0,
            'home_id': 1, 'away_id': 2, 'home_team_defending_side': 'left'
        },
        
        # Scenario 3: Rush (Left)
        # Event 5: Turnover at Center (-10)
        {
            'event': 'giveaway',
            'team_id': 2, 'period': 1, 'periodTime': '00:20', 'periodTimeType': 'elapsed',
            'x': -10.0, 'y': 0.0,
            'home_id': 1, 'away_id': 2, 'home_team_defending_side': 'left'
        },
        # Event 6: Shot by Team 2 (Attacking Left Goal -89)
        # If attacking Left, Defending Zone is X < -25.
        # Last event X=-10. -10 >= -25 is TRUE.
        # Time 3s <= 5s. -> RUSH.
        {
            'event': 'shot-on-goal',
            'team_id': 2, 'period': 1, 'periodTime': '00:23', 'periodTimeType': 'elapsed',
            'x': -80.0, 'y': 0.0,
            'home_id': 1, 'away_id': 2, 'home_team_defending_side': 'left'
        },
        
        # Scenario 4: No Rush (Time Exceeded)
        # Event 7: Faceoff at Center
        {
            'event': 'faceoff',
            'team_id': 1, 'period': 1, 'periodTime': '00:40', 'periodTimeType': 'elapsed',
            'x': 0.0, 'y': 0.0,
            'home_id': 1, 'away_id': 2, 'home_team_defending_side': 'left'
        },
        # Event 8: Shot 10s later
        {
            'event': 'shot-on-goal',
            'team_id': 1, 'period': 1, 'periodTime': '00:50', 'periodTimeType': 'elapsed',
            'x': 80.0, 'y': 0.0,
            'home_id': 1, 'away_id': 2, 'home_team_defending_side': 'left'
        },
    ]

    df = _elaborate(rows)
    
    # Check Event 2 (Rush Right)
    print("Event 2 is_rush:", df.iloc[1]['is_rush'])
    assert df.iloc[1]['is_rush'] == 1, "Scenario 1 Failed: Should be Rush"
    
    # Check Event 4 (No Rush - Inside Zone)
    print("Event 4 is_rush:", df.iloc[3]['is_rush'])
    assert df.iloc[3]['is_rush'] == 0, "Scenario 2 Failed: Should NOT be Rush"
    
    # Check Event 6 (Rush Left)
    print("Event 6 is_rush:", df.iloc[5]['is_rush'])
    assert df.iloc[5]['is_rush'] == 1, "Scenario 3 Failed: Should be Rush"

    # Check Event 8 (No Rush - Time)
    print("Event 8 is_rush:", df.iloc[7]['is_rush'])
    assert df.iloc[7]['is_rush'] == 0, "Scenario 4 Failed: Should NOT be Rush (Time)"

if __name__ == "__main__":
    try:
        test_rush_logic()
        print("ALL TESTS PASSED")
    except AssertionError as e:
        print(f"TEST FAILED: {e}")
        sys.exit(1)
