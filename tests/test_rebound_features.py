import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from puck.parse import _elaborate

def test_rebound_logic():
    # Mock data: a series of events
    # Event 1: Shot by Team A
    # Event 2: Faceoff (precedes another shot)
    # Event 3: Shot by Team A (2 seconds later) -> REBOUND
    # Event 4: Shot by Team B -> NOT REBOUND
    # Event 5: Shot by Team A (10 seconds later) -> NOT REBOUND (threshold exceeded)
    
    rows = [
        # Event 1
        {
            'event': 'shot-on-goal',
            'team_id': 1,
            'period': 1,
            'periodTime': '00:10',
            'periodTimeType': 'elapsed',
            'x': 50.0,
            'y': 20.0,
            'home_id': 1,
            'away_id': 2,
            'home_team_defending_side': 'left'
        },
        # Event 2: Faceoff
        {
            'event': 'faceoff',
            'team_id': 1,
            'period': 1,
            'periodTime': '00:11',
            'periodTimeType': 'elapsed',
            'x': 0.0,
            'y': 0.0,
            'home_id': 1,
            'away_id': 2,
            'home_team_defending_side': 'left'
        },
        # Event 3: Rebound for Team A
        {
            'event': 'shot-on-goal',
            'team_id': 1,
            'period': 1,
            'periodTime': '00:12',
            'periodTimeType': 'elapsed',
            'x': 55.0,
            'y': 22.0,
            'home_id': 1,
            'away_id': 2,
            'home_team_defending_side': 'left'
        },
        # Event 4: Shot by Team B
        {
            'event': 'shot-on-goal',
            'team_id': 2,
            'period': 1,
            'periodTime': '00:15',
            'periodTimeType': 'elapsed',
            'x': -50.0,
            'y': -20.0,
            'home_id': 1,
            'away_id': 2,
            'home_team_defending_side': 'left'
        },
        # Event 5: Late shot by Team A
        {
            'event': 'shot-on-goal',
            'team_id': 1,
            'period': 1,
            'periodTime': '00:25',
            'periodTimeType': 'elapsed',
            'x': 60.0,
            'y': 10.0,
            'home_id': 1,
            'away_id': 2,
            'home_team_defending_side': 'left'
        }
    ]
    
    df = _elaborate(rows)
    
    # Assertions
    assert len(df) == 5
    
    # Event 1: First shot
    assert df.iloc[0]['is_rebound'] == 0
    assert df.iloc[0]['last_event_type'] is None
    
    # Event 2: Faceoff
    assert df.iloc[1]['last_event_type'] == 'shot-on-goal'
    assert df.iloc[1]['last_event_time_diff'] == 1.0
    
    # Event 3: Rebound
    # Total time diff between shot 1 (10s) and shot 2 (12s) is 2s.
    assert df.iloc[2]['is_rebound'] == 1
    assert df.iloc[2]['rebound_time_diff'] == 2.0
    assert df.iloc[2]['last_event_type'] == 'faceoff' # Preceding event was faceoff
    assert df.iloc[2]['last_event_time_diff'] == 1.0
    assert df.iloc[2]['rebound_angle_change'] is not None
    
    # Event 4: Shot by Team B (no prior shot by Team B)
    assert df.iloc[3]['is_rebound'] == 0
    assert df.iloc[3]['last_event_type'] == 'shot-on-goal'
    
    # Event 5: Late shot by Team A (13s after Event 3)
    assert df.iloc[4]['is_rebound'] == 0
    assert df.iloc[4]['last_event_type'] == 'shot-on-goal'

if __name__ == "__main__":
    test_rebound_logic()
    print("Test passed!")
