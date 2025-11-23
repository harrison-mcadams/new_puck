"""
Test interval-based filtering edge cases.

This test validates that events at boundary times (e.g., power-play goals occurring
at the exact moment when game state changes) are correctly included/excluded based
on their actual game_state, not just time interval membership.
"""

import pandas as pd
import sys
import os

# Add parent directory to path - using relative path from test location
test_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(test_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import analyze


def test_boundary_event_game_state_validation():
    """
    Test that events at interval boundaries are validated by game_state.
    
    Scenario:
    - Event at time 100 occurs at the boundary between two intervals
    - Interval 1: [50, 100] with game_state '5v4'
    - Interval 2: [100, 150] with game_state '5v5'
    - The event at time 100 has game_state '5v4'
    
    Expected behavior when filtering with condition {'game_state': ['5v5']}:
    - The event at time 100 should be excluded because its game_state is '5v4', not '5v5'
    """
    # Create test dataframe with events at boundary times
    rows = [
        # Game 1 events
        {'game_id': 'G1', 'total_time_elapsed_seconds': 50, 'x': -40, 'y': 0, 
         'event': 'shot-on-goal', 'team_id': 1, 'home_id': 1, 'away_id': 2,
         'home_abb': 'PHI', 'away_abb': 'NYR', 'game_state': '5v4', 'is_net_empty': 0},
        
        {'game_id': 'G1', 'total_time_elapsed_seconds': 75, 'x': -30, 'y': 5,
         'event': 'shot-on-goal', 'team_id': 1, 'home_id': 1, 'away_id': 2,
         'home_abb': 'PHI', 'away_abb': 'NYR', 'game_state': '5v4', 'is_net_empty': 0},
        
        # Boundary event: at exact time when state changes from 5v4 to 5v5
        {'game_id': 'G1', 'total_time_elapsed_seconds': 100, 'x': -20, 'y': -5,
         'event': 'goal', 'team_id': 1, 'home_id': 1, 'away_id': 2,
         'home_abb': 'PHI', 'away_abb': 'NYR', 'game_state': '5v4', 'is_net_empty': 0},
        
        {'game_id': 'G1', 'total_time_elapsed_seconds': 100, 'x': -15, 'y': 2,
         'event': 'faceoff', 'team_id': 1, 'home_id': 1, 'away_id': 2,
         'home_abb': 'PHI', 'away_abb': 'NYR', 'game_state': '5v5', 'is_net_empty': 0},
        
        {'game_id': 'G1', 'total_time_elapsed_seconds': 125, 'x': -10, 'y': 1,
         'event': 'shot-on-goal', 'team_id': 1, 'home_id': 1, 'away_id': 2,
         'home_abb': 'PHI', 'away_abb': 'NYR', 'game_state': '5v5', 'is_net_empty': 0},
    ]
    
    df = pd.DataFrame(rows)
    
    # Create intervals that span the boundary
    # Note: timing intervals typically represent when a condition is true
    # Here we simulate that 5v5 state is active during [100, 150]
    intervals_obj = {
        'per_game': {
            'G1': {
                'sides': {
                    'team': {
                        'intersection_intervals': [[100, 150]]  # 5v5 interval
                    }
                }
            }
        }
    }
    
    # Test with condition requesting 5v5 game state
    condition = {'game_state': ['5v5'], 'is_net_empty': [0], 'team': 1}
    
    # Add xgs column to avoid model training
    df['xgs'] = 0.5
    
    try:
        out_path, heatmaps, filtered_df, summary = analyze.xgs_map(
            season='20252026',
            data_df=df,
            use_intervals=True,
            intervals_input=intervals_obj,
            condition=condition,
            return_heatmaps=False,
            return_filtered_df=True,
            behavior='load'  # Don't try to train model
        )
        
        print("\n=== Test: Boundary Event Game State Validation ===")
        print(f"Input dataframe: {len(df)} rows")
        print(f"Filtered dataframe: {len(filtered_df) if filtered_df is not None else 0} rows")
        
        if filtered_df is not None and not filtered_df.empty:
            print("\nFiltered events:")
            for idx, row in filtered_df.iterrows():
                print(f"  Time: {row['total_time_elapsed_seconds']}, Event: {row['event']}, "
                      f"Game State: {row.get('game_state', 'N/A')}")
            
            # Validate that the goal at time 100 with game_state '5v4' is excluded
            goal_at_100 = filtered_df[
                (filtered_df['total_time_elapsed_seconds'] == 100) & 
                (filtered_df['event'] == 'goal')
            ]
            
            if not goal_at_100.empty:
                print("\n❌ FAIL: Goal at time 100 (5v4) should be excluded from 5v5 filter")
                print(f"   Found: {len(goal_at_100)} goal event(s) at boundary")
                return False
            else:
                print("\n✓ PASS: Goal at time 100 (5v4) correctly excluded from 5v5 filter")
            
            # Validate that the faceoff at time 100 with game_state '5v5' is included
            faceoff_at_100 = filtered_df[
                (filtered_df['total_time_elapsed_seconds'] == 100) & 
                (filtered_df['event'] == 'faceoff')
            ]
            
            if faceoff_at_100.empty:
                print("❌ FAIL: Faceoff at time 100 (5v5) should be included in 5v5 filter")
                return False
            else:
                print("✓ PASS: Faceoff at time 100 (5v5) correctly included in 5v5 filter")
            
            # Validate that event at time 125 (clearly in 5v5) is included
            event_at_125 = filtered_df[filtered_df['total_time_elapsed_seconds'] == 125]
            if event_at_125.empty:
                print("❌ FAIL: Event at time 125 (5v5) should be included")
                return False
            else:
                print("✓ PASS: Event at time 125 (5v5) correctly included")
                
            return True
        else:
            print("\n⚠ WARNING: Filtered dataframe is empty or None")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_is_net_empty_validation():
    """
    Test that events are validated by is_net_empty condition at boundaries.
    
    Scenario:
    - Event at time 200 at boundary where net becomes empty
    - Should be excluded if condition requires is_net_empty=0 but event has is_net_empty=1
    """
    rows = [
        {'game_id': 'G2', 'total_time_elapsed_seconds': 190, 'x': -40, 'y': 0,
         'event': 'shot-on-goal', 'team_id': 1, 'home_id': 1, 'away_id': 2,
         'home_abb': 'PHI', 'away_abb': 'BOS', 'game_state': '5v5', 'is_net_empty': 0},
        
        # Boundary event: net becomes empty at time 200
        {'game_id': 'G2', 'total_time_elapsed_seconds': 200, 'x': -30, 'y': 5,
         'event': 'shot-on-goal', 'team_id': 1, 'home_id': 1, 'away_id': 2,
         'home_abb': 'PHI', 'away_abb': 'BOS', 'game_state': '5v5', 'is_net_empty': 1},
        
        {'game_id': 'G2', 'total_time_elapsed_seconds': 210, 'x': -20, 'y': -5,
         'event': 'shot-on-goal', 'team_id': 1, 'home_id': 1, 'away_id': 2,
         'home_abb': 'PHI', 'away_abb': 'BOS', 'game_state': '5v5', 'is_net_empty': 0},
    ]
    
    df = pd.DataFrame(rows)
    
    intervals_obj = {
        'per_game': {
            'G2': {
                'sides': {
                    'team': {
                        'intersection_intervals': [[180, 220]]
                    }
                }
            }
        }
    }
    
    condition = {'game_state': ['5v5'], 'is_net_empty': [0], 'team': 1}
    
    # Add xgs column to avoid model training
    df['xgs'] = 0.5
    
    try:
        out_path, heatmaps, filtered_df, summary = analyze.xgs_map(
            season='20252026',
            data_df=df,
            use_intervals=True,
            intervals_input=intervals_obj,
            condition=condition,
            return_heatmaps=False,
            return_filtered_df=True,
            behavior='load'  # Don't try to train model
        )
        
        print("\n=== Test: is_net_empty Validation ===")
        print(f"Input dataframe: {len(df)} rows")
        print(f"Filtered dataframe: {len(filtered_df) if filtered_df is not None else 0} rows")
        
        if filtered_df is not None and not filtered_df.empty:
            print("\nFiltered events:")
            for idx, row in filtered_df.iterrows():
                print(f"  Time: {row['total_time_elapsed_seconds']}, is_net_empty: {row.get('is_net_empty', 'N/A')}")
            
            # Validate that the event at time 200 with is_net_empty=1 is excluded
            empty_net_event = filtered_df[
                (filtered_df['total_time_elapsed_seconds'] == 200) &
                (filtered_df['is_net_empty'] == 1)
            ]
            
            if not empty_net_event.empty:
                print("\n❌ FAIL: Event at time 200 (is_net_empty=1) should be excluded")
                return False
            else:
                print("\n✓ PASS: Event at time 200 (is_net_empty=1) correctly excluded")
            
            # Events at 190 and 210 (is_net_empty=0) should be included
            valid_events = filtered_df[filtered_df['is_net_empty'] == 0]
            if len(valid_events) == 2:
                print("✓ PASS: Events with is_net_empty=0 correctly included")
                return True
            else:
                print(f"❌ FAIL: Expected 2 events with is_net_empty=0, got {len(valid_events)}")
                return False
        else:
            print("\n⚠ WARNING: Filtered dataframe is empty or None")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("Running interval edge case tests...\n")
    
    test1_passed = test_boundary_event_game_state_validation()
    test2_passed = test_is_net_empty_validation()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Boundary Event Game State Validation: {'PASS' if test1_passed else 'FAIL'}")
    print(f"is_net_empty Validation: {'PASS' if test2_passed else 'FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)
