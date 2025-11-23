"""
Demonstration script for interval edge case handling.

This script creates a concrete example of the power-play goal edge case
mentioned in the problem statement and demonstrates how the post-filter
validation correctly handles it.
"""

import pandas as pd
import sys
import os

# Add parent directory to path
test_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(test_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import analyze


def demonstrate_powerplay_goal_edge_case():
    """
    Demonstrate the power-play goal edge case scenario.
    
    Scenario:
    - Team is on power play (5v4) from t=50 to t=100
    - Team scores a goal at exactly t=100
    - At t=100, the penalty expires and game state becomes 5v5
    - Question: Should the goal be counted in 5v4 or 5v5 statistics?
    
    Answer: The goal should be counted in 5v4 statistics because that was
    the game state when the goal was scored, even though the penalty expires
    at the same timestamp.
    """
    
    print("="*70)
    print("POWER-PLAY GOAL EDGE CASE DEMONSTRATION")
    print("="*70)
    print()
    print("Scenario:")
    print("  - Team PHI is on power play (5v4) from t=50 to t=100 seconds")
    print("  - PHI scores a goal at exactly t=100 seconds")
    print("  - At t=100, penalty expires and state becomes 5v5")
    print("  - Interval [100, 150] represents 5v5 play")
    print()
    
    # Create a realistic sequence of events
    rows = [
        # Penalty is called at t=40, creating 5v4 situation
        {'game_id': 2025020339, 'total_time_elapsed_seconds': 40, 'x': 0, 'y': 0,
         'event': 'penalty', 'team_id': 2, 'home_id': 1, 'away_id': 2,
         'home_abb': 'PHI', 'away_abb': 'NYR', 'game_state': '5v4', 'is_net_empty': 0},
        
        # Faceoff starts 5v4 play
        {'game_id': 2025020339, 'total_time_elapsed_seconds': 50, 'x': -30, 'y': 20,
         'event': 'faceoff', 'team_id': 1, 'home_id': 1, 'away_id': 2,
         'home_abb': 'PHI', 'away_abb': 'NYR', 'game_state': '5v4', 'is_net_empty': 0},
        
        # Shot during power play
        {'game_id': 2025020339, 'total_time_elapsed_seconds': 75, 'x': -85, 'y': 5,
         'event': 'shot-on-goal', 'team_id': 1, 'home_id': 1, 'away_id': 2,
         'home_abb': 'PHI', 'away_abb': 'NYR', 'game_state': '5v4', 'is_net_empty': 0},
        
        # POWER-PLAY GOAL at exactly t=100 (boundary time)
        # This goal occurs during 5v4, even though the penalty expires at t=100
        {'game_id': 2025020339, 'total_time_elapsed_seconds': 100, 'x': -89, 'y': 0,
         'event': 'goal', 'team_id': 1, 'home_id': 1, 'away_id': 2,
         'home_abb': 'PHI', 'away_abb': 'NYR', 'game_state': '5v4', 'is_net_empty': 0},
        
        # Faceoff after goal - now at 5v5
        {'game_id': 2025020339, 'total_time_elapsed_seconds': 100, 'x': 0, 'y': 0,
         'event': 'faceoff', 'team_id': 1, 'home_id': 1, 'away_id': 2,
         'home_abb': 'PHI', 'away_abb': 'NYR', 'game_state': '5v5', 'is_net_empty': 0},
        
        # Regular 5v5 play continues
        {'game_id': 2025020339, 'total_time_elapsed_seconds': 125, 'x': -60, 'y': -10,
         'event': 'shot-on-goal', 'team_id': 1, 'home_id': 1, 'away_id': 2,
         'home_abb': 'PHI', 'away_abb': 'NYR', 'game_state': '5v5', 'is_net_empty': 0},
    ]
    
    df = pd.DataFrame(rows)
    df['xgs'] = 0.5  # Add xgs to avoid model training
    
    print("Events in sequence:")
    print("-" * 70)
    for _, row in df.iterrows():
        print(f"  t={row['total_time_elapsed_seconds']:3.0f}s: {row['event']:15s} "
              f"game_state={row['game_state']:3s} "
              f"{'<-- POWER-PLAY GOAL' if row['event'] == 'goal' else ''}")
    print()
    
    # Intervals representing 5v5 play (after the power play ends)
    intervals_obj = {
        'per_game': {
            2025020339: {
                'sides': {
                    'team': {
                        'intersection_intervals': [[100, 150]]  # 5v5 interval
                    }
                }
            }
        }
    }
    
    # Test 1: Filter for 5v5 events
    print("Test 1: Filtering for 5v5 game state")
    print("-" * 70)
    condition_5v5 = {'game_state': ['5v5'], 'is_net_empty': [0], 'team': 1}
    
    _, _, filtered_5v5, _ = analyze.xgs_map(
        season='20252026',
        data_df=df,
        use_intervals=True,
        intervals_input=intervals_obj,
        condition=condition_5v5,
        return_heatmaps=False,
        return_filtered_df=True,
        behavior='load'
    )
    
    print(f"  Requested: game_state=['5v5']")
    print(f"  Interval: [100, 150]")
    print(f"  Matched events: {len(filtered_5v5)}")
    
    if filtered_5v5 is not None:
        for _, row in filtered_5v5.iterrows():
            print(f"    t={row['total_time_elapsed_seconds']:3.0f}s: {row['event']:15s} game_state={row['game_state']}")
        
        # Check if power-play goal was incorrectly included
        ppg = filtered_5v5[(filtered_5v5['event'] == 'goal') & 
                           (filtered_5v5['total_time_elapsed_seconds'] == 100)]
        if ppg.empty:
            print("\n  ✓ CORRECT: Power-play goal at t=100 NOT included in 5v5 filter")
            print("    (Goal occurred during 5v4, even though penalty expired at t=100)")
        else:
            print("\n  ✗ INCORRECT: Power-play goal was included in 5v5 filter!")
    
    print()
    
    # Test 2: Filter for 5v4 events (power play)
    # Note: We need to create intervals for 5v4 to demonstrate this
    intervals_5v4 = {
        'per_game': {
            2025020339: {
                'sides': {
                    'team': {
                        'intersection_intervals': [[50, 100]]  # 5v4 interval
                    }
                }
            }
        }
    }
    
    print("Test 2: Filtering for 5v4 game state (power play)")
    print("-" * 70)
    condition_5v4 = {'game_state': ['5v4'], 'is_net_empty': [0], 'team': 1}
    
    _, _, filtered_5v4, _ = analyze.xgs_map(
        season='20252026',
        data_df=df,
        use_intervals=True,
        intervals_input=intervals_5v4,
        condition=condition_5v4,
        return_heatmaps=False,
        return_filtered_df=True,
        behavior='load'
    )
    
    print(f"  Requested: game_state=['5v4']")
    print(f"  Interval: [50, 100]")
    print(f"  Matched events: {len(filtered_5v4)}")
    
    if filtered_5v4 is not None:
        for _, row in filtered_5v4.iterrows():
            print(f"    t={row['total_time_elapsed_seconds']:3.0f}s: {row['event']:15s} game_state={row['game_state']}")
        
        # Check if power-play goal was correctly included
        ppg = filtered_5v4[(filtered_5v4['event'] == 'goal') & 
                           (filtered_5v4['total_time_elapsed_seconds'] == 100)]
        if not ppg.empty:
            print("\n  ✓ CORRECT: Power-play goal at t=100 IS included in 5v4 filter")
            print("    (Goal occurred during 5v4, credited to power play)")
        else:
            print("\n  ✗ INCORRECT: Power-play goal was NOT included in 5v4 filter!")
    
    print()
    print("="*70)
    print("CONCLUSION")
    print("="*70)
    print()
    print("The post-filter validation in _apply_intervals correctly handles")
    print("boundary events by checking each event's actual game_state attribute")
    print("rather than relying solely on time-interval membership.")
    print()
    print("This ensures that:")
    print("  - Power-play goals are credited to the power play (5v4)")
    print("  - Events are not double-counted across adjacent intervals")
    print("  - Statistics accurately reflect the game state when events occurred")
    print()


if __name__ == '__main__':
    try:
        demonstrate_powerplay_goal_edge_case()
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
