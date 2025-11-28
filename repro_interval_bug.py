
import pandas as pd
import timing_new as timing
import nhl_api
import sys

# Get a recent game for PHI
try:
    game_id = nhl_api.get_game_id(team='PHI')
    print(f"Got recent PHI game: {game_id}")
except Exception as e:
    print(f"Failed to get game: {e}")
    sys.exit(1)

# Determine if PHI is Home or Away
feed = nhl_api.get_game_feed(game_id)
home_abb = feed.get('homeTeam', {}).get('abbrev') or feed.get('home', {}).get('abbrev')
away_abb = feed.get('awayTeam', {}).get('abbrev') or feed.get('away', {}).get('abbrev')

print(f"Home: {home_abb}, Away: {away_abb}")

phi_is_home = (home_abb == 'PHI')
phi_is_away = (away_abb == 'PHI')

if not (phi_is_home or phi_is_away):
    print("PHI not found in this game??")
    sys.exit(1)

target_team = 'PHI'
print(f"Target Team: {target_team} (Home={phi_is_home}, Away={phi_is_away})")

# 1. Compute intervals for '5v4' (Home 5, Away 4)
print(f"\nComputing intervals for game {game_id} with condition {{'game_state': ['5v4'], 'team': '{target_team}'}}")
res_5v4 = timing.compute_intervals_for_game(game_id, {'game_state': ['5v4'], 'team': target_team})
intervals_5v4 = res_5v4['intervals_per_condition'].get('game_state', [])
print(f"Found {len(intervals_5v4)} intervals for 5v4 (Home 5, Away 4)")

# 2. Compute intervals for '4v5' (Home 4, Away 5)
print(f"\nComputing intervals for game {game_id} with condition {{'game_state': ['4v5'], 'team': '{target_team}'}}")
res_4v5 = timing.compute_intervals_for_game(game_id, {'game_state': ['4v5'], 'team': target_team})
intervals_4v5 = res_4v5['intervals_per_condition'].get('game_state', [])
print(f"Found {len(intervals_4v5)} intervals for 4v5 (Home 4, Away 5)")

# Analysis
print("\n--- ANALYSIS ---")
if phi_is_home:
    print("PHI is HOME.")
    print(f"5v4 (Home 5, Away 4) = PHI Power Play. Count: {len(intervals_5v4)}")
    print(f"4v5 (Home 4, Away 5) = PHI Penalty Kill. Count: {len(intervals_4v5)}")
else:
    print("PHI is AWAY.")
    print(f"5v4 (Home 5, Away 4) = PHI Penalty Kill (Opp 5, PHI 4). Count: {len(intervals_5v4)}")
    print(f"4v5 (Home 4, Away 5) = PHI Power Play (Opp 4, PHI 5). Count: {len(intervals_4v5)}")

print("\nIf the user requested '5v4' expecting PHI Power Play:")
if phi_is_home:
    print("  Matches 5v4 (Correct).")
else:
    print("  Matches 4v5 (BUT code requested 5v4, so they got Penalty Kill instead!)")

