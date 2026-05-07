import pandas as pd
import numpy as np
from puck import timing
from puck import get_game_state

game_id = 2025020058
season = '20252026'

df_shifts = timing._get_shifts_df(game_id, season=season)
print(f"Game {game_id}: {len(df_shifts)} shifts found.")
print(f"Teams: {df_shifts['team_id'].unique()}")

# ints, filt = get_game_state.get_game_state(game_id, condition={'game_state': ['5v5']}, return_df=True, df_shifts=df_shifts)
cond = {'game_state': ['5v5'], 'is_net_empty': [0]}
intervals = timing.get_game_intervals_cached(game_id, season, cond, force_refresh=True)
if intervals:
    total_5v5 = sum(e-s for s,e in intervals)
    print(f"Total 5v5 Time (Cached): {total_5v5/60.0:.2f} minutes")
else:
    print("No 5v5 found in cache.")

# Check skaters count at time 600 (10 mins in)
time = 600
active = df_shifts[(df_shifts['start_total_seconds'] <= time) & (df_shifts['end_total_seconds'] > time)]
print(f"Active players at {time}s:")
for tid in active['team_id'].unique():
    pids = active[active['team_id'] == tid]['player_id'].unique()
    print(f"  Team {tid}: {len(pids)} players")
