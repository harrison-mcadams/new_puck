import pandas as pd
import numpy as np
from puck import timing
from puck import get_game_state

game_id = 2025020058
season = '20252026'

df_shifts = timing._get_shifts_df(game_id, season=season)
print(f"Game {game_id}: {len(df_shifts)} shifts found.")

# Sample every 10 seconds
times = np.arange(0, 3600, 10)
max_skaters = 0
for t in times:
    active = df_shifts[(df_shifts['start_total_seconds'] <= t) & (df_shifts['end_total_seconds'] > t)]
    for tid in active['team_id'].unique():
        count = len(active[active['team_id'] == tid]['player_id'].unique())
        max_skaters = max(max_skaters, count)

print(f"Max Skaters seen on any team: {max_skaters}")
