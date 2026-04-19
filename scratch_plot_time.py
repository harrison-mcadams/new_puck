import pandas as pd
from puck import analyze
import numpy as np

season = '20252026'
game_id = 2025020001
print(f"Running xgs_map for {game_id}...")
ret = analyze.xgs_map(
    game_id=game_id,
    condition={},
    out_path="temp.png",
    show=False,
    return_heatmaps=False,
    events_to_plot=['shot-on-goal', 'goal', 'xgs'],
    return_filtered_df=True,
    force_refresh=True
)

if isinstance(ret, tuple) and len(ret) >= 3:
    df = ret[2]
    print(f"DF returned with shape {df.shape}")
    print("Columns:", list(df.columns))
    time_cols = [c for c in df.columns if 'time' in c.lower() or 'period' in c.lower() or 'sec' in c.lower()]
    print("Time/Period columns:", time_cols)
    if 'total_time_elapsed_seconds' in df.columns and 'xgs' in df.columns:
        print(df[['event', 'team_id', 'total_time_elapsed_seconds', 'xgs']].dropna(subset=['xgs']).head(10))
    else:
        print("Missing time or xgs column!")
else:
    print("Failed to get df from analyze")
