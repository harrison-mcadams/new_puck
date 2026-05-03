import os
import sys
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import get_game_state
from puck import timing

gid = 2025020352
df_shifts = timing._get_shifts_df(gid, season='20252026')
df_gs, _ = get_game_state.get_game_state(gid, df_shifts=df_shifts, return_df=True)

print(f"GS Timeline for {gid}:")
print(df_gs[['start', 'end', 'label']].head(20))
print("\nUnique labels:", df_gs['label'].unique())
