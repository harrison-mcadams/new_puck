#!/usr/bin/env python3
"""Run xG analysis for the most recent Flyers (PHI) game for ALL situations.
Saves image to analysis/<gameid>_PHI_all.png
"""
from pathlib import Path
import sys
# Ensure project root is on sys.path so top-level modules can be imported
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
from datetime import datetime, timezone
from puck import timing, plot
import os

analysis_dir = project_root / 'analysis'
analysis_dir.mkdir(parents=True, exist_ok=True)

season = '20252026'
sdf = timing.load_season_df(season)

# Logic to find season DF if timing.load_season_df fails (copied from demo_plot_flyers.py for robustness)
if sdf is None or sdf.empty:
    print('timing.load_season_df did not find a CSV for season', season)
    candidates = [
        project_root / 'data' / season / f"{season}_df.csv",
        project_root / 'data' / season / f"{season}.csv",
        project_root / 'data' / f"{season}_df.csv",
        project_root / 'data' / f"{season}.csv",
        project_root / 'analysis' / f"{season}_df.csv", # Check analysis too
    ]
    found = None
    for p in candidates:
        if p.exists():
            found = p
            break
    if found:
        print('Loading season CSV from', found)
        sdf = pd.read_csv(found)
    else:
        print('No CSV found. Aborting.')
        sys.exit(1)

# Get candidate game ids for PHI
gids = timing.select_team_game(sdf, 'PHI')
if not gids:
    print('No games found for PHI in season dataframe; aborting.')
    sys.exit(1)

# Robustly choose most recent game not in the future
date_cols = [c for c in sdf.columns if any(k in c.lower() for k in ('date','time','start'))]
chosen_gid = None
now = datetime.now(timezone.utc)
if date_cols:
    for dc in date_cols:
        try:
            sdf[dc + '_dt'] = pd.to_datetime(sdf[dc], errors='coerce')
            # filter to PHI games
            mask_g = sdf['game_id'].isin(gids) if isinstance(gids, list) else (sdf['game_id'] == gids)
            cand = sdf.loc[mask_g].copy()
            if cand.empty: continue
            cand = cand[cand[dc + '_dt'].notna()]
            if cand.empty: continue
            
            cand_past = cand[cand[dc + '_dt'] <= now]
            cand_use = cand_past if not cand_past.empty else cand
            
            cand_use['delta'] = (now - cand_use[dc + '_dt']).abs()
            chosen = cand_use.sort_values('delta').iloc[0]
            chosen_gid = str(chosen['game_id'])
            break
        except Exception:
            continue

if chosen_gid is None:
    # fallback
    chosen_gid = str(gids[-1]) if isinstance(gids, (list, tuple)) else str(gids)

print('Selected game id for PHI:', chosen_gid)

# Run analysis for ALL situations (no game_state filter)
conditions = {'team': 'PHI'} # Filters for events where PHI is involved (home/away), but includes all game states
plot_kwargs = {
    'out_path': str(analysis_dir / f'{chosen_gid}_PHI_all.png'),
    'events_to_plot': ['shot-on-goal', 'goal', 'xgs'],
    'heatmap_split_mode': 'team_not_team',
    'team_for_heatmap': 'PHI',
    'return_heatmaps': False,
    'return_timing': True,
    'title': f'{chosen_gid} PHI - All Situations'
}

print(f"Running analysis for game {chosen_gid} (All Situations)...")
res = plot._game(chosen_gid, conditions=conditions, plot_kwargs=plot_kwargs)
print('Plot completed, saved to', plot_kwargs['out_path'])
