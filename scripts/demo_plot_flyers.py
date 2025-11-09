#!/usr/bin/env python3
"""Demo: plot most recent Flyers (PHI) game for 5v5 only.
Saves image to static/<gameid>_PHI_5v5.png
"""
from pathlib import Path
import sys
# Ensure project root is on sys.path so top-level modules can be imported when
# running this script directly (e.g., .venv/bin/python scripts/demo_plot_flyers.py)
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
from datetime import datetime, timezone
import timing, plot

Path('static').mkdir(parents=True, exist_ok=True)

season = '20252026'
sdf = timing.load_season_df(season)
if sdf is None or sdf.empty:
    print('No season dataframe found; aborting.')
    raise SystemExit(1)

# get candidate game ids for PHI
gids = timing.select_team_game(sdf, 'PHI')
if not gids:
    print('No games found for PHI in season dataframe; aborting.')
    raise SystemExit(1)

# Robustly choose most recent game not in the future, prefer date columns
date_cols = [c for c in sdf.columns if any(k in c.lower() for k in ('date','time','start'))]
chosen_gid = None
now = datetime.now(timezone.utc)
if date_cols:
    # try to parse the first useful date column
    for dc in date_cols:
        try:
            sdf[dc + '_dt'] = pd.to_datetime(sdf[dc], errors='coerce')
            # filter to PHI games
            mask_g = False
            try:
                mask_g = sdf['game_id'].isin(gids)
            except Exception:
                # if gids is scalar
                try:
                    mask_g = sdf['game_id'] == gids
                except Exception:
                    mask_g = pd.Series([False]*len(sdf), index=sdf.index)
            cand = sdf.loc[mask_g].copy()
            if cand.empty:
                continue
            # drop future games
            cand = cand[cand[dc + '_dt'].notna()]
            if cand.empty:
                continue
            cand_past = cand[cand[dc + '_dt'] <= now]
            if cand_past.empty:
                # if all are future, pick the earliest (or latest depending)
                cand_use = cand
            else:
                cand_use = cand_past
            # choose closest to now
            cand_use['delta'] = (now - cand_use[dc + '_dt']).abs()
            chosen = cand_use.sort_values('delta').iloc[0]
            chosen_gid = str(chosen['game_id'])
            break
        except Exception:
            continue

if chosen_gid is None:
    # fallback: numeric max or first element
    try:
        if isinstance(gids, (list,tuple)):
            nums = [int(g) for g in gids]
            chosen_gid = str(max(nums))
        else:
            chosen_gid = str(gids)
    except Exception:
        try:
            chosen_gid = str(gids[0])
        except Exception:
            chosen_gid = str(gids)

print('Selected game id for PHI:', chosen_gid)

conditions = {'team': 'PHI', 'game_state': '5v5'}
plot_kwargs = {
    'out_path': f'static/{chosen_gid}_PHI_5v5.png',
    'events_to_plot': ['shot-on-goal', 'goal', 'xgs'],
    'heatmap_split_mode': 'team_not_team',
    'team_for_heatmap': 'PHI',
    'return_heatmaps': False,
}

res = plot._game(chosen_gid, conditions=conditions, plot_kwargs=plot_kwargs)
print('Plot completed, saved to', plot_kwargs['out_path'])

# if caller wants the figure returned, res can be used; we just exit
