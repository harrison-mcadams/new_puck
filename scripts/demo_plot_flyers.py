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
    print('timing.load_season_df did not find a CSV for season', season)
    # Try additional candidate locations relative to project root
    candidates = [
        project_root / 'data' / season / f"{season}_df.csv",
        project_root / 'data' / season / f"{season}.csv",
        project_root / 'data' / f"{season}_df.csv",
        project_root / 'data' / f"{season}.csv",
        project_root / 'static' / f"{season}_df.csv",
        project_root / 'static' / f"{season}.csv",
    ]
    found = None
    for p in candidates:
        try:
            print('Checking', p)
        except Exception:
            pass
        if p.exists():
            found = p
            break
    if found is not None:
        try:
            print('Loading season CSV from', found)
            sdf = pd.read_csv(found)
        except Exception as e:
            print('Failed to read', found, '->', e)
    else:
        # fallback: recursive search under project_root/data and project_root/static
        search_roots = [project_root / 'data', project_root / 'static', project_root]
        matches = []
        for root in search_roots:
            try:
                if root.exists():
                    for p in root.rglob(f'*{season}*.csv'):
                        matches.append(p)
            except Exception:
                pass
        if matches:
            p = matches[0]
            try:
                print('Found candidate CSV via recursive search:', p)
                sdf = pd.read_csv(p)
            except Exception as e:
                print('Failed to read', p, '->', e)
        else:
            print('No CSV found for season', season, 'in candidates or recursive search under data/static/project root.')
            print('Searched candidates:', [str(x) for x in candidates])
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
    'return_timing': True,
}

# Request timing info along with the plot
res = plot._game(chosen_gid, conditions=conditions, plot_kwargs=plot_kwargs)
print('Plot completed, saved to', plot_kwargs['out_path'])

# If timing info was returned, save it to JSON for inspection
try:
    timing_info = None
    if isinstance(res, tuple):
        # timing_info is appended as the last element by plot._game when return_timing=True
        timing_info = res[-1]
    else:
        timing_info = None
    if timing_info:
        import json
        out_json = Path(plot_kwargs['out_path']).with_name(f"{chosen_gid}_timing.json")
        with open(out_json, 'w') as fh:
            json.dump(timing_info, fh, indent=2)
        print('Saved timing info to', out_json)
    else:
        print('No timing info returned')
except Exception as e:
    print('Failed to save timing info:', e)

# if caller wants the figure returned, res can be used; we just exit
