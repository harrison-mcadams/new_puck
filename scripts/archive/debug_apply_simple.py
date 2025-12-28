#!/usr/bin/env python3
import sys
import json
import time
from pprint import pprint
import nhl_api, parse, timing

def dbg(gid):
    print('DEBUG START', gid, flush=True)
    feed = nhl_api.get_game_feed(gid)
    print('feed type:', type(feed).__name__, 'keys:', list(feed.keys())[:10] if isinstance(feed, dict) else None, flush=True)
    ev_df = parse._game(feed)
    print('_game rows:', None if ev_df is None else getattr(ev_df,'shape',None), flush=True)
    gdf = parse._elaborate(ev_df)
    print('_elaborate rows:', None if gdf is None else getattr(gdf,'shape',None), flush=True)
    if gdf is None:
        print('NO GDF', flush=True)
        return
    if 'game_id' not in gdf.columns:
        gdf['game_id'] = gid
    print('time col exists:', 'total_time_elapsed_seconds' in gdf.columns, flush=True)
    if 'total_time_elapsed_seconds' in gdf.columns:
        times = gdf['total_time_elapsed_seconds'].dropna().astype(float)
        print('time min/max:', times.min(), times.max(), flush=True)
    cond = {'game_state':['5v5'], 'is_net_empty':[0], 'team':'STL'}
    print('Calling timing.demo_for_export...', flush=True)
    res = timing.demo_for_export(gdf, cond, verbose=False)
    print('Timing keys:', list(res.keys()), flush=True)
    per = res.get('per_game',{})
    print('per_game count:', len(per), flush=True)
    # print keys types
    sample_keys = list(per.keys())[:20]
    print('sample per_game keys types:', [(k, type(k)) for k in sample_keys], flush=True)
    entry = per.get(gid) or per.get(str(gid))
    print('found entry?', entry is not None, flush=True)
    if not entry:
        print('per_game keys sample:', sample_keys, flush=True)
        return
    pprint(entry, stream=sys.stdout)
    sides = entry.get('sides')
    print('sides type:', type(sides), flush=True)
    team_side = sides.get('team') if isinstance(sides, dict) else None
    print('team_side type:', type(team_side), flush=True)
    team_intervals = team_side.get('intersection_intervals') if isinstance(team_side, dict) else entry.get('intersection_intervals')
    print('team_intervals:', team_intervals, flush=True)
    # Now check matches
    matches = []
    for idx, row in gdf.iterrows():
        t = row.get('total_time_elapsed_seconds')
        if t is None:
            continue
        for s,e in (team_intervals or []):
            try:
                if float(s) <= float(t) <= float(e):
                    matches.append((idx,float(t), row.get('event')))
                    break
            except Exception:
                continue
    print('matches count:', len(matches), flush=True)
    print('sample matches:', matches[:20], flush=True)
    print('DEBUG END', gid, flush=True)

if __name__ == '__main__':
    gid = int(sys.argv[1]) if len(sys.argv)>1 else 2025020280
    dbg(gid)

