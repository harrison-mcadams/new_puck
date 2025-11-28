#!/usr/bin/env python3
"""Debug helper: reproduce _apply_intervals matching for a single game.

Usage: python scripts/debug_apply_intervals.py 2025020280
"""
import sys
from pprint import pprint
import nhl_api, parse, timing

def main(argv):
    gid = int(argv[1]) if len(argv) > 1 else 2025020280
    cond = {'game_state': ['5v5'], 'is_net_empty': [0], 'team': 'STL'}
    print('Game:', gid)
    feed = nhl_api.get_game_feed(gid)
    print('feed type:', type(feed).__name__, 'keys sample:', list(feed.keys())[:10] if isinstance(feed, dict) else None)
    ev_df = parse._game(feed)
    print('_game ->', None if ev_df is None else ev_df.shape)
    gdf = parse._elaborate(ev_df)
    print('_elaborate ->', None if gdf is None else gdf.shape)
    if gdf is None:
        print('No gdf produced; abort')
        return 1
    if 'game_id' not in gdf.columns:
        gdf['game_id'] = gid
    # ensure time col numeric
    if 'total_time_elapsed_seconds' in gdf.columns:
        times = gdf['total_time_elapsed_seconds'].dropna().astype(float)
        print('time min/max:', times.min(), times.max())
    else:
        print('No total_time_elapsed_seconds in gdf')
    # call timing
    print('Calling timing.demo_for_export...')
    res = timing.demo_for_export(gdf, cond, verbose=False)
    pprint({'res_keys': list(res.keys())})
    per = res.get('per_game', {})
    print('per_game length:', len(per))
    # find our gid
    entry = per.get(gid) or per.get(str(gid))
    if not entry:
        print('No per_game entry for gid; keys sample:', list(per.keys())[:10])
        return 2
    print('per_game entry keys:', list(entry.keys()))
    sides = entry.get('sides')
    print('sides type:', type(sides))
    team_side = None
    if isinstance(sides, dict):
        team_side = sides.get('team') or {}
        print('team_side keys:', list(team_side.keys()))
        team_intervals = team_side.get('intersection_intervals') or team_side.get('merged_intervals', {}).get('game_state') or []
    else:
        team_intervals = entry.get('intersection_intervals') or []
    print('team_intervals:', team_intervals)
    # Now try to match rows
    matches = []
    for idx, row in gdf.iterrows():
        t = row.get('total_time_elapsed_seconds')
        if t is None:
            continue
        for s, e in team_intervals:
            try:
                if float(s) <= float(t) <= float(e):
                    matches.append({'idx': idx, 't': float(t), 'event': row.get('event'), 'x': row.get('x'), 'y': row.get('y')})
                    break
            except Exception:
                continue
    print('matched rows count:', len(matches))
    print('sample matches:')
    pprint(matches[:20])
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))

