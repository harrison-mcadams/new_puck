#!/usr/bin/env python3
import sys, traceback
from pprint import pprint

def main(gid):
    try:
        import nhl_api
        print('import nhl_api OK', flush=True)
    except Exception as e:
        print('import nhl_api FAILED', e, flush=True)
        traceback.print_exc()
        return
    try:
        import parse
        print('import parse OK', flush=True)
    except Exception as e:
        print('import parse FAILED', e, flush=True)
        traceback.print_exc()
        return
    try:
        import timing
        print('import timing OK', flush=True)
    except Exception as e:
        print('import timing FAILED', e, flush=True)
        traceback.print_exc()
        return

    try:
        feed = nhl_api.get_game_feed(gid)
        print('fetched feed type:', type(feed).__name__, 'keys_sample=', list(feed.keys())[:10] if isinstance(feed, dict) else None, flush=True)
    except Exception as e:
        print('get_game_feed FAILED', e, flush=True)
        traceback.print_exc()
        return

    try:
        ev_df = parse._game(feed)
        print('_game rows:', None if ev_df is None else getattr(ev_df,'shape',None), flush=True)
    except Exception as e:
        print('parse._game FAILED', e, flush=True)
        traceback.print_exc()
        return

    try:
        gdf = parse._elaborate(ev_df)
        print('_elaborate rows:', None if gdf is None else getattr(gdf,'shape',None), flush=True)
    except Exception as e:
        print('parse._elaborate FAILED', e, flush=True)
        traceback.print_exc()
        return

    if gdf is None:
        print('NO GDF, abort', flush=True)
        return
    if 'game_id' not in gdf.columns:
        gdf['game_id'] = gid

    try:
        print('time col exists?', 'total_time_elapsed_seconds' in gdf.columns, flush=True)
        if 'total_time_elapsed_seconds' in gdf.columns:
            times = gdf['total_time_elapsed_seconds'].dropna().astype(float)
            print('time min/max:', times.min(), times.max(), flush=True)
    except Exception:
        print('time column processing failed', flush=True)
        traceback.print_exc()

    cond = {'game_state':['5v5'], 'is_net_empty':[0], 'team':'STL'}
    try:
        print('Calling timing.demo_for_export...', flush=True)
        res = timing.demo_for_export(gdf, cond, verbose=True)
        print('demo_for_export returned keys:', list(res.keys()), flush=True)
    except Exception as e:
        print('timing.demo_for_export FAILED', e, flush=True)
        traceback.print_exc()
        return

    try:
        per = res.get('per_game', {})
        print('per_game count:', len(per), flush=True)
        keys = list(per.keys())[:20]
        print('per_game keys sample:', [(k, type(k)) for k in keys], flush=True)
        entry = per.get(gid) or per.get(str(gid))
        print('found entry?', entry is not None, flush=True)
    except Exception:
        print('Inspecting per_game failed', flush=True)
        traceback.print_exc()
        return

    if not entry:
        print('No entry for gid, sample keys:', keys, flush=True)
        return

    try:
        print('Entry keys:', list(entry.keys()), flush=True)
        sides = entry.get('sides')
        print('sides type:', type(sides), flush=True)
        team_side = sides.get('team') if isinstance(sides, dict) else None
        team_intervals = team_side.get('intersection_intervals') if isinstance(team_side, dict) else entry.get('intersection_intervals')
        print('team_intervals:', team_intervals, flush=True)
    except Exception:
        print('Inspect entry failed', flush=True)
        traceback.print_exc()
        return

    try:
        matches = 0
        sample = []
        for idx,row in gdf.iterrows():
            t = row.get('total_time_elapsed_seconds')
            if t is None:
                continue
            for s,e in (team_intervals or []):
                try:
                    if float(s) <= float(t) <= float(e):
                        matches += 1
                        if len(sample) < 10:
                            sample.append((int(idx), float(t), row.get('event')))
                        break
                except Exception:
                    continue
        print('matches count:', matches, 'sample:', sample, flush=True)
    except Exception:
        print('matching failed', flush=True)
        traceback.print_exc()

if __name__ == '__main__':
    gid = int(sys.argv[1]) if len(sys.argv) > 1 else 2025020280
    main(gid)

