"""Debug script: examine penalty -> faceoff assignment behavior.

Usage: python scripts/debug_penalty_assign.py [GAME_ID]

Prints:
 - faceoff event times
 - penalty-like rows and assigned game_state_relative_to_team
 - computed intervals for 5v5 and 5v4 (with verbose debug from intervals_for_condition)
"""

from pathlib import Path
import sys
# Ensure project root is on sys.path so top-level modules can be imported when
# running this script directly (e.g., .venv/bin/python scripts/demo_plot_flyers.py)
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import sys
import nhl_api
import parse
import timing
import pandas as pd

def main(gid='2025020280'):
    print('Fetching feed for', gid)
    feed = None
    try:
        try:
            feed = nhl_api.get_game_feed(int(gid))
        except Exception:
            feed = nhl_api.get_game_feed(gid)
    except Exception as e:
        print('nhl_api.get_game_feed raised exception:', repr(e))

    print('feed repr:', repr(feed)[:500])
    if not feed:
        print('No feed returned from nhl_api.get_game_feed; attempting direct HTTP probe...')
        try:
            import requests
            url = f'https://api-web.nhle.com/v1/gamecenter/{gid}/play-by-play'
            r = requests.get(url, timeout=10)
            print('HTTP probe status:', r.status_code)
            headers = {k:v for k,v in r.headers.items() if k.lower() in ('content-type','retry-after')}
            print('HTTP probe headers:', headers)
            txt = r.text or ''
            print('HTTP probe body head:', txt[:1000])
        except Exception as e:
            print('HTTP probe failed:', repr(e))
        return

    ev_df = parse._game(feed)
    if ev_df is None or ev_df.empty:
        print('parse._game returned no events; inspect feed structure (repr above)')
        return
    gdf = parse._elaborate(ev_df)
    print('Parsed events rows:', len(gdf))

    # infer home team when available
    home_abb = None
    if 'home_abb' in gdf.columns:
        try:
            home_abb = gdf['home_abb'].dropna().unique().tolist()[0]
        except Exception:
            home_abb = None
    print('inferred home_abb =', home_abb)

    print('\nRunning add_game_state_relative_column(debug=True)')
    df_rel = timing.add_game_state_relative_column(gdf.copy(), home_abb, debug=True)

    # Build event text for detecting faceoffs/penalties
    ev_text_cols = [c for c in df_rel.columns if any(k in c.lower() for k in ('event','type','desc','description'))]
    if ev_text_cols:
        ev_text = df_rel[ev_text_cols].astype(str).fillna('').agg(' '.join, axis=1).str.lower()
    else:
        ev_text = df_rel.get('event', pd.Series(['']*len(df_rel))).astype(str).fillna('').str.lower()

    face_mask = ev_text.str.contains(r'\bface[ -]?off\b', regex=True)
    pen_mask = ev_text.str.contains('penal') | ev_text.str.contains('penalty')

    print('\nFaceoff events (index, time):')
    series_fo = df_rel.loc[face_mask, 'total_time_elapsed_seconds']
    if series_fo is not None and not series_fo.empty:
        for idx, t in series_fo.items():
            print(f"  idx={idx}, time={t}")
    else:
        print('  (no faceoff events detected)')

    print('\nPenalty-like events and assigned states:')
    if pen_mask.any():
        cols = [c for c in ('total_time_elapsed_seconds','event','game_state','game_state_relative_to_team') if c in df_rel.columns]
        for idx, row in df_rel.loc[pen_mask, cols].iterrows():
            print(f"  idx={idx}, time={row.get('total_time_elapsed_seconds')}, event={row.get('event')}, game_state={row.get('game_state')}, assigned={row.get('game_state_relative_to_team')}")
    else:
        print('  (no penalty-like rows detected)')

    print('\nIntervals for game_state_relative_to_team == 5v5:')
    intervals_5v5, sec5v5, tot = timing.intervals_for_condition(df_rel, {'game_state_relative_to_team':'5v5'}, verbose=True)
    print('Result intervals:', intervals_5v5)
    print('Total seconds 5v5:', sec5v5)

    print('\nIntervals for game_state_relative_to_team == 5v4:')
    intervals_5v4, sec5v4, tot = timing.intervals_for_condition(df_rel, {'game_state_relative_to_team':'5v4'}, verbose=True)
    print('Result intervals:', intervals_5v4)
    print('Total seconds 5v4:', sec5v4)

if __name__ == '__main__':
    gid = sys.argv[1] if len(sys.argv) > 1 else '2025020111'
    main(gid)
