#!/usr/bin/env python3
"""Debug helper: compare parse._shifts outputs for get_shifts and get_shifts_from_nhl_html.

Usage: python scripts/debug_parse_shifts.py <game_id> [--save]

This script fetches both sources, runs parse._shifts on their outputs, and
prints concise summaries and sample rows to help debugging HTML parsing.
"""
import sys
import logging
import traceback
from pathlib import Path

logging.basicConfig(level=logging.INFO)

try:
    import pandas as pd
except Exception:
    pd = None

import nhl_api
import parse

OUT_DIR = Path('static')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def summarize_df(df, name):
    print(f"\n=== Summary for {name} ===")
    if df is None:
        print("<None>")
        return
    try:
        print('type:', type(df))
        if hasattr(df, 'shape'):
            print('shape:', df.shape)
        # print columns and dtypes
        try:
            print('columns:', list(df.columns))
            print(df.dtypes.to_dict())
        except Exception:
            pass
        # show first few rows
        try:
            if hasattr(df, 'head'):
                print('\nHead:')
                print(df.head(10).to_string())
        except Exception:
            pass
    except Exception:
        print('Failed to summarize df:', traceback.format_exc())


def run_debug(game_id: str, save: bool = True):
    print(f"Running debug for game_id={game_id}")
    try:
        print('\nFetching API shifts via get_shifts()...')
        api_res = nhl_api.get_shifts(game_id, force_refresh=True)
    except Exception as e:
        print('get_shifts raised:', e)
        api_res = {'game_id': game_id, 'raw': None, 'all_shifts': [], 'shifts_by_player': {}}

    try:
        print('\nFetching HTML shifts via get_shifts_from_nhl_html()...')
        html_res = nhl_api.get_shifts_from_nhl_html(game_id, force_refresh=True, debug=True)
    except Exception as e:
        print('get_shifts_from_nhl_html raised:', e)
        html_res = {'game_id': game_id, 'raw': None, 'all_shifts': [], 'shifts_by_player': {}, 'debug': {'error': str(e)}}

    print('\nRunning parse._shifts on API result...')
    try:
        df_api = parse._shifts(api_res)
    except Exception as e:
        print('parse._shifts(api_res) raised:', e)
        traceback.print_exc()
        df_api = None

    print('\nRunning parse._shifts on HTML result...')
    try:
        df_html = parse._shifts(html_res)
    except Exception as e:
        print('parse._shifts(html_res) raised:', e)
        traceback.print_exc()
        df_html = None

    summarize_df(df_api, 'API (_shifts output)')
    summarize_df(df_html, 'HTML (_shifts output)')

    # Also print compare_shifts summary from nhl_api for quick diff
    try:
        print('\nRunning nhl_api.compare_shifts() for a quick diff...')
        cmp = nhl_api.compare_shifts(game_id, debug=True)
        print('compare_shifts diff summary:')
        d = cmp.get('diff') or {}
        print('api_count:', d.get('api_count'), 'html_count:', d.get('html_count'))
        print('only_api_players:', d.get('only_api_players'))
        print('only_html_players:', d.get('only_html_players'))
        print('sample_diffs (truncated):')
        for s in d.get('sample_diffs', [])[:10]:
            print(' ', s)
    except Exception as e:
        print('compare_shifts raised:', e)
        traceback.print_exc()

    if save and pd is not None:
        try:
            if df_api is not None and not df_api.empty:
                p_api = OUT_DIR / f'debug_shifts_api_{game_id}.csv'
                df_api.to_csv(p_api, index=False)
                print(f'Wrote API _shifts CSV to: {p_api}')
            if df_html is not None and not df_html.empty:
                p_html = OUT_DIR / f'debug_shifts_html_{game_id}.csv'
                df_html.to_csv(p_html, index=False)
                print(f'Wrote HTML _shifts CSV to: {p_html}')
        except Exception:
            print('Failed to save CSVs:', traceback.format_exc())

    # return objects for programmatic use
    return {'game_id': game_id, 'api_res': api_res, 'html_res': html_res, 'df_api': df_api, 'df_html': df_html}


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python scripts/debug_parse_shifts.py <game_id> [--no-save]')
        sys.exit(1)
    gid = sys.argv[1]
    save_flag = True
    if len(sys.argv) > 2 and sys.argv[2] == '--no-save':
        save_flag = False
    run_debug(gid, save=save_flag)

