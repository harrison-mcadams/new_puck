#!/usr/bin/env python3
"""Debug helper: compare parse._shifts outputs for get_shifts and get_shifts_from_nhl_html.

Usage: python scripts/debug_parse_shifts.py <game_id> [--save]

This script fetches both sources, runs parse._shifts on their outputs, and
prints concise summaries and sample rows to help debugging HTML parsing.
It validates that team_id and player_id are correctly resolved.
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


def validate_mappings(html_res, df_html):
    """Validate team_id and player_id mappings in parsed shifts.
    
    Returns dict with validation results and any errors found.
    """
    results = {
        'total_shifts': 0,
        'missing_team_id': 0,
        'player_id_is_jersey': 0,
        'valid_shifts': 0,
        'errors': [],
        'samples': {'missing_team_id': [], 'player_id_is_jersey': []}
    }
    
    if not html_res or not isinstance(html_res, dict):
        results['errors'].append('html_res is None or not a dict')
        return results
    
    all_shifts = html_res.get('all_shifts', [])
    results['total_shifts'] = len(all_shifts)
    
    if results['total_shifts'] == 0:
        results['errors'].append('No shifts found in html_res')
        return results
    
    # Check each shift
    for i, shift in enumerate(all_shifts):
        has_issues = False
        
        # Check team_id
        team_id = shift.get('team_id')
        if team_id is None:
            results['missing_team_id'] += 1
            has_issues = True
            if len(results['samples']['missing_team_id']) < 3:
                results['samples']['missing_team_id'].append({
                    'index': i,
                    'team_side': shift.get('team_side'),
                    'player_number': shift.get('player_number'),
                    'player_name': shift.get('player_name'),
                    'team_id': team_id
                })
        
        # Check player_id - if it's a simple integer that matches player_number, 
        # it's likely still a jersey number
        player_id = shift.get('player_id')
        player_number = shift.get('player_number')
        if player_id is not None and player_number is not None:
            # Check if player_id looks like a jersey number (1-99 typically)
            # vs a real NHL player ID (typically 8-digit numbers like 8471675)
            try:
                pid_int = int(player_id)
                pnum_int = int(player_number)
                # If they match and it's under 100, it's probably still a jersey
                if pid_int == pnum_int and pid_int < 100:
                    results['player_id_is_jersey'] += 1
                    has_issues = True
                    if len(results['samples']['player_id_is_jersey']) < 3:
                        results['samples']['player_id_is_jersey'].append({
                            'index': i,
                            'team_side': shift.get('team_side'),
                            'player_number': player_number,
                            'player_id': player_id,
                            'player_name': shift.get('player_name')
                        })
            except (ValueError, TypeError):
                pass
        
        if not has_issues:
            results['valid_shifts'] += 1
    
    return results


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

    # Validate team_id and player_id mappings
    print('\n' + '='*70)
    print('VALIDATION: team_id and player_id mappings')
    print('='*70)
    
    validation = validate_mappings(html_res, df_html)
    
    print(f"\nTotal shifts parsed: {validation['total_shifts']}")
    print(f"Valid shifts (team_id set, player_id canonical): {validation['valid_shifts']}")
    print(f"Shifts missing team_id: {validation['missing_team_id']}")
    print(f"Shifts where player_id is still jersey number: {validation['player_id_is_jersey']}")
    
    if validation['errors']:
        print(f"\nErrors encountered:")
        for err in validation['errors']:
            print(f"  - {err}")
    
    if validation['samples']['missing_team_id']:
        print(f"\nSample shifts missing team_id:")
        for sample in validation['samples']['missing_team_id']:
            print(f"  {sample}")
    
    if validation['samples']['player_id_is_jersey']:
        print(f"\nSample shifts where player_id is jersey number:")
        for sample in validation['samples']['player_id_is_jersey']:
            print(f"  {sample}")
    
    # Print debug info from html_res
    if html_res and html_res.get('debug'):
        print(f"\nDebug info from get_shifts_from_nhl_html:")
        dbg = html_res['debug']
        print(f"  Tables scanned: {dbg.get('tables_scanned', 0)}")
        print(f"  Players scanned: {dbg.get('players_scanned', 0)}")
        print(f"  Shifts found: {dbg.get('found_shifts', 0)}")
        print(f"  Team IDs: {dbg.get('team_ids', {})}")
        print(f"  Team ID set for: {dbg.get('team_id_set_count', 0)}/{dbg.get('found_shifts', 0)} shifts")
        if 'roster_mapping' in dbg:
            rm = dbg['roster_mapping']
            print(f"  Roster players: home={rm.get('home_players', 0)}, away={rm.get('away_players', 0)}")
            print(f"  Player ID mapped: {rm.get('mapped_shifts', 0)}/{dbg.get('found_shifts', 0)} shifts")
            print(f"  Unmapped players: {rm.get('unmapped_shifts', 0)}")
    
    # Assertions to ensure correctness
    print('\n' + '='*70)
    print('ASSERTIONS')
    print('='*70)
    
    if validation['total_shifts'] > 0:
        try:
            assert validation['missing_team_id'] == 0, \
                f"FAIL: {validation['missing_team_id']} shifts missing team_id (should be 0)"
            print("✓ PASS: All shifts have team_id set")
        except AssertionError as e:
            print(f"✗ {e}")
        
        try:
            assert validation['player_id_is_jersey'] == 0, \
                f"FAIL: {validation['player_id_is_jersey']} shifts have player_id as jersey number (should be 0)"
            print("✓ PASS: All shifts have canonical player_id")
        except AssertionError as e:
            print(f"✗ {e}")
    else:
        print("⚠ WARNING: No shifts to validate (total_shifts = 0)")

    # Also print compare_shifts summary from nhl_api for quick diff
    try:
        print('\n' + '='*70)
        print('Running nhl_api.compare_shifts() for a quick diff...')
        print('='*70)
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
                print(f'\nWrote API _shifts CSV to: {p_api}')
            if df_html is not None and not df_html.empty:
                p_html = OUT_DIR / f'debug_shifts_html_{game_id}.csv'
                df_html.to_csv(p_html, index=False)
                print(f'Wrote HTML _shifts CSV to: {p_html}')
        except Exception:
            print('Failed to save CSVs:', traceback.format_exc())

    # return objects for programmatic use
    return {'game_id': game_id, 'api_res': api_res, 'html_res': html_res, 'df_api': df_api, 'df_html': df_html, 'validation': validation}


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python scripts/debug_parse_shifts.py <game_id> [--no-save]')
        sys.exit(1)
    gid = sys.argv[1]
    save_flag = True
    if len(sys.argv) > 2 and sys.argv[2] == '--no-save':
        save_flag = False
    run_debug(gid, save=save_flag)

