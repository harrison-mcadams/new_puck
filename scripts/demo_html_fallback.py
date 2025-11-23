#!/usr/bin/env python3
"""Demo script showing HTML fallback integration in timing_new.py

This script demonstrates:
1. Using get_shifts_with_html_fallback() as a drop-in replacement for nhl_api.get_shifts()
2. How _get_shifts_df() in timing_new.py automatically uses HTML fallback
3. Comparing results between API and HTML sources

Usage:
    python scripts/demo_html_fallback.py --game 2025020339
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import nhl_api
import timing_new
import parse

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def demo_wrapper_function(game_id: int):
    """Demo the get_shifts_with_html_fallback() wrapper function."""
    print('\n' + '='*70)
    print('Demo 1: get_shifts_with_html_fallback() wrapper function')
    print('='*70)
    
    print(f"\nFetching shifts for game {game_id} with HTML fallback...")
    shifts_res = timing_new.get_shifts_with_html_fallback(game_id, min_rows_threshold=5)
    
    all_shifts = shifts_res.get('all_shifts', [])
    print(f"Result: {len(all_shifts)} shifts returned")
    
    if all_shifts:
        # Parse to DataFrame
        df = parse._shifts(shifts_res)
        print(f"Parsed to DataFrame: {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        
        # Show sample
        if not df.empty:
            print("\nSample rows:")
            print(df.head(3).to_string())
    
    return shifts_res


def demo_get_shifts_df(game_id: int):
    """Demo the _get_shifts_df() function with integrated HTML fallback."""
    print('\n' + '='*70)
    print('Demo 2: timing_new._get_shifts_df() with integrated HTML fallback')
    print('='*70)
    
    print(f"\nCalling _get_shifts_df({game_id})...")
    df = timing_new._get_shifts_df(game_id, min_rows_threshold=5)
    
    print(f"Result: {len(df)} rows")
    if not df.empty:
        print(f"Columns: {list(df.columns)}")
        print("\nSample rows:")
        print(df.head(3).to_string())
    
    return df


def demo_direct_comparison(game_id: int):
    """Demo direct comparison of API vs HTML sources."""
    print('\n' + '='*70)
    print('Demo 3: Direct comparison of API vs HTML shift sources')
    print('='*70)
    
    print(f"\nFetching API shifts...")
    try:
        api_res = nhl_api.get_shifts(game_id)
        api_count = len(api_res.get('all_shifts', []))
    except Exception as e:
        print(f"API failed: {e}")
        api_count = 0
    
    print(f"Fetching HTML shifts...")
    try:
        html_res = nhl_api.get_shifts_from_nhl_html(game_id, force_refresh=True, debug=True)
        html_count = len(html_res.get('all_shifts', []))
        
        # Print debug info
        if html_res.get('debug'):
            dbg = html_res['debug']
            print(f"\nHTML parsing debug info:")
            print(f"  Tables scanned: {dbg.get('tables_scanned', 0)}")
            print(f"  Players scanned: {dbg.get('players_scanned', 0)}")
            print(f"  Shifts found: {dbg.get('found_shifts', 0)}")
            print(f"  Team IDs: {dbg.get('team_ids', {})}")
            if 'roster_mapping' in dbg:
                rm = dbg['roster_mapping']
                print(f"  Roster mapping: home={rm.get('home_players', 0)} players, away={rm.get('away_players', 0)} players")
                print(f"  Mapped {rm.get('mapped_shifts', 0)}/{dbg.get('found_shifts', 0)} shifts to canonical player_id")
                if rm.get('unmapped_shifts', 0) > 0:
                    print(f"  ⚠ {rm['unmapped_shifts']} shifts unmapped (player not in roster)")
    except Exception as e:
        print(f"HTML failed: {e}")
        html_count = 0
    
    print(f"\n{'Source':<15} {'Shift Count':<15}")
    print('-' * 30)
    print(f"{'API':<15} {api_count:<15}")
    print(f"{'HTML':<15} {html_count:<15}")
    
    if api_count == 0 and html_count > 0:
        print("\n✓ HTML fallback would be used (API returned 0 shifts)")
    elif api_count > 0 and html_count > 0:
        print(f"\n✓ Both sources returned data (difference: {abs(api_count - html_count)} shifts)")
    elif api_count > 0 and html_count == 0:
        print("\n⚠ API returned data but HTML failed")
    else:
        print("\n⚠ Both sources returned no data")


def main():
    parser = argparse.ArgumentParser(description='Demo HTML fallback integration')
    parser.add_argument('--game', '-g', type=int, required=True, help='Game ID')
    parser.add_argument('--all', action='store_true', help='Run all demos')
    parser.add_argument('--wrapper', action='store_true', help='Run wrapper demo')
    parser.add_argument('--df', action='store_true', help='Run _get_shifts_df demo')
    parser.add_argument('--compare', action='store_true', help='Run comparison demo')
    args = parser.parse_args()
    
    # Default to all if no specific demo selected
    run_all = args.all or not (args.wrapper or args.df or args.compare)
    
    try:
        if run_all or args.wrapper:
            demo_wrapper_function(args.game)
        
        if run_all or args.df:
            demo_get_shifts_df(args.game)
        
        if run_all or args.compare:
            demo_direct_comparison(args.game)
        
        print('\n' + '='*70)
        print('Demo complete!')
        print('='*70)
        return 0
        
    except Exception as e:
        logging.exception('Demo failed: %s', e)
        return 1


if __name__ == '__main__':
    sys.exit(main())
