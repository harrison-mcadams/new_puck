#!/usr/bin/env python3
"""Compare game_state and is_net_empty parsed from API vs HTML shift sources.

Usage:
    python scripts/compare_game_state.py --game 2025020339
    
This script fetches shifts from both the API (get_shifts) and HTML (get_shifts_from_nhl_html)
sources, derives game_state and is_net_empty values from each, and reports differences.

The comparison helps validate that the HTML parsing produces equivalent results to the
official API data.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import nhl_api
import parse

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def derive_game_state_from_shifts(df_shifts: pd.DataFrame, 
                                   home_id: Optional[int] = None, 
                                   away_id: Optional[int] = None) -> pd.DataFrame:
    """Derive game_state intervals from shift data.
    
    Returns a DataFrame with columns: start, end, label (game_state like '5v5'), source
    """
    if df_shifts is None or df_shifts.empty:
        return pd.DataFrame(columns=['start', 'end', 'label', 'source'])
    
    # Classify players as goalies or skaters
    try:
        from timing_new import _classify_player_roles
        roles = _classify_player_roles(df_shifts).get('roles', {})
    except Exception:
        # fallback: assume all players are skaters
        roles = {}
    
    # Build event timeline
    events = []
    for _, r in df_shifts.iterrows():
        s = r.get('start_total_seconds')
        e = r.get('end_total_seconds')
        tid = r.get('team_id')
        pid = r.get('player_id')
        if s is None or e is None or tid is None or pid is None:
            continue
        events.append((float(s), 'start', str(tid), str(pid)))
        events.append((float(e), 'end', str(tid), str(pid)))
    
    if not events:
        return pd.DataFrame(columns=['start', 'end', 'label', 'source'])
    
    events.sort(key=lambda x: (x[0], 0 if x[1] == 'end' else 1))
    
    # Track active players per team
    active = defaultdict(set)
    intervals = []
    prev_t = events[0][0]
    
    i = 0
    while i < len(events):
        t0 = events[i][0]
        if t0 > prev_t:
            # Determine game state for interval [prev_t, t0)
            def count_skaters(tid_val):
                if tid_val is None:
                    return 0
                pids = active.get(str(tid_val), set())
                return sum(1 for pid in pids if roles.get(str(pid), 'S') != 'G')
            
            if home_id is not None:
                home_count = count_skaters(home_id)
            else:
                home_count = 0
            if away_id is not None:
                away_count = count_skaters(away_id)
            else:
                away_count = 0
            
            label = f"{home_count}v{away_count}"
            intervals.append({'start': prev_t, 'end': t0, 'label': label, 'source': 'shifts'})
            prev_t = t0
        
        # Apply all events at this timestamp
        while i < len(events) and events[i][0] == t0:
            _, typ, tid, pid = events[i]
            if typ == 'start':
                active[str(tid)].add(str(pid))
            else:
                active[str(tid)].discard(str(pid))
            i += 1
    
    return pd.DataFrame(intervals)


def derive_is_net_empty_from_shifts(df_shifts: pd.DataFrame, 
                                     home_id: Optional[int] = None, 
                                     away_id: Optional[int] = None) -> pd.DataFrame:
    """Derive is_net_empty intervals from shift data.
    
    Returns a DataFrame with columns: start, end, is_net_empty, source
    is_net_empty values: 0 = both nets have goalies, 1 = at least one net is empty
    """
    if df_shifts is None or df_shifts.empty:
        return pd.DataFrame(columns=['start', 'end', 'is_net_empty', 'source'])
    
    try:
        from timing_new import _classify_player_roles, _intervals_from_goalie_presence
        team_id_map = {'home_id': home_id, 'away_id': away_id}
        goalie_presence = _intervals_from_goalie_presence(df_shifts, team_id_map)
        home_pres = goalie_presence.get('home', [])
        away_pres = goalie_presence.get('away', [])
    except Exception as e:
        logging.warning('derive_is_net_empty_from_shifts: failed to get goalie presence: %s', e)
        return pd.DataFrame(columns=['start', 'end', 'is_net_empty', 'source'])
    
    # Build event timeline from goalie presence intervals
    events = []
    for s, e in home_pres:
        events.append((float(s), 'start', 'home'))
        events.append((float(e), 'end', 'home'))
    for s, e in away_pres:
        events.append((float(s), 'start', 'away'))
        events.append((float(e), 'end', 'away'))
    
    if not events:
        return pd.DataFrame(columns=['start', 'end', 'is_net_empty', 'source'])
    
    events.sort(key=lambda x: (x[0], 0 if x[1] == 'end' else 1))
    
    active_goalies = {'home': 0, 'away': 0}
    intervals = []
    prev_t = events[0][0]
    
    i = 0
    while i < len(events):
        t0 = events[i][0]
        if t0 > prev_t:
            # Determine is_net_empty for interval [prev_t, t0)
            # 0 = both teams have goalie, 1 = at least one team has no goalie
            is_empty = 1 if (active_goalies['home'] == 0 or active_goalies['away'] == 0) else 0
            intervals.append({'start': prev_t, 'end': t0, 'is_net_empty': is_empty, 'source': 'shifts'})
            prev_t = t0
        
        # Apply all events at this timestamp
        while i < len(events) and events[i][0] == t0:
            _, typ, team = events[i]
            if typ == 'start':
                active_goalies[team] = active_goalies.get(team, 0) + 1
            else:
                active_goalies[team] = max(0, active_goalies.get(team, 0) - 1)
            i += 1
    
    return pd.DataFrame(intervals)


def compare_game_state_and_net_empty(game_id: int, debug: bool = False) -> Dict[str, Any]:
    """Compare game_state and is_net_empty for a game between API and HTML shift sources.
    
    Returns a dict with:
      - game_id
      - api_shifts_count, html_shifts_count
      - game_state_comparison: dict with match stats
      - is_net_empty_comparison: dict with match stats
      - samples: list of mismatch examples
    """
    logging.info('Fetching shifts for game %s...', game_id)
    
    # Fetch from both sources
    try:
        api_res = nhl_api.get_shifts(game_id, force_refresh=True)
    except Exception as e:
        logging.error('Failed to get API shifts: %s', e)
        api_res = {'game_id': game_id, 'all_shifts': []}
    
    try:
        html_res = nhl_api.get_shifts_from_nhl_html(game_id, force_refresh=True, debug=debug)
    except Exception as e:
        logging.error('Failed to get HTML shifts: %s', e)
        html_res = {'game_id': game_id, 'all_shifts': []}
    
    api_all = api_res.get('all_shifts', []) if isinstance(api_res, dict) else []
    html_all = html_res.get('all_shifts', []) if isinstance(html_res, dict) else []
    
    logging.info('API shifts: %d, HTML shifts: %d', len(api_all), len(html_all))
    
    # Parse to DataFrames
    try:
        df_api = parse._shifts(api_res)
    except Exception as e:
        logging.error('Failed to parse API shifts: %s', e)
        df_api = pd.DataFrame()
    
    try:
        df_html = parse._shifts(html_res)
    except Exception as e:
        logging.error('Failed to parse HTML shifts: %s', e)
        df_html = pd.DataFrame()
    
    # Get team IDs
    try:
        feed = nhl_api.get_game_feed(game_id)
        home_id = feed.get('homeTeam', {}).get('id') if feed else None
        away_id = feed.get('awayTeam', {}).get('id') if feed else None
    except Exception:
        home_id = None
        away_id = None
    
    # Derive game_state intervals from each source
    logging.info('Deriving game_state intervals...')
    api_gs = derive_game_state_from_shifts(df_api, home_id, away_id)
    html_gs = derive_game_state_from_shifts(df_html, home_id, away_id)
    
    # Derive is_net_empty intervals from each source
    logging.info('Deriving is_net_empty intervals...')
    api_ne = derive_is_net_empty_from_shifts(df_api, home_id, away_id)
    html_ne = derive_is_net_empty_from_shifts(df_html, home_id, away_id)
    
    # Compare game_state
    gs_comparison = compare_intervals(api_gs, html_gs, 'label')
    ne_comparison = compare_intervals(api_ne, html_ne, 'is_net_empty')
    
    result = {
        'game_id': game_id,
        'api_shifts_count': len(api_all),
        'html_shifts_count': len(html_all),
        'team_ids': {'home': home_id, 'away': away_id},
        'game_state_comparison': gs_comparison,
        'is_net_empty_comparison': ne_comparison,
    }
    
    return result


def compare_intervals(df_a: pd.DataFrame, df_b: pd.DataFrame, value_col: str) -> Dict[str, Any]:
    """Compare two sets of intervals and return match statistics.
    
    Args:
        df_a: DataFrame with columns start, end, <value_col>
        df_b: DataFrame with columns start, end, <value_col>
        value_col: column name containing the value to compare
    
    Returns dict with:
        - total_seconds_a, total_seconds_b
        - overlap_seconds: total seconds where both sources agree
        - mismatch_seconds: total seconds where sources disagree
        - unique_values_a, unique_values_b
        - sample_mismatches: list of example mismatch intervals
    """
    if df_a is None or df_a.empty:
        return {
            'total_seconds_a': 0.0,
            'total_seconds_b': sum((r['end'] - r['start']) for _, r in df_b.iterrows()) if df_b is not None and not df_b.empty else 0.0,
            'overlap_seconds': 0.0,
            'mismatch_seconds': 0.0,
            'unique_values_a': [],
            'unique_values_b': df_b[value_col].unique().tolist() if df_b is not None and not df_b.empty else [],
            'sample_mismatches': []
        }
    
    if df_b is None or df_b.empty:
        return {
            'total_seconds_a': sum((r['end'] - r['start']) for _, r in df_a.iterrows()),
            'total_seconds_b': 0.0,
            'overlap_seconds': 0.0,
            'mismatch_seconds': 0.0,
            'unique_values_a': df_a[value_col].unique().tolist(),
            'unique_values_b': [],
            'sample_mismatches': []
        }
    
    # Build event timeline
    events = []
    for _, r in df_a.iterrows():
        events.append((float(r['start']), 'start', 'a', r.get(value_col)))
        events.append((float(r['end']), 'end', 'a', r.get(value_col)))
    for _, r in df_b.iterrows():
        events.append((float(r['start']), 'start', 'b', r.get(value_col)))
        events.append((float(r['end']), 'end', 'b', r.get(value_col)))
    
    events.sort(key=lambda x: (x[0], 0 if x[1] == 'end' else 1))
    
    active_a = []
    active_b = []
    prev_t = events[0][0]
    
    total_a = 0.0
    total_b = 0.0
    overlap = 0.0
    mismatch = 0.0
    sample_mismatches = []
    
    i = 0
    while i < len(events):
        t0 = events[i][0]
        if t0 > prev_t:
            dur = t0 - prev_t
            # Determine values at prev_t
            val_a = active_a[-1] if active_a else None
            val_b = active_b[-1] if active_b else None
            
            has_a = val_a is not None
            has_b = val_b is not None
            
            if has_a:
                total_a += dur
            if has_b:
                total_b += dur
            
            if has_a and has_b:
                if val_a == val_b:
                    overlap += dur
                else:
                    mismatch += dur
                    if len(sample_mismatches) < 10:
                        sample_mismatches.append({
                            'start': prev_t,
                            'end': t0,
                            'duration': dur,
                            'value_a': val_a,
                            'value_b': val_b
                        })
            
            prev_t = t0
        
        # Apply all events at this timestamp
        while i < len(events) and events[i][0] == t0:
            _, typ, source, val = events[i]
            if source == 'a':
                if typ == 'start':
                    active_a.append(val)
                else:
                    if active_a:
                        active_a.pop()
            else:
                if typ == 'start':
                    active_b.append(val)
                else:
                    if active_b:
                        active_b.pop()
            i += 1
    
    unique_a = df_a[value_col].unique().tolist() if value_col in df_a.columns else []
    unique_b = df_b[value_col].unique().tolist() if value_col in df_b.columns else []
    
    return {
        'total_seconds_a': float(total_a),
        'total_seconds_b': float(total_b),
        'overlap_seconds': float(overlap),
        'mismatch_seconds': float(mismatch),
        'unique_values_a': unique_a,
        'unique_values_b': unique_b,
        'sample_mismatches': sample_mismatches
    }


def print_comparison_report(result: Dict[str, Any]):
    """Print a human-readable comparison report."""
    print('\n' + '='*70)
    print(f"Game State and Is Net Empty Comparison for Game {result['game_id']}")
    print('='*70)
    
    print(f"\nShift Counts:")
    print(f"  API shifts:  {result['api_shifts_count']}")
    print(f"  HTML shifts: {result['html_shifts_count']}")
    
    print(f"\nTeam IDs:")
    print(f"  Home: {result['team_ids']['home']}")
    print(f"  Away: {result['team_ids']['away']}")
    
    print('\n' + '-'*70)
    print('GAME_STATE Comparison:')
    print('-'*70)
    gs = result['game_state_comparison']
    print(f"  API total seconds:   {gs['total_seconds_a']:.1f}")
    print(f"  HTML total seconds:  {gs['total_seconds_b']:.1f}")
    print(f"  Overlap (matching):  {gs['overlap_seconds']:.1f}s")
    print(f"  Mismatch:            {gs['mismatch_seconds']:.1f}s")
    print(f"  API unique states:   {gs['unique_values_a']}")
    print(f"  HTML unique states:  {gs['unique_values_b']}")
    
    if gs['sample_mismatches']:
        print(f"\n  Sample mismatches (up to 10):")
        for m in gs['sample_mismatches'][:10]:
            print(f"    [{m['start']:.1f} - {m['end']:.1f}] ({m['duration']:.1f}s): API={m['value_a']}, HTML={m['value_b']}")
    else:
        print("  ✓ No mismatches found!")
    
    print('\n' + '-'*70)
    print('IS_NET_EMPTY Comparison:')
    print('-'*70)
    ne = result['is_net_empty_comparison']
    print(f"  API total seconds:   {ne['total_seconds_a']:.1f}")
    print(f"  HTML total seconds:  {ne['total_seconds_b']:.1f}")
    print(f"  Overlap (matching):  {ne['overlap_seconds']:.1f}s")
    print(f"  Mismatch:            {ne['mismatch_seconds']:.1f}s")
    print(f"  API unique values:   {ne['unique_values_a']}")
    print(f"  HTML unique values:  {ne['unique_values_b']}")
    
    if ne['sample_mismatches']:
        print(f"\n  Sample mismatches (up to 10):")
        for m in ne['sample_mismatches'][:10]:
            print(f"    [{m['start']:.1f} - {m['end']:.1f}] ({m['duration']:.1f}s): API={m['value_a']}, HTML={m['value_b']}")
    else:
        print("  ✓ No mismatches found!")
    
    print('\n' + '='*70)


def main():
    parser = argparse.ArgumentParser(description='Compare game_state and is_net_empty from API vs HTML shifts')
    parser.add_argument('--game', '-g', type=int, required=True, help='Game ID to compare')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        result = compare_game_state_and_net_empty(args.game, debug=args.debug)
        print_comparison_report(result)
        return 0
    except Exception as e:
        logging.exception('Comparison failed: %s', e)
        return 1


if __name__ == '__main__':
    sys.exit(main())
