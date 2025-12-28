#!/usr/bin/env python3
"""Trace which shifts are active during a specific time interval.

Usage: python scripts/trace_interval.py <game_id> <start_seconds> <end_seconds> [--source {api|html|both}]

This script helps debug specific game state discrepancies by showing exactly which
shifts (and players) are considered active during a given time range.

Example: python scripts/trace_interval.py 2025020339 457 540
This traces the interval from 457s to 540s where Timo Meier was incorrectly shown as on-ice.
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import nhl_api

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def get_shifts_in_interval(all_shifts: List[Dict[str, Any]], start: float, end: float) -> List[Dict[str, Any]]:
    """Get all shifts that overlap with the given time interval.
    
    A shift overlaps if:
    - It starts before the interval ends AND
    - It ends after the interval starts
    
    Args:
        all_shifts: List of shift dicts with start_total_seconds and end_total_seconds
        start: Start of interval in total seconds
        end: End of interval in total seconds
    
    Returns:
        List of shifts that overlap the interval
    """
    overlapping = []
    
    for shift in all_shifts:
        # Try to get total seconds (preferred) or fall back to period seconds
        shift_start = shift.get('start_total_seconds')
        shift_end = shift.get('end_total_seconds')
        
        if shift_start is None or shift_end is None:
            continue
        
        # Check for overlap
        if shift_start < end and shift_end > start:
            overlapping.append(shift)
    
    return overlapping


def format_shift(shift: Dict[str, Any]) -> str:
    """Format a shift for display."""
    pid = shift.get('player_id', 'N/A')
    pnum = shift.get('player_number', 'N/A')
    pname = shift.get('player_name', 'N/A')
    tid = shift.get('team_id', 'N/A')
    tside = shift.get('team_side', 'N/A')
    start = shift.get('start_total_seconds', 'N/A')
    end = shift.get('end_total_seconds', 'N/A')
    period = shift.get('period', 'N/A')
    
    return f"  Player #{pnum} ({pname}) [ID:{pid}] Team:{tside}({tid}) Period:{period} [{start}s - {end}s]"


def count_skaters_by_team(shifts: List[Dict[str, Any]], 
                           home_id: int = None, away_id: int = None) -> Dict[str, int]:
    """Count active skaters by team at a specific moment.
    
    Note: This is a simplified count that doesn't distinguish goalies.
    For accurate game state, we'd need to classify players as goalies vs skaters.
    
    Args:
        shifts: List of active shifts
        home_id: Team ID for home team
        away_id: Team ID for away team
    
    Returns:
        Dict with counts: {'home': X, 'away': Y, 'unknown': Z}
    """
    counts = {'home': 0, 'away': 0, 'unknown': 0}
    
    # Group players by team
    for shift in shifts:
        tid = shift.get('team_id')
        
        if home_id is not None and tid == home_id:
            counts['home'] += 1
        elif away_id is not None and tid == away_id:
            counts['away'] += 1
        else:
            # Try team_side as fallback
            tside = shift.get('team_side')
            if tside in ('home', 'away'):
                counts[tside] += 1
            else:
                counts['unknown'] += 1
    
    return counts


def trace_interval(game_id: str, start_seconds: float, end_seconds: float, source: str = 'both'):
    """Trace which shifts are active during the specified interval."""
    print(f"Tracing interval [{start_seconds}s - {end_seconds}s] for game {game_id}")
    print(f"Source: {source}")
    print("=" * 80)
    
    # Get team IDs for classification
    try:
        team_ids = nhl_api._get_team_ids(game_id)
        home_id = team_ids.get('home')
        away_id = team_ids.get('away')
        print(f"\nTeam IDs: Home={home_id}, Away={away_id}")
    except Exception as e:
        print(f"Warning: Could not get team IDs: {e}")
        home_id = None
        away_id = None
    
    # Trace API shifts
    if source in ('api', 'both'):
        print("\n" + "-" * 80)
        print("API SHIFTS")
        print("-" * 80)
        
        try:
            api_result = nhl_api.get_shifts(game_id, force_refresh=True)
            api_shifts = api_result.get('all_shifts', [])
            
            print(f"Total API shifts: {len(api_shifts)}")
            
            overlapping = get_shifts_in_interval(api_shifts, start_seconds, end_seconds)
            print(f"Shifts overlapping interval: {len(overlapping)}")
            
            if overlapping:
                # Count by team
                counts = count_skaters_by_team(overlapping, home_id, away_id)
                print(f"\nPlayer count: Home={counts['home']}, Away={counts['away']}, Unknown={counts['unknown']}")
                print(f"Estimated game state: {counts['home']}v{counts['away']}")
                
                print("\nShift details:")
                # Group by player to avoid duplicates
                by_player = {}
                for shift in overlapping:
                    pid = shift.get('player_id')
                    if pid not in by_player:
                        by_player[pid] = shift
                
                # Sort by team and player number
                sorted_shifts = sorted(by_player.values(), 
                                       key=lambda s: (s.get('team_side', 'z'), s.get('player_number', 999)))
                
                for shift in sorted_shifts:
                    print(format_shift(shift))
            else:
                print("No shifts overlap this interval")
                
        except Exception as e:
            print(f"Error tracing API shifts: {e}")
            import traceback
            traceback.print_exc()
    
    # Trace HTML shifts
    if source in ('html', 'both'):
        print("\n" + "-" * 80)
        print("HTML SHIFTS")
        print("-" * 80)
        
        try:
            html_result = nhl_api.get_shifts_from_nhl_html(game_id, force_refresh=True, debug=True)
            html_shifts = html_result.get('all_shifts', [])
            
            print(f"Total HTML shifts: {len(html_shifts)}")
            
            overlapping = get_shifts_in_interval(html_shifts, start_seconds, end_seconds)
            print(f"Shifts overlapping interval: {len(overlapping)}")
            
            if overlapping:
                # Count by team
                counts = count_skaters_by_team(overlapping, home_id, away_id)
                print(f"\nPlayer count: Home={counts['home']}, Away={counts['away']}, Unknown={counts['unknown']}")
                print(f"Estimated game state: {counts['home']}v{counts['away']}")
                
                print("\nShift details:")
                # Group by player to avoid duplicates
                by_player = {}
                for shift in overlapping:
                    pid = shift.get('player_id')
                    pnum = shift.get('player_number')
                    key = pid if pid is not None else f"num_{pnum}"
                    if key not in by_player:
                        by_player[key] = shift
                
                # Sort by team and player number
                sorted_shifts = sorted(by_player.values(), 
                                       key=lambda s: (s.get('team_side', 'z'), s.get('player_number', 999)))
                
                for shift in sorted_shifts:
                    print(format_shift(shift))
            else:
                print("No shifts overlap this interval")
            
            # Print debug info if available
            if html_result.get('debug'):
                print("\nHTML parsing debug info:")
                dbg = html_result['debug']
                print(f"  Tables scanned: {dbg.get('tables_scanned', 0)}")
                print(f"  Players scanned: {dbg.get('players_scanned', 0)}")
                print(f"  Shifts found: {dbg.get('found_shifts', 0)}")
                if 'roster_mapping' in dbg:
                    rm = dbg['roster_mapping']
                    print(f"  Mapped shifts: {rm.get('mapped_shifts', 0)}/{dbg.get('found_shifts', 0)}")
                    print(f"  Unmapped shifts: {rm.get('unmapped_shifts', 0)}")
                    
        except Exception as e:
            print(f"Error tracing HTML shifts: {e}")
            import traceback
            traceback.print_exc()
    
    # Compare if both sources used
    if source == 'both':
        print("\n" + "=" * 80)
        print("COMPARISON")
        print("=" * 80)
        
        try:
            api_result = nhl_api.get_shifts(game_id, force_refresh=True)
            html_result = nhl_api.get_shifts_from_nhl_html(game_id, force_refresh=True, debug=True)
            
            api_overlapping = get_shifts_in_interval(api_result.get('all_shifts', []), start_seconds, end_seconds)
            html_overlapping = get_shifts_in_interval(html_result.get('all_shifts', []), start_seconds, end_seconds)
            
            # Get unique player IDs from each source
            api_players = set()
            for shift in api_overlapping:
                pid = shift.get('player_id')
                if pid is not None:
                    api_players.add(pid)
            
            html_players = set()
            for shift in html_overlapping:
                pid = shift.get('player_id')
                if pid is not None:
                    html_players.add(pid)
            
            only_api = api_players - html_players
            only_html = html_players - api_players
            both = api_players & html_players
            
            print(f"\nPlayers in both sources: {len(both)}")
            print(f"Players only in API: {len(only_api)}")
            if only_api:
                print(f"  Player IDs: {sorted(only_api)}")
                # Show details for these players
                for shift in api_overlapping:
                    if shift.get('player_id') in only_api:
                        print(f"    {format_shift(shift)}")
            
            print(f"Players only in HTML: {len(only_html)}")
            if only_html:
                print(f"  Player IDs: {sorted(only_html)}")
                # Show details for these players
                for shift in html_overlapping:
                    if shift.get('player_id') in only_html:
                        print(f"    {format_shift(shift)}")
                        
        except Exception as e:
            print(f"Error during comparison: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Trace which shifts are active during a specific time interval',
        epilog='Example: python scripts/trace_interval.py 2025020339 457 540'
    )
    parser.add_argument('game_id', type=str, help='Game ID')
    parser.add_argument('start_seconds', type=float, help='Start of interval in total game seconds')
    parser.add_argument('end_seconds', type=float, help='End of interval in total game seconds')
    parser.add_argument('--source', '-s', choices=['api', 'html', 'both'], default='both',
                        help='Which shift source to trace (default: both)')
    
    args = parser.parse_args()
    
    try:
        trace_interval(args.game_id, args.start_seconds, args.end_seconds, args.source)
        return 0
    except Exception as e:
        logging.exception('Interval tracing failed: %s', e)
        return 1


if __name__ == '__main__':
    sys.exit(main())
