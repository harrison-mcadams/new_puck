#!/usr/bin/env python3
"""Save intermediary game data (API responses, HTML pages) for offline analysis.

Usage: python scripts/save_game_data.py <game_id> [--output-dir DIR]

This script fetches and saves:
- Game feed JSON
- Shifts API JSON (if available)
- HTML shift reports (Home and Away)
- Processed shifts from both API and HTML sources

This allows offline inspection and debugging of discrepancies between sources.
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import nhl_api

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def save_game_data(game_id: str, output_dir: str = 'game_data'):
    """Fetch and save all intermediary data for a game."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    game_path = output_path / f'game_{game_id}'
    game_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving game data for {game_id} to {game_path}")
    
    # Save timestamp
    timestamp_file = game_path / 'timestamp.txt'
    timestamp_file.write_text(f"Fetched at: {datetime.now().isoformat()}\n")
    
    # 1. Fetch and save game feed
    print("\n1. Fetching game feed...")
    try:
        feed = nhl_api.get_game_feed(game_id)
        if feed:
            feed_file = game_path / 'game_feed.json'
            with open(feed_file, 'w') as f:
                json.dump(feed, f, indent=2)
            print(f"   Saved game feed to {feed_file}")
        else:
            print("   No game feed available")
    except Exception as e:
        print(f"   Error fetching game feed: {e}")
    
    # 2. Fetch and save API shifts
    print("\n2. Fetching API shifts...")
    try:
        api_shifts = nhl_api.get_shifts(game_id, force_refresh=True)
        if api_shifts:
            # Save full result
            api_file = game_path / 'api_shifts_full.json'
            with open(api_file, 'w') as f:
                json.dump(api_shifts, f, indent=2)
            print(f"   Saved full API shifts to {api_file}")
            
            # Save just all_shifts for easier inspection
            if api_shifts.get('all_shifts'):
                api_shifts_only = game_path / 'api_shifts_only.json'
                with open(api_shifts_only, 'w') as f:
                    json.dump(api_shifts['all_shifts'], f, indent=2)
                print(f"   Saved API all_shifts ({len(api_shifts['all_shifts'])} shifts) to {api_shifts_only}")
        else:
            print("   No API shifts available")
    except Exception as e:
        print(f"   Error fetching API shifts: {e}")
    
    # 3. Fetch and save HTML shifts
    print("\n3. Fetching HTML shifts...")
    try:
        html_shifts = nhl_api.get_shifts_from_nhl_html(game_id, force_refresh=True, debug=True)
        if html_shifts:
            # Save full result
            html_file = game_path / 'html_shifts_full.json'
            with open(html_file, 'w') as f:
                json.dump(html_shifts, f, indent=2, default=str)
            print(f"   Saved full HTML shifts to {html_file}")
            
            # Save just all_shifts
            if html_shifts.get('all_shifts'):
                html_shifts_only = game_path / 'html_shifts_only.json'
                with open(html_shifts_only, 'w') as f:
                    json.dump(html_shifts['all_shifts'], f, indent=2, default=str)
                print(f"   Saved HTML all_shifts ({len(html_shifts['all_shifts'])} shifts) to {html_shifts_only}")
            
            # Save debug info
            if html_shifts.get('debug'):
                debug_file = game_path / 'html_debug.json'
                with open(debug_file, 'w') as f:
                    json.dump(html_shifts['debug'], f, indent=2, default=str)
                print(f"   Saved HTML debug info to {debug_file}")
            
            # Save raw HTML if available
            if html_shifts.get('raw'):
                raw_html_file = game_path / 'raw_html.txt'
                raw_html_file.write_text(html_shifts['raw'])
                print(f"   Saved raw HTML to {raw_html_file}")
        else:
            print("   No HTML shifts available")
    except Exception as e:
        print(f"   Error fetching HTML shifts: {e}")
    
    # 4. Save roster mapping
    print("\n4. Extracting roster mapping...")
    try:
        roster_map = nhl_api._get_roster_mapping(game_id)
        if roster_map:
            roster_file = game_path / 'roster_mapping.json'
            with open(roster_file, 'w') as f:
                json.dump(roster_map, f, indent=2)
            home_count = len(roster_map.get('home', {}))
            away_count = len(roster_map.get('away', {}))
            print(f"   Saved roster mapping to {roster_file} (home: {home_count}, away: {away_count})")
    except Exception as e:
        print(f"   Error extracting roster mapping: {e}")
    
    # 5. Save team IDs
    print("\n5. Extracting team IDs...")
    try:
        team_ids = nhl_api._get_team_ids(game_id)
        if team_ids:
            team_ids_file = game_path / 'team_ids.json'
            with open(team_ids_file, 'w') as f:
                json.dump(team_ids, f, indent=2)
            print(f"   Saved team IDs to {team_ids_file}: {team_ids}")
    except Exception as e:
        print(f"   Error extracting team IDs: {e}")
    
    # 6. Save name-to-ID mapping
    print("\n6. Building name-to-ID mapping...")
    try:
        name_map = nhl_api._build_name_to_id_map(game_id)
        if name_map:
            name_map_file = game_path / 'name_to_id_mapping.json'
            with open(name_map_file, 'w') as f:
                json.dump(name_map, f, indent=2)
            print(f"   Saved name-to-ID mapping to {name_map_file} ({len(name_map)} entries)")
    except Exception as e:
        print(f"   Error building name-to-ID mapping: {e}")
    
    # 7. Create a summary report
    print("\n7. Creating summary report...")
    summary = {
        'game_id': game_id,
        'timestamp': datetime.now().isoformat(),
        'files_created': [str(f.relative_to(output_path)) for f in sorted(game_path.iterdir())],
    }
    
    # Add counts
    try:
        if (game_path / 'api_shifts_only.json').exists():
            with open(game_path / 'api_shifts_only.json', 'r') as f:
                api_shifts_data = json.load(f)
                summary['api_shifts_count'] = len(api_shifts_data)
    except Exception:
        pass
    
    try:
        if (game_path / 'html_shifts_only.json').exists():
            with open(game_path / 'html_shifts_only.json', 'r') as f:
                html_shifts_data = json.load(f)
                summary['html_shifts_count'] = len(html_shifts_data)
    except Exception:
        pass
    
    try:
        if (game_path / 'roster_mapping.json').exists():
            with open(game_path / 'roster_mapping.json', 'r') as f:
                roster_data = json.load(f)
                summary['roster_home_count'] = len(roster_data.get('home', {}))
                summary['roster_away_count'] = len(roster_data.get('away', {}))
    except Exception:
        pass
    
    summary_file = game_path / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   Saved summary to {summary_file}")
    
    print(f"\n✓ All data saved to {game_path}")
    print(f"\nSummary:")
    print(f"  API shifts: {summary.get('api_shifts_count', 'N/A')}")
    print(f"  HTML shifts: {summary.get('html_shifts_count', 'N/A')}")
    print(f"  Roster players: home={summary.get('roster_home_count', 'N/A')}, away={summary.get('roster_away_count', 'N/A')}")
    print(f"  Files: {len(summary['files_created'])}")


def main():
    parser = argparse.ArgumentParser(description='Save intermediary game data for offline analysis')
    parser.add_argument('game_id', type=str, help='Game ID to fetch')
    parser.add_argument('--output-dir', '-o', type=str, default='game_data',
                        help='Output directory (default: game_data)')
    
    args = parser.parse_args()
    
    try:
        save_game_data(args.game_id, args.output_dir)
        return 0
    except Exception as e:
        logging.exception('Failed to save game data: %s', e)
        return 1


if __name__ == '__main__':
    sys.exit(main())
