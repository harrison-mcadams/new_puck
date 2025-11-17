#!/usr/bin/env python3
"""Quick test harness for experimenting with nhl_api.get_shifts + parse._shifts

This standalone script is intentionally disposable (you can delete it later).
It provides a simple CLI to fetch shift data for a game, convert to intervals,
and optionally compute the union or intersection of intervals across a list
of player IDs.

Usage examples:
  # Fetch most-recent PHI game and show per-shift sample
  python scripts/test_shifts.py

  # Fetch specific game and show union of two players' shifts
  python scripts/test_shifts.py --game 2025020293 --players 8475755,8475794 --combine union

  # Treat parsed times as "time remaining in period"
  python scripts/test_shifts.py --game 2025020293 --time-remaining

The script prints a compact summary and a small sample of the produced DataFrame
for quick manual inspection.
"""

import argparse
import pprint
import sys

import nhl_api
import parse


def parse_player_list(s: str):
    if not s:
        return None
    out = []
    for token in s.split(','):
        t = token.strip()
        if not t:
            continue
        try:
            out.append(int(t))
        except Exception:
            out.append(t)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description='Test shifts -> intervals')
    parser.add_argument('--game', '-g', type=int, default=None, help='Game ID to fetch (default: most-recent PHI)')
    parser.add_argument('--players', '-p', type=str, default=None, help='Comma-separated player ids to filter/combine')
    parser.add_argument('--combine', '-c', type=str, default=None, choices=['union', 'intersection', 'none'], help="Combine intervals across players: 'union'|'intersection'|'none'")
    parser.add_argument('--time-remaining', action='store_true', help='Treat parsed start/end times as time-remaining in period')
    parser.add_argument('--sample', type=int, default=10, help='Number of sample rows to print')

    args = parser.parse_args(argv)

    game_id = args.game
    try:
        if game_id is None:
            game_id = nhl_api.get_game_id(method='most_recent', team='PHI')
    except Exception as e:
        print('Failed to determine game id:', e, file=sys.stderr)
        return 2

    print(f'Using game_id: {game_id}')

    print('Fetching shifts...')
    shifts_res = nhl_api.get_shifts(game_id)
    if not shifts_res or shifts_res.get('raw') is None:
        print('No shifts data available (API returned empty or access denied).')
        return 3

    total_shifts = len(shifts_res.get('all_shifts', []))
    players_avail = list(shifts_res.get('shifts_by_player', {}).keys())
    print(f'Parsed shifts: total={total_shifts}, players={len(players_avail)}')

    player_ids = parse_player_list(args.players) if args.players else None
    combine = None if args.combine in (None, 'none') else args.combine

    print('Calling parse._shifts(...). This may take a second...')
    df = parse._shifts(shifts_res, player_ids=player_ids, time_is_remaining=args.time_remaining, combine=combine)

    if df is None or (hasattr(df, 'empty') and df.empty):
        print('Resulting DataFrame is empty.')
        return 0

    print('\nResult DataFrame shape:', getattr(df, 'shape', None))

    # Print sample rows; if combined intervals small, show all
    n = args.sample
    try:
        if combine:
            print(f"Combined intervals (mode={combine}) - count={len(df)}")
            if len(df) <= max(20, n*2):
                print(df.to_string(index=False))
            else:
                print(df.head(n).to_string(index=False))
                print('...')
                print(df.tail(n).to_string(index=False))
        else:
            print(f'Per-shift sample (first {n} rows):')
            print(df.head(n).to_string(index=False))
    except Exception as e:
        print('Error printing DataFrame sample:', e)

    # print out some summary metrics
    # print out total across all intervals for either the player ids inputted
    # or the combination option provided

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

