#!/usr/bin/env python3
"""
Demo and debug script for timing.demo_for_export with player_id(s) condition.

This script runs on the most recent Flyers game, lets you specify player(s) and other conditions,
then runs timing.demo_for_export and prints the results. Intended for interactive debugging.
"""
import argparse
import pprint
import timing
import nhl_api
import parse


def parse_player_list(s):
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


def main():
    parser = argparse.ArgumentParser(description='Demo timing.demo_for_export with player_id(s) on most recent Flyers game')
    parser.add_argument('--players', type=str, default=None, help='Comma-separated player ids (e.g. 8475755,8475794)')
    parser.add_argument('--game_state', type=str, default=None, help='Game state (e.g. 5v5,5v4)')
    parser.add_argument('--is_net_empty', type=str, default=None, help='is_net_empty values (e.g. 0,1)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    # Always use PHI
    team = 'PHI'
    # Get most recent Flyers game id
    try:
        game_id = nhl_api.get_game_id(method='most_recent', team=team)
        print(f'Using most recent Flyers game_id: {game_id}')
    except Exception as e:
        print('Failed to determine most recent Flyers game id:', e)
        return 1

    # Fetch and parse the game directly (skip season/CSV logic)
    try:
        feed = nhl_api.get_game_feed(game_id)
        ev_df = parse._game(feed)
        if ev_df is None or ev_df.empty:
            print(f'No events found for game_id {game_id}.')
            return 1
        df = parse._elaborate(ev_df)
    except Exception as e:
        print(f'Failed to fetch/parse game {game_id}: {e}')
        return 1

    # Build condition dict
    condition = {'team': team}
    if args.players:
        pids = parse_player_list(args.players)
        if len(pids) == 1:
            condition['player_id'] = pids[0]
        elif len(pids) > 1:
            condition['player_ids'] = pids
    if args.game_state:
        gs = [x.strip() for x in args.game_state.split(',') if x.strip()]
        if gs:
            condition['game_state'] = gs
    if args.is_net_empty:
        ine = [int(x) for x in args.is_net_empty.split(',') if x.strip()]
        if ine:
            condition['is_net_empty'] = ine

    print('Condition:', condition)
    print('Running timing.demo_for_export...')
    res = timing.demo_for_export(df, condition=condition, verbose=args.verbose)

    print('\n=== Aggregate results ===')
    pprint.pprint(res.get('aggregate', {}))

    print('\n=== Per-game result ===')
    per_game = res.get('per_game', {})
    for gid, info in per_game.items():
        print(f'Game {gid}:')
        pprint.pprint(info)
        break  # Only one game

    print('\nDone.')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
