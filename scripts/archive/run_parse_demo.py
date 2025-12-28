"""Run a safe demo of parse._scrape with a mocked nhl_api to show output.

This avoids network access and limits to a few fake games.
"""
import parse
import json
import os

FAKE_SEASON = '20252026'
FAKE_GAMES = [{'gamePk': 2025010017}, {'gamePk': 2025020196}, {'gamePk': 2025020197}]

calls = []

def fake_get_season(team='all', season=FAKE_SEASON):
    return FAKE_GAMES


def fake_get_game_feed(gid):
    calls.append(int(gid))
    gid = int(gid)
    return {
        'gamePk': gid,
        'id': gid,
        'plays': [
            {
                'type': {'description': 'shot-on-goal'},
                'details': {'xCoord': 50 + (gid % 5), 'yCoord': 5, 'playerId': 123, 'eventOwnerTeamId': 1},
                'coordinates': {'x': 50 + (gid % 5), 'y': 5},
                'period': 1,
                'timeRemaining': '10:00',
                'team': {'triCode': 'AAA', 'id': 1},
            }
        ],
        'homeTeam': {'id': 1, 'abbrev': 'AAA'},
        'awayTeam': {'id': 2, 'abbrev': 'BBB'}
    }


def main():
    parse.nhl_api.get_season = fake_get_season
    parse.nhl_api.get_game_feed = fake_get_game_feed

    out = parse._scrape(season=FAKE_SEASON, team='all', out_dir='data_test', use_cache=False, max_games=3, max_workers=2, verbose=True, process_elaborated=True, save_elaborated=True, return_raw_feeds=True, save_json=True, save_csv=True)
    print('Demo _scrape returned type:', type(out))
    if isinstance(out, dict):
        sp = out.get('saved_paths', {})
        print('saved_paths:', sp)
        raw = out.get('raw_feeds')
        if raw is not None:
            print('raw_feeds count:', len(raw))
        edf = out.get('elaborated_df')
        if edf is not None:
            print('elaborated_df shape:', getattr(edf, 'shape', None))
            print(edf.head())
    print('Calls to fake_get_game_feed:', calls)


if __name__ == '__main__':
    main()
