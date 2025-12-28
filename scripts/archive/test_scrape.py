"""Quick test harness for parse._scrape using mocked nhl_api functions.

This script avoids network access by monkeypatching the `nhl_api` object
already imported by `parse` and provides a small set of fake game feeds.

Run from the repository root with:
    python3 scripts/test_scrape.py

It will write to `data_test/{season}/` and print the CSV path and a small
preview of the file contents.
"""

from pathlib import Path
import parse
import json
import os

# Provide deterministic fake season games and feeds
FAKE_SEASON = '20252026'

FAKE_GAMES = [
    {'gamePk': 2025010017},
    {'gamePk': 2025020196},
    {'gamePk': 2025020197},
]

game_feed_calls = []


def fake_get_season(team='all', season=FAKE_SEASON):
    # ignore team/season params for the fake; return a list of game dicts
    return FAKE_GAMES


def fake_get_game_feed(game_id):
    # record the call so we can assert caching behavior
    gid = int(game_id)
    game_feed_calls.append(gid)
    # return a minimal, valid-ish game feed dict
    feed = {
        'gamePk': gid,
        'id': gid,
        'plays': [
            {
                'type': {'description': 'shot-on-goal'},
                'details': {'xCoord': 50, 'yCoord': 5, 'playerId': 123, 'eventOwnerTeamId': 1},
                'coordinates': {'x': 50, 'y': 5},
                'period': 1,
                'timeRemaining': '10:00',
                'team': {'triCode': 'AAA', 'id': 1},
            }
        ],
        'homeTeam': {'id': 1, 'abbrev': 'AAA'},
        'awayTeam': {'id': 2, 'abbrev': 'BBB'}
    }
    return feed


def main():
    # Monkeypatch the nhl_api module used by parse
    parse.nhl_api.get_season = fake_get_season
    parse.nhl_api.get_game_feed = fake_get_game_feed

    out_dir = 'data_test'
    # Ensure fresh output
    season_dir = Path(out_dir) / FAKE_SEASON
    if season_dir.exists():
        # remove existing files (only the CSV/JSON that would be created)
        for p in season_dir.iterdir():
            try:
                p.unlink()
            except Exception:
                pass

    csv_path = parse._scrape(season=FAKE_SEASON, team='all', out_dir=out_dir, use_cache=False, max_games=3, max_workers=2, verbose=True)
    print('\n_scrape returned CSV path:', csv_path)

    # Print a small preview of the CSV
    if csv_path and os.path.exists(csv_path):
        print('\nCSV preview:')
        with open(csv_path, 'r', encoding='utf-8') as fh:
            for i, line in enumerate(fh.readlines()):
                print(line.rstrip())
                if i >= 10:
                    break
    else:
        print('CSV not found at', csv_path)

    print('\nNumber of calls to fake_get_game_feed during first run:', len(game_feed_calls))

    # Second run: exercise caching behavior (use_cache=True)
    print('\nSecond run with use_cache=True (should prefer cache)')
    # reset the call log
    game_feed_calls.clear()
    res2 = parse._scrape(season=FAKE_SEASON, team='all', out_dir=out_dir, use_cache=True, max_games=3, max_workers=2, verbose=True, return_raw_feeds=True)
    print('_scrape second run returned (dict):', res2)
    print('Number of calls to fake_get_game_feed during second run:', len(game_feed_calls))


if __name__ == '__main__':
    main()
