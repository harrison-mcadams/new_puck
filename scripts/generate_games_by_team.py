#!/usr/bin/env python3
"""Generate static/games_by_team.json by querying the NHL club-schedule-season endpoint per team.

Usage:
  python3 scripts/generate_games_by_team.py [--force] [--season 20252026]

Behavior:
- Reads `static/teams.json` for team abbreviations.
- For each team, calls `nhl_api.get_season(team=TEAM_ABB, season=season)` to retrieve games.
- Builds a mapping {team_abbr: [ {id, label, start}, ... ] }
- Writes `static/games_by_team.json` (overwrites only with --force).

Notes:
- This script uses the local `nhl_api.get_season` helper which already handles
  HTTP 403/429 by returning an empty list; the script will continue on errors
  and include empty arrays for teams with no available games.
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / 'static'
TEAMS_PATH = STATIC / 'teams.json'
OUT_PATH = STATIC / 'games_by_team.json'


def normalize_game_entry(g: Dict[str, Any], team_abbr: str):
    # extract id
    gid = g.get('id') or g.get('gamePk') or g.get('gameID')
    if gid is None:
        return None
    try:
        gid_val = int(gid)
    except Exception:
        gid_val = str(gid)

    start = g.get('startTimeUTC') or g.get('gameDate') or g.get('startTime') or ''

    # derive opponent and home/away
    opp = ''
    homeaway = ''
    try:
        teams = g.get('teams') or {}
        if isinstance(teams, dict):
            home = teams.get('home') or {}
            away = teams.get('away') or {}
            # extract inner 'team' dict if present
            if isinstance(home, dict) and 'team' in home and isinstance(home['team'], dict):
                home_team = home['team']
            else:
                home_team = home if isinstance(home, dict) else {}
            if isinstance(away, dict) and 'team' in away and isinstance(away['team'], dict):
                away_team = away['team']
            else:
                away_team = away if isinstance(away, dict) else {}

            home_ab = (home_team.get('triCode') or home_team.get('abbrev') or home_team.get('teamAbbrev')) if isinstance(home_team, dict) else None
            away_ab = (away_team.get('triCode') or away_team.get('abbrev') or away_team.get('teamAbbrev')) if isinstance(away_team, dict) else None
            if str(home_ab).upper() == str(team_abbr).upper():
                opp = away_ab or ''
                homeaway = 'home'
            elif str(away_ab).upper() == str(team_abbr).upper():
                opp = home_ab or ''
                homeaway = 'away'
    except Exception:
        pass

    if opp:
        label = f"{start} - vs {opp} ({homeaway})"
    else:
        label = f"{start} - id {gid_val}"

    return {'id': gid_val, 'label': label, 'start': start}


def main(argv: List[str]):
    force = '--force' in argv
    # season string like '20252026'
    season = None
    if '--season' in argv:
        try:
            si = argv.index('--season')
            season = argv[si + 1]
        except Exception:
            season = None
    if season is None:
        season = '20252026'

    if not TEAMS_PATH.exists():
        print(f"Missing {TEAMS_PATH}; generate teams.json first or create it manually.")
        return 2

    if OUT_PATH.exists() and not force:
        print(f"{OUT_PATH} already exists. Use --force to overwrite.")
        return 0

    try:
        with TEAMS_PATH.open('r', encoding='utf-8') as fh:
            teams = json.load(fh)
    except Exception as e:
        print('Failed to read teams.json:', e)
        return 3

    # Ensure teams is a list of objects with 'abbr'
    team_abbrs = [t.get('abbr') for t in teams if isinstance(t, dict) and t.get('abbr')]

    results: Dict[str, List[Dict[str, Any]]] = {}

    # Import nhl_api locally so script can run even if module has heavy imports
    try:
        import nhl_api
    except Exception as e:
        print('Failed to import nhl_api:', e)
        return 4

    for abbr in team_abbrs:
        print(f'Fetching season for team {abbr}...')
        try:
            games = nhl_api.get_season(team=abbr, season=season)
        except Exception as e:
            print(f'Error fetching season for {abbr}:', e)
            games = []

        entries = []
        if isinstance(games, list):
            for g in games:
                try:
                    ne = normalize_game_entry(g, abbr)
                    if ne is not None:
                        entries.append(ne)
                except Exception:
                    continue
        results[abbr] = entries

    try:
        STATIC.mkdir(parents=True, exist_ok=True)
        with OUT_PATH.open('w', encoding='utf-8') as fh:
            json.dump(results, fh, indent=2, sort_keys=True)
        print(f'Wrote {OUT_PATH} with {len(results)} teams')
        return 0
    except Exception as e:
        print('Failed to write games_by_team.json:', e)
        return 5


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))

