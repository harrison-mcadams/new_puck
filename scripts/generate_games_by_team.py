#!/usr/bin/env python3
"""Generate static/games_by_team.json by querying the NHL club-schedule-season endpoint per team.

Usage:
  python3 scripts/generate_games_by_team.py [--force] [--season 20252026] [--no-api]

Behavior:
- Reads `static/teams.json` for team abbreviations.
- For each team, calls `nhl_api.get_season(team=TEAM_ABB, season=season)` to retrieve games,
  unless `--no-api` is provided which forces generation without network calls.
- Builds a mapping {team_abbr: [ {id, label, start}, ... ] }
- Writes `static/games_by_team.json` (overwrites only with --force).

Notes:
- This script uses the local `nhl_api.get_season` helper which already handles
  HTTP 403/429 by returning an empty list in many cases; this script also catches
  exceptions and continues. If the NHL API denies access, this script will
  include empty arrays for affected teams instead of crashing.
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
import time
import requests
import traceback
import os
from datetime import datetime

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
    no_api = '--no-api' in argv
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

    # Debug startup info (printed after we decide how we'll fetch)
    print(f'generate_games_by_team: force={force}, no_api={no_api}, season={season}')

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

    # If an existing OUT_PATH exists, load its mapping so we can reuse prior
    # entries for teams that fail to fetch (helps when API access is denied).
    prior_results: Dict[str, List[Dict[str, Any]]] = {}
    if OUT_PATH.exists():
        try:
            with OUT_PATH.open('r', encoding='utf-8') as fh:
                prior_results = json.load(fh)
        except Exception:
            prior_results = {}

    # Conditionally import nhl_api only if API use is desired. If import fails
    # we'll fall back to a direct HTTP fetch for the club-schedule-season URL.
    nhl_api = None
    use_direct_http = False
    if not no_api:
        try:
            import nhl_api as _nhl_api
            nhl_api = _nhl_api
        except Exception as e:
            print('Warning: failed to import nhl_api; will attempt direct HTTP fetches instead:', e)
            traceback.print_exc()
            use_direct_http = True

    # Allow overriding behavior via environment: FORCE_DIRECT=1 forces direct HTTP fetches
    if os.environ.get('FORCE_DIRECT', '') == '1':
        print('FORCE_DIRECT=1 detected: forcing direct HTTP fetches (overriding --no-api)')
        no_api = False
        use_direct_http = True
    print(f'Operational flags: no_api={no_api}, use_direct_http={use_direct_http}, nhl_api_imported={nhl_api is not None}')

    # helper to find games in various JSON shapes (reusable)
    def find_games(obj):
        found = []
        if isinstance(obj, dict):
            # direct 'games' key
            if 'games' in obj and isinstance(obj.get('games'), list):
                for g in obj.get('games') or []:
                    if isinstance(g, dict):
                        found.append(g)
            # 'dates' -> each date may have 'games'
            if 'dates' in obj and isinstance(obj.get('dates'), list):
                for d in obj.get('dates') or []:
                    if isinstance(d, dict) and isinstance(d.get('games'), list):
                        for g in d.get('games') or []:
                            if isinstance(g, dict):
                                found.append(g)
            # 'gameWeek' shape
            if 'gameWeek' in obj and isinstance(obj.get('gameWeek'), list):
                for day in obj.get('gameWeek') or []:
                    if isinstance(day, dict):
                        for g in day.get('games', []) or []:
                            if isinstance(g, dict):
                                found.append(g)
            # recurse
            for v in obj.values():
                try:
                    found.extend(find_games(v))
                except Exception:
                    continue
        elif isinstance(obj, list):
            for item in obj:
                try:
                    found.extend(find_games(item))
                except Exception:
                    continue
        return found

    def is_preseason_game(g: Dict[str, Any]) -> bool:
        """Return True if the game dict appears to be a preseason game.

        Use several heuristics to detect preseason games across different
        API shapes: explicit flags, gameType codes, or season/seasonType
        labels containing 'pre' or 'pr'. This is intentionally permissive.
        """
        if not isinstance(g, dict):
            return False
        # explicit flag some feeds include
        try:
            if g.get('isPreseason') is True:
                return True
        except Exception:
            pass
        # check common top-level keys
        candidates = []
        try:
            if 'gameType' in g:
                candidates.append(str(g.get('gameType')))
            if 'seasonType' in g:
                candidates.append(str(g.get('seasonType')))
            if 'seasonTypeName' in g:
                candidates.append(str(g.get('seasonTypeName')))
            if 'type' in g:
                t = g.get('type')
                if isinstance(t, dict):
                    candidates.append(str(t.get('description') or t.get('type') or t.get('code') or ''))
                else:
                    candidates.append(str(t))
        except Exception:
            pass

        for c in candidates:
            try:
                if not c:
                    continue
                cl = c.strip().lower()
                # common short code for preseason is 'pr' or strings containing 'pre'
                if cl == 'pr' or 'pre' in cl or 'preseason' in cl:
                    return True
            except Exception:
                continue
        # Additional heuristics: game id pattern and start date
        try:
            gid = g.get('id') or g.get('gamePk') or g.get('gameID')
            if gid is not None:
                gid_s = str(gid)
                # season start year like '2025' from main()'s season string
                # We'll infer the season start from the 'season' variable in outer scope
                # fallback if not present
                season_start = str(season)[:4] if season is not None else None
                if season_start and gid_s.startswith(season_start + '01'):
                    return True
        except Exception:
            pass

        try:
            start = g.get('startTimeUTC') or g.get('gameDate') or g.get('startTime') or g.get('start')
            if start:
                # parse ISO-ish datetime defensively
                try:
                    dt = datetime.fromisoformat(str(start).replace('Z', '+00:00'))
                except Exception:
                    try:
                        dt = datetime.strptime(str(start)[:19], '%Y-%m-%dT%H:%M:%S')
                    except Exception:
                        dt = None
                if dt is not None:
                    # preseason typically occurs in September (month < 10) for the season start year
                    season_start = int(str(season)[:4]) if season is not None else None
                    if season_start is not None and dt.year == season_start and dt.month < 10:
                        return True
        except Exception:
            pass

        return False

    def direct_fetch_season(team_abbr: str, season_str: str):
        """Directly fetch the club-schedule-season endpoint for a team and
        return a list of game dicts (or [] on error).
        """
        url = f"https://api-web.nhle.com/v1/club-schedule-season/{team_abbr}/{season_str}"
        headers = {'User-Agent': 'new_puck/0.1 (+https://github.com/harrisonmcadams/new_puck)'}
        try:
            resp = requests.get(url, headers=headers, timeout=15)
        except Exception as e:
            print(f'    direct fetch error for {team_abbr}:', e)
            return []
        if resp.status_code == 403:
            print(f'    direct fetch: access denied (403) for {team_abbr} {season_str}')
            return []
        if resp.status_code == 429:
            print(f'    direct fetch: rate limited (429) for {team_abbr} {season_str}')
            return []
        try:
            data = resp.json()
        except Exception as e:
            print(f'    failed to parse JSON for {team_abbr}:', e)
            return []

        games_list = find_games(data)
        # deduplicate by gamePk/id if possible
        seen = set()
        out = []
        for g in games_list:
            gid = g.get('id') or g.get('gamePk') or g.get('gameID')
            if gid is None:
                out.append(g)
                continue
            if gid in seen:
                continue
            seen.add(gid)
            out.append(g)
        return out

    for abbr in team_abbrs:
        print(f'Processing team {abbr}...')
        entries: List[Dict[str, Any]] = []
        if no_api:
            # In no-api mode we intentionally leave entries empty so the UI
            # can still function; user can later populate games_by_team.json.
            print(f'  Skipping NHL API call for {abbr} (no-api mode).')
            results[abbr] = entries
            continue

        # attempt to fetch games with retry/backoff for transient errors
        max_attempts = 4
        attempt = 0
        games = []
        last_err = None
        backoff = 1.0
        while attempt < max_attempts:
            attempt += 1
            try:
                if nhl_api is not None:
                    raw = nhl_api.get_season(team=abbr, season=season)
                    # nhl_api may return a list or a dict (with nested games); handle both
                    if isinstance(raw, dict):
                        # extract games from dict structure
                        extracted = find_games(raw)
                        games = extracted
                        print(f'    nhl_api returned dict for {abbr}; extracted {len(games)} games')
                    elif isinstance(raw, list):
                        games = raw
                        print(f'    nhl_api returned list for {abbr}; {len(games)} games')
                    else:
                        games = []
                        print(f'    nhl_api returned unexpected type {type(raw)} for {abbr}; treating as no games')
                    # If no games found, try direct HTTP fallback
                    if not games:
                        alt = direct_fetch_season(abbr, season)
                        if isinstance(alt, list) and len(alt) > 0:
                            print(f'    direct HTTP fallback returned {len(alt)} games for {abbr}')
                            games = alt
                else:
                    # fallback to direct HTTP fetch if import failed
                    games = direct_fetch_season(abbr, season)
                # nhl_api.get_season should return [] on 403/429 in many cases but
                # may raise for unexpected errors. If it's successful we break.
                break
            except Exception as e:
                last_err = e
                print(f'  Attempt {attempt}: error fetching season for {abbr}: {e}')
                traceback.print_exc()
                if attempt < max_attempts:
                    sleep_for = backoff
                    print(f'    Retrying after {sleep_for:.1f}s...')
                    time.sleep(sleep_for)
                    backoff = min(60.0, backoff * 2)
                else:
                    print(f'    Giving up on {abbr}; will attempt to use prior cached games (if any).')
                    games = []
                    break

        if isinstance(games, list) and games:
            # Filter out preseason games
            try:
                filtered = [g for g in games if not is_preseason_game(g)]
            except Exception:
                filtered = list(games)
            if len(filtered) < len(games):
                print(f'    Filtered out {len(games) - len(filtered)} preseason games for {abbr}')
            for g in filtered:
                try:
                    ne = normalize_game_entry(g, abbr)
                    if ne is not None:
                        entries.append(ne)
                except Exception:
                    continue
            results[abbr] = entries
        else:
            # No games returned from API. Try to reuse prior cached entries to
            # avoid losing previously gathered schedule data (common when API
            # access is denied). If none available, leave empty list.
            prior = prior_results.get(abbr) if isinstance(prior_results, dict) else None
            if prior:
                print(f'  No games from API for {abbr}; using {len(prior)} cached games from previous run.')
                results[abbr] = prior
            else:
                if last_err is not None:
                    print(f'  No games and last error for {abbr}: {last_err}')
                else:
                    print(f'  No games found for {abbr}.')
                # Direct HTTP fetch diagnostic
                diag = direct_fetch_season(abbr, season)
                if isinstance(diag, list) and len(diag) > 0:
                    print(f'  Direct HTTP fetch succeeded for {abbr}, but nhl_api failed.')
                else:
                    print(f'  Direct HTTP fetch also failed for {abbr}.')
                results[abbr] = []
        # polite pause between requests to avoid hitting rate limits
        time.sleep(0.25)

    # After processing all teams, print a summary of teams with zero games
    zero_game_teams = [abbr for abbr, entries in results.items() if len(entries) == 0]
    if zero_game_teams:
        print(f'\nTeams with zero games: {", ".join(zero_game_teams)}')

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
