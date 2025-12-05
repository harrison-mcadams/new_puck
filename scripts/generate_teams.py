#!/usr/bin/env python3
"""Generate analysis/teams.json from the NHL standings endpoint.

Usage:
  python3 scripts/generate_teams.py [--force] [--date YYYY-MM-DD]

Behavior:
- Fetches current standings from NHL API
- Parses the JSON for team entries and writes analysis/teams.json as an array of
  objects: { "id": int, "name": str, "abbr": str, "logo": url }
- By default will not overwrite an existing analysis/teams.json unless --force is used.

Note: This script makes a single HTTP call.
"""
import json
import sys
from pathlib import Path
from datetime import date

import requests

ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[1]
# Analysis output directory
ANALYSIS = ROOT / 'analysis'
GAMES_PATH = ANALYSIS / 'games_by_team.json'
TEAMS_PATH = ANALYSIS / 'teams.json'


def extract_teams(obj):
    """Recursively search obj for team dicts and return set of (abbr,name).

    The standings JSON can have nested structures. We consider a dict to be a
    team container if it has a 'team' key whose value is a dict with one of
    the expected abbreviation keys (triCode, abbreviation, abbrev, teamAbbrev)
    or if the dict itself contains those keys directly.
    """
    teams = set()

    def visit(o):
        if isinstance(o, dict):
            # direct team-like dict
            abbr = None
            name = None
            # direct keys
            def _pick_scalar(val):
                # If val is a dict like {'default':'X', 'fr':...}, prefer 'default'
                if isinstance(val, dict):
                    if 'default' in val and val.get('default'):
                        return str(val.get('default'))
                    # fall back to first non-empty value
                    for vv in val.values():
                        if vv:
                            return str(vv)
                    return None
                # otherwise return scalar as string
                return str(val) if val is not None else None

            for k in ('triCode', 'abbrev', 'abbreviation', 'teamAbbrev'):
                if k in o and o.get(k) is not None:
                    abbr = _pick_scalar(o.get(k))
                    if abbr:
                        break
            for k in ('name', 'teamName', 'clubName'):
                if k in o and o.get(k) is not None:
                    name = _pick_scalar(o.get(k))
                    if name:
                        break
            if abbr:
                teams.add((abbr, name or abbr))

            # nested 'team' object
            t = o.get('team')
            if isinstance(t, dict):
                t_abbr = None
                t_name = None
                for k in ('triCode', 'abbrev', 'abbreviation', 'teamAbbrev'):
                    if k in t and t.get(k) is not None:
                        t_abbr = _pick_scalar(t.get(k))
                        if t_abbr:
                            break
                for k in ('name', 'teamName', 'clubName'):
                    if k in t and t.get(k) is not None:
                        t_name = _pick_scalar(t.get(k))
                        if t_name:
                            break
                if t_abbr:
                    teams.add((t_abbr, t_name or t_abbr))

            # visit children
            for v in o.values():
                visit(v)
        elif isinstance(o, list):
            for v in o:
                visit(v)

    visit(obj)
    return teams


def fetch_standings(date_str: str):
    url = f'https://api-web.nhle.com/v1/standings/{date_str}'
    headers = {'User-Agent': 'new_puck/0.1 (+https://github.com/harrisonmcadams/new_puck)'}
    resp = requests.get(url, headers=headers, timeout=15)
    # handle common denial/rate-limit responses
    if resp.status_code == 403:
        raise RuntimeError(f'Access denied (403) when fetching standings for {date_str}')
    if resp.status_code == 429:
        raise RuntimeError(f'Rate limited (429) when fetching standings for {date_str}')
    resp.raise_for_status()
    return resp.json()


def main(argv):
    force = '--force' in argv
    # optional --date YYYY-MM-DD
    date_arg = None
    for i, a in enumerate(argv):
        if a == '--date' and i + 1 < len(argv):
            date_arg = argv[i + 1]

    if TEAMS_PATH.exists() and not force:
        print(f"{TEAMS_PATH} already exists. Use --force to overwrite.")
        return 0

    # determine date
    try:
        date_str = date_arg or date.today().isoformat()
    except Exception:
        date_str = date.today().isoformat()

    print(f'Fetching NHL standings for {date_str}...')
    try:
        data = fetch_standings(date_str)
    except Exception as e:
        print('Failed to fetch standings:', e)
        return 2

    # extract teams
    found = extract_teams(data)

    if not found:
        print('No teams found in standings JSON; aborting.')
        return 3

    # build output list sorted by abbreviation
    teams = [{'abbr': abbr, 'name': name} for abbr, name in sorted(found, key=lambda t: t[0])]

    try:
        ANALYSIS.mkdir(parents=True, exist_ok=True)
        with TEAMS_PATH.open('w', encoding='utf-8') as fh:
            json.dump(teams, fh, indent=2, sort_keys=False)
        print(f'Wrote {TEAMS_PATH} with {len(teams)} entries')
        return 0
    except Exception as e:
        print('Failed to write teams.json:', e)
        return 4


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
