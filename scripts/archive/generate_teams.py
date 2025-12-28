#!/usr/bin/env python3
"""Generate analysis/teams.json from the NHL Team Stats endpoint.

Usage:
  python3 scripts/generate_teams.py [--force]

Behavior:
- Fetches team list from https://api.nhle.com/stats/rest/en/team
- Parses the JSON for team entries and writes analysis/teams.json as an array of
  objects: { "id": int, "name": str, "abbr": str }
- By default will not overwrite an existing analysis/teams.json unless --force is used.
"""
import json
import sys
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[1]
# Output directory (web/static)
STATIC_DIR = ROOT / 'web' / 'static'
TEAMS_PATH = STATIC_DIR / 'teams.json'


def fetch_teams():
    url = 'https://api.nhle.com/stats/rest/en/team'
    headers = {'User-Agent': 'new_puck/0.1 (+https://github.com/harrisonmcadams/new_puck)'}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()


def main(argv):
    force = '--force' in argv
    
    if TEAMS_PATH.exists() and not force:
        print(f"{TEAMS_PATH} already exists. Use --force to overwrite.")
        return 0

    print('Fetching NHL teams...')
    try:
        data = fetch_teams()
    except Exception as e:
        print('Failed to fetch teams:', e)
        return 2

    # Parse response
    # Expected format: { "data": [ { "id": 11, "triCode": "ATL", "fullName": "Atlanta Thrashers", ... }, ... ] }
    
    raw_list = data.get('data', [])
    if not raw_list:
        print('No teams found in JSON response.')
        return 3
        
    teams = []
    for t in raw_list:
        tid = t.get('id')
        abbr = t.get('triCode')
        name = t.get('fullName')
        
        # Only keep active or reasonable teams? ID is simpler.
        if tid is not None and abbr:
            teams.append({
                'id': int(tid),
                'abbr': abbr,
                'name': name or abbr
            })
            
    # Sort by ID
    teams.sort(key=lambda x: x['id'])

    try:
        STATIC_DIR.mkdir(parents=True, exist_ok=True)
        with TEAMS_PATH.open('w', encoding='utf-8') as fh:
            json.dump(teams, fh, indent=2, sort_keys=False)
        print(f'Wrote {TEAMS_PATH} with {len(teams)} entries')
        return 0
    except Exception as e:
        print('Failed to write teams.json:', e)
        return 4


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
