"""Compare shifts from API `get_shifts` and HTML fallback `get_shifts_from_nhl_html`.

Usage:
    python scripts/compare_shifts.py GAME_ID [--force]

The script:
 - fetches shifts via `get_shifts(game_id)` and `get_shifts_from_nhl_html(game_id)`
 - fetches play-by-play game feed to attempt mapping jersey numbers -> personId
 - normalizes both outputs into comparable tuples and reports differences
"""
import sys
import json
from pprint import pprint
from typing import Dict, Any, Optional

sys.path.insert(0, '..')  # allow running from repository root

import nhl_api


def try_build_number_map(feed: Dict[str, Any]) -> Dict[int, int]:
    """Attempt to discover jersey number -> personId mapping from the game feed.

    This is heuristic: it searches the feed for dicts that contain a player/"person"
    id and a jersey number. Returns mapping {number: personId}.
    """
    mapping: Dict[int, int] = {}

    def walk(obj):
        if isinstance(obj, dict):
            # common shapes: {'person': {'id': ...}, 'jerseyNumber': '12'} or {'player': {'id': ...}, 'number': '12'}
            pid = None; num = None
            # person/player nested
            if 'person' in obj and isinstance(obj.get('person'), dict):
                p = obj.get('person')
                pid = p.get('id') or p.get('personId')
            if 'player' in obj and isinstance(obj.get('player'), dict):
                p = obj.get('player')
                pid = pid or p.get('id') or p.get('playerId')
            # direct id fields
            for k in ('personId', 'playerId', 'id'):
                if pid is None and k in obj and isinstance(obj.get(k), (int, str)):
                    try:
                        pid_val = int(obj.get(k))
                        pid = pid_val
                    except Exception:
                        pass
            # number fields
            for k in ('jerseyNumber', 'jersey', 'number'):
                if k in obj:
                    try:
                        num = int(str(obj.get(k)).strip())
                    except Exception:
                        num = None
                        break
            if pid is not None and num is not None:
                mapping[num] = int(pid)
            # recurse
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for el in obj:
                walk(el)

    walk(feed)
    return mapping


def normalize_api_shifts(shifts_res: Dict[str, Any]):
    """Return list of normalized tuples for API shifts: (player_id, team_id, period, start, end)
    times rounded to nearest second (int or None).
    """
    out = []
    for s in shifts_res.get('all_shifts', []):
        pid = s.get('player_id')
        tid = s.get('team_id')
        period = s.get('period')
        start = s.get('start_seconds')
        end = s.get('end_seconds')
        out.append((pid, tid, period, int(start) if start is not None else None, int(end) if end is not None else None, s))
    return out


def normalize_html_shifts(html_res: Dict[str, Any], num_map: Optional[Dict[int,int]] = None):
    """Return list of normalized tuples for HTML shifts: (player_number, mapped_person_id_or_None, team_side, period, start, end)
    """
    out = []
    for s in html_res.get('all_shifts', []):
        pnum = s.get('player_number')
        mapped = None
        if pnum is not None and num_map:
            mapped = num_map.get(int(pnum))
        tid = s.get('team_id')
        side = s.get('team_side')
        period = s.get('period')
        start = s.get('start_seconds')
        end = s.get('end_seconds')
        out.append((pnum, mapped, tid, side, period, int(start) if start is not None else None, int(end) if end is not None else None, s))
    return out


def compare(game_id: Any, force: bool = False):
    print('Fetching API shifts (get_shifts)')
    api_shifts = nhl_api.get_shifts(game_id, force_refresh=force)
    print('Fetching HTML shifts (get_shifts_from_nhl_html)')
    html_shifts = nhl_api.get_shifts_from_nhl_html(game_id, force_refresh=force, debug=True)
    print('Fetching game feed (for roster mapping)')
    feed = nhl_api.get_game_feed(game_id)

    num_map = try_build_number_map(feed) if isinstance(feed, dict) else {}
    print('Discovered jersey->person map (partial):')
    pprint(num_map)

    api_norm = normalize_api_shifts(api_shifts)
    html_norm = normalize_html_shifts(html_shifts, num_map)

    print('\nAPI shifts: total=%d' % len(api_norm))
    print('HTML shifts: total=%d' % len(html_norm))

    # Build simple lookup sets
    api_set = set()
    for pid, tid, per, st, ed, raw in api_norm:
        api_set.add(('pid', pid, tid, per, st, ed))
    html_set = set()
    for pnum, mapped, tid, side, per, st, ed, raw in html_norm:
        # use mapped person id if available, otherwise use number token
        key_id = mapped if mapped is not None else ('num', pnum)
        html_set.add(('pid', key_id, tid, per, st, ed))

    only_api = [x for x in api_set if x not in html_set]
    only_html = [x for x in html_set if x not in api_set]

    print('\nShifts present in API but not in HTML (sample 20):')
    for x in only_api[:20]:
        print(x)
    print('\nShifts present in HTML but not in API (sample 20):')
    for x in only_html[:20]:
        print(x)

    # Save detailed JSON for inspection
    out = {
        'game_id': game_id,
        'api': api_shifts,
        'html': html_shifts,
        'jersey_map': num_map,
        'only_api': list(only_api),
        'only_html': list(only_html),
    }
    with open(f'.cache/compare_shifts_{game_id}.json', 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2)
    print('\nDetailed comparison written to .cache/compare_shifts_%s.json' % game_id)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python scripts/compare_shifts.py GAME_ID [--force]')
        sys.exit(1)
    game = sys.argv[1]
    force_flag = '--force' in sys.argv or '-f' in sys.argv
    compare(game, force=force_flag)

