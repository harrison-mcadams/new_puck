"""Small NHL API helpers used by the demo.

This module provides a lightweight helper to locate a recent game ID for a
team (currently defaulting to PHI) from the api-web schedule endpoint, and a
simple helper to fetch the play-by-play feed for a given game ID.

The helpers are intentionally small and defensive: they try a couple of common
timestamp formats and return the most-recent past game (or the next future
one if no past games are available).
"""

import logging
import requests
import random
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
import email.utils as email_utils


# module-level session for connection reuse and default headers
SESSION = requests.Session()
SESSION.headers.update({
    'User-Agent': 'new_puck/0.1 (+https://github.com/harrisonmcadams/new_puck)'
})


def get_season(team: str = 'PHI', season: str = '20252026') -> List[Dict[str, Any]]:
    """Return the season's games list for the given team.

    Parameters
    - team: optional team abbreviation (e.g. 'PHI', 'NYR') or 'all'. Defaults to 'PHI'.
    - season: optional season string (e.g. '20252026'). Defaults to '20252026'.

    This function fetches the api-web schedule for the team and returns the
    top-level 'games' list from the response (or an empty list).
    """
    # Handle the special 'all' request which pages weekly schedule endpoints
    if team == 'all':
        # Pare the season string to a start year (e.g. '20252026' -> '2025')
        season_start_year = season[:4] if season and len(season) >= 4 else '2025'
        try:
            start_year = int(season_start_year)
        except Exception:
            start_year = 2025

        # season typically starts in October of the start year
        week_start_dt = datetime(start_year, 10, 1)
        # stop once we've passed the following calendar year (safe upper bound)
        stop_year = start_year + 1

        games: List[Dict[str, Any]] = []

        # Page week-by-week until we've covered the season year
        while week_start_dt.year <= stop_year:
            date_str = week_start_dt.strftime('%Y-%m-%d')
            url = f'https://api-web.nhle.com/v1/schedule/{date_str}'

            try:
                resp = SESSION.get(url, timeout=10)
                try:
                    resp.raise_for_status()
                except Exception as he:
                    # If the API explicitly denies access (403) or is rate-limiting (429),
                    # return an empty list so callers can handle the absence of data
                    status = getattr(resp, 'status_code', None)
                    if status in (403, 429):
                        logging.warning('get_season: HTTP %s received for %s; returning empty games list', status, url)
                        return []
                    # otherwise re-raise
                    raise
                data = resp.json()
            except Exception:
                # stop on any network or parsing error — we've gathered what we can
                break

            # The schedule endpoint can return either a 'gameWeek' structure or
            # a 'dates' list; handle both shapes defensively.
            if isinstance(data, dict):
                # handle 'gameWeek' shape (older / weekly API)
                gw = data.get('gameWeek')
                if isinstance(gw, list):
                    for day in gw:
                        if not isinstance(day, dict):
                            continue
                        for g in day.get('games', []) or []:
                            if isinstance(g, dict):
                                games.append(g)

                # handle 'dates' shape (the more common schedule API)
                dates = data.get('dates')
                if isinstance(dates, list):
                    for d in dates:
                        if not isinstance(d, dict):
                            continue
                        for g in d.get('games', []) or []:
                            if isinstance(g, dict):
                                games.append(g)

            # advance by one week
            week_start_dt = week_start_dt + timedelta(days=7)

        return games

    TEAM_ABB = (team or 'PHI').upper()

    # The api-web endpoint for a club's schedule for a season
    url = (f"https://api-web.nhle.com/v1/club-schedule-season/"
           f"{TEAM_ABB}/{season}")

    resp = SESSION.get(url, timeout=10)
    try:
        resp.raise_for_status()
    except Exception as he:
        # If the API explicitly denies access (403) or is rate-limiting (429),
        # return an empty list so callers can handle the absence of data
        status = getattr(resp, 'status_code', None)
        if status in (403, 429):
            logging.warning('get_season: HTTP %s received for %s; returning empty games list', status, url)
            return []
        # otherwise re-raise
        raise

    # be defensive about the JSON shape
    data = resp.json()
    if not isinstance(data, dict):
        return []

    games_raw = data.get('games')
    if not isinstance(games_raw, list):
        return []

    # ensure entries are dicts
    games: List[Dict[str, Any]] = [g for g in games_raw if isinstance(g, dict)]

    return games


def get_game_id(method: str = 'most_recent', team: str = 'PHI') -> int:
    """Return a game ID for the team's schedule found in the api-web response.

    Parameters
    - method: 'most_recent' (default) returns the most recent game on or before
      now; if none found, return the next future game.
    - team: optional team abbreviation (e.g. 'PHI', 'NYR'). Defaults to 'PHI'.

    This function expects the api-web response to contain a top-level 'games'
    list where each game has an 'id' and a start time such as 'startTimeUTC'.

    The function is defensive about timestamp shapes: it first expects an
    ISO-like UTC format (YYYY-MM-DDTHH:MM:SSZ) and will fall back to
    datetime.fromisoformat for other forms that include offsets.
    """
    TEAM_ABB = (team or 'PHI').upper()

    games = get_season(team=TEAM_ABB) or []  # ensure we have a list

    now_utc = datetime.now(timezone.utc)

    past_games: List[Dict[str, Any]] = []
    future_games: List[Dict[str, Any]] = []

    for game in games:
        # skip unexpected entries
        if not isinstance(game, dict):
            continue

        # expected keys: 'id', 'startTimeUTC' e.g. '2025-09-21T23:00:00Z'
        game_id_raw = game.get('id') or game.get('gamePk') or game.get('gameID')
        start_ts = game.get('startTimeUTC') or game.get('gameDate') or game.get('startTime')
        if game_id_raw is None or start_ts is None:
            continue

        # normalize id to int when possible
        try:
            game_id = int(game_id_raw)
        except Exception:
            # fallback: try string -> strip and cast
            try:
                game_id = int(str(game_id_raw).strip())
            except Exception:
                continue

        try:
            # prefer the simple UTC 'Z' format
            start_dt = datetime.strptime(start_ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except Exception:
            # try parsing with fromisoformat as fallback (may include offset)
            try:
                start_dt = datetime.fromisoformat(start_ts)
                if start_dt.tzinfo is None:
                    start_dt = start_dt.replace(tzinfo=timezone.utc)
            except Exception:
                # skip entries that we cannot parse
                continue
        if start_dt <= now_utc:
            past_games.append({'id': game_id, 'start': start_dt})
        else:
            future_games.append({'id': game_id, 'start': start_dt})

    # prefer the most recent past game
    if past_games:
        # sort by start and return last (most recent)
        past_games.sort(key=lambda g: g['start'])
        return int(past_games[-1]['id'])

    # otherwise return the soonest future game
    if future_games:
        future_games.sort(key=lambda g: g['start'])
        return int(future_games[0]['id'])

    raise RuntimeError("No games found in api-web schedule response.")


def get_game_feed(game_id: int, max_retries: int = 8, backoff_base: float =
1.0, max_backoff: float = 300.0) -> Dict[str, Any]:
    """Fetch and return the play-by-play feed JSON for the requested game.

    The function retries on 429 responses honoring Retry-After headers when
    present, and performs exponential backoff on transient network errors.
    """
    url = f'https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play'

    backoff = backoff_base
    for attempt in range(1, max_retries + 1):
        try:
            resp = SESSION.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.HTTPError as he:
            resp = getattr(he, 'response', None) or (locals().get('resp') if 'resp' in locals() else None)
            status = getattr(resp, 'status_code', None) if resp is not None else None

            # Treat 429 (rate-limited) specially and retry with backoff. If 403
            # (forbidden), log and return empty dict so callers can handle it.
            if status == 429:
                # parse Retry-After header if present
                ra_hdr = None
                try:
                    ra_hdr = resp.headers.get('Retry-After') if resp is not None and resp.headers is not None else None
                except Exception:
                    ra_hdr = None

                sleep_for = None
                if ra_hdr:
                    try:
                        sleep_for = float(ra_hdr)
                    except Exception:
                        try:
                            parsed = email_utils.parsedate_to_datetime(ra_hdr)
                            if parsed.tzinfo is None:
                                parsed = parsed.replace(tzinfo=timezone.utc)
                            delta = (parsed - datetime.now(timezone.utc)).total_seconds()
                            sleep_for = max(0.0, delta)
                        except Exception:
                            sleep_for = None

                if sleep_for is None:
                    sleep_for = backoff
                sleep_for = min(max_backoff, float(sleep_for)) + random.uniform(0, 1.0)
                logging.warning('get_game_feed: 429 for game %s, attempt %d/%d; sleeping %.1fs (Retry-After=%s)', game_id, attempt, max_retries, sleep_for, ra_hdr)
                time.sleep(sleep_for)
                backoff = min(max_backoff, backoff * 2)
                continue

            elif status == 403:
                logging.warning('get_game_feed: access denied (403) for game %s; returning empty feed', game_id)
                return {}

            else:
                # non-retryable HTTP error — re-raise for caller to handle
                logging.warning('get_game_feed HTTP error for game %s: %s', game_id, he)
                raise

        except requests.exceptions.RequestException as re:
            logging.warning('get_game_feed network error for game %s: %s — attempt %d/%d, sleeping %.1fs', game_id, re, attempt, max_retries, backoff)
            time.sleep(backoff + random.uniform(0, 1.0))
            backoff = min(max_backoff, backoff * 2)
            continue

        except Exception as e:
            logging.warning('get_game_feed unexpected error for game %s: %s', game_id, e)
            raise

    raise requests.exceptions.HTTPError(f'Failed to fetch game feed for {game_id} after {max_retries} attempts')

def get_shifts(game_id):
    """Fetch and return the shifts feed JSON for the requested game.

    This function is intentionally defensive and debug-friendly. It returns a
    dictionary containing:
      - 'game_id': the input game id
      - 'raw': the raw JSON returned by the NHL API (or None on error)
      - 'all_shifts': a flat list of shift dicts discovered in the payload
      - 'shifts_by_player': a dict mapping player_id -> list of parsed shift dicts

    The parser uses heuristic rules to extract common fields from varying
    JSON shapes. For each parsed shift we attempt to pull out:
      - player_id (int or string)
      - team_id (when available)
      - period (when available)
      - start_raw / end_raw: original values from the feed
      - start_seconds / end_seconds: best-effort numeric parse for MM:SS style values
      - raw: the original shift dict

    The function deliberately does not attempt to convert period-relative
    MM:SS "time remaining" values into absolute game seconds because the
    feed formats vary (some report time remaining, some report elapsed). The
    timing module will be responsible for consistent conversion when it has
    the full game context.
    """
    import logging
    from time import sleep

    url = ('https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId='
           + str(game_id))
    # simple retry loop to be resilient to transient network / rate errors
    max_retries = 4
    backoff = 1.0
    resp = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = SESSION.get(url, timeout=10)
            resp.raise_for_status()
            break
        except requests.exceptions.HTTPError as he:
            status = getattr(resp, 'status_code', None) if resp is not None else None
            if status == 429:
                # rate limited; wait then retry
                ra = None
                try:
                    ra = resp.headers.get('Retry-After')
                except Exception:
                    ra = None
                wait = backoff if ra is None else float(ra) if ra.isdigit() else backoff
                wait = min(300.0, wait) + random.uniform(0, 1.0)
                logging.warning('get_shifts: 429 for game %s, sleeping %.1fs (Retry-After=%s)', game_id, wait, ra)
                sleep(wait)
                backoff = min(300.0, backoff * 2)
                continue
            if status == 403:
                logging.warning('get_shifts: access denied (403) for game %s; returning empty structure', game_id)
                return {'game_id': game_id, 'raw': None, 'all_shifts': [], 'shifts_by_player': {}}
            # non-retryable; re-raise
            raise
        except requests.exceptions.RequestException as re:
            logging.warning('get_shifts network error for game %s: %s; attempt %d/%d', game_id, re, attempt, max_retries)
            sleep(backoff + random.uniform(0, 1.0))
            backoff = min(300.0, backoff * 2)
            continue

    if resp is None:
        return {'game_id': game_id, 'raw': None, 'all_shifts': [], 'shifts_by_player': {}}

    try:
        data = resp.json()
    except Exception as e:
        logging.warning('get_shifts: failed to parse JSON for game %s: %s', game_id, e)
        return {'game_id': game_id, 'raw': None, 'all_shifts': [], 'shifts_by_player': {}}

    # Heuristic: collect lists of candidate shift entries from the JSON payload
    candidates = []

    def _collect_lists(obj):
        """Recursively collect lists of dict-like items that look like shifts."""
        if isinstance(obj, list):
            # if list of dicts where dicts have 'player' or 'playerId' or 'personId', consider it
            if obj and all(isinstance(i, dict) for i in obj):
                sample = obj[0]
                keys = set(sample.keys())
                if keys & {'playerId', 'player_id', 'personId', 'player'} or keys & {'start', 'startTime', 'start_time', 'startTimeUTC'}:
                    candidates.append(obj)
            # still traverse list elements
            for el in obj:
                _collect_lists(el)
        elif isinstance(obj, dict):
            for v in obj.values():
                _collect_lists(v)

    _collect_lists(data)

    # If we didn't find any obvious lists, try a few common top-level keys
    if not candidates:
        for k in ('shifts', 'data', 'shiftReports', 'shiftReport'):
            if isinstance(data.get(k), list):
                candidates.append(data.get(k))
            elif isinstance(data.get(k), dict):
                # sometimes shifts are nested under team keys
                for v in data.get(k).values():
                    if isinstance(v, list):
                        candidates.append(v)

    all_shifts = []

    def _parse_time_to_seconds(s: Any) -> Optional[float]:
        """Best-effort parse of MM:SS or M:SS strings to seconds (returns None if unknown)."""
        if s is None:
            return None
        if isinstance(s, (int, float)):
            return float(s)
        if isinstance(s, str):
            s = s.strip()
            # match mm:ss or m:ss
            import re
            m = re.match(r'^(\d+):(\d{2})$', s)
            if m:
                try:
                    mm = int(m.group(1))
                    ss = int(m.group(2))
                    return float(mm * 60 + ss)
                except Exception:
                    return None
            # try ISO timestamp
            try:
                from datetime import datetime
                if 'T' in s:
                    dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
                    return dt.timestamp()
            except Exception:
                pass
        return None

    for lst in candidates:
        for entry in lst:
            if not isinstance(entry, dict):
                continue
            # try to extract player id
            pid = None
            for k in ('playerId', 'player_id', 'personId', 'playerIdRef'):
                if k in entry:
                    pid = entry.get(k)
                    break
            # sometimes nested 'player' object exists
            if pid is None and 'player' in entry and isinstance(entry.get('player'), dict):
                pid = entry.get('player').get('id') or entry.get('player').get('playerId')
            # team id
            tid = None
            for k in ('teamId', 'team_id', 'teamIdRef'):
                if k in entry:
                    tid = entry.get(k)
                    break
            if tid is None and 'team' in entry and isinstance(entry.get('team'), dict):
                tid = entry.get('team').get('id') or entry.get('team').get('teamId')

            period = entry.get('period') or entry.get('periodNumber') or entry.get('p')

            start_raw = entry.get('start') or entry.get('startTime') or entry.get('start_time') or entry.get('startTimeUTC')
            end_raw = entry.get('end') or entry.get('endTime') or entry.get('end_time') or entry.get('endTimeUTC')

            start_seconds = _parse_time_to_seconds(start_raw)
            end_seconds = _parse_time_to_seconds(end_raw)

            shift_parsed = {
                'game_id': game_id,
                'player_id': pid,
                'team_id': tid,
                'period': period,
                'start_raw': start_raw,
                'end_raw': end_raw,
                'start_seconds': start_seconds,
                'end_seconds': end_seconds,
                'raw': entry,
            }
            all_shifts.append(shift_parsed)

    # group by player
    shifts_by_player = {}
    for s in all_shifts:
        key = s.get('player_id') or 'unknown'
        shifts_by_player.setdefault(key, []).append(s)

    return {
        'game_id': game_id,
        'raw': data,
        'all_shifts': all_shifts,
        'shifts_by_player': shifts_by_player,
    }


if __name__ == "__main__":
    # keep original main behavior for quick command-line inspection
    # show the most-recent PHI game by default
    game_id = get_game_id(method='most_recent', team='PHI')

    # basic play-by-play feed summary
    feed = get_game_feed(game_id)
    print('GameID:', game_id)
    if isinstance(feed, dict):
        print('Feed keys:', list(feed.keys())[:10])

    # Debug: fetch and summarize shifts to help development of timing/shift logic
    print('\nDebug: fetching shifts for game', game_id)
    shifts_res = get_shifts(game_id)

    if not shifts_res or shifts_res.get('raw') is None:
        print('No shifts data available (possible 403 or parsing error).')
    else:
        total_shifts = len(shifts_res.get('all_shifts', []))
        n_players = len(shifts_res.get('shifts_by_player', {}))
        print(f'Parsed shifts: total={total_shifts}, players={n_players}')
        # Print a small sample (first 3) for quick inspection
        for i, s in enumerate(shifts_res.get('all_shifts', [])[:3]):
            print(f' SAMPLE {i+1}: player={s.get("player_id")}, team={s.get("team_id")}, period={s.get("period")}, start={s.get("start_raw")}, end={s.get("end_raw")}')

    # For deeper debugging you can inspect `shifts_res['raw']` or
    # `shifts_res['shifts_by_player']` programmatically

