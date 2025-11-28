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
import re
import unicodedata
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
import email.utils as email_utils
import os
import json
import hashlib
from collections import deque

# Pattern for removing punctuation in name normalization
_PUNCTUATION_PATTERN = re.compile(r"[,.'\"]")
_WHITESPACE_PATTERN = re.compile(r"\s+")

# Simple on-disk and in-memory cache to reduce repeated API calls
_CACHE_DIR = os.path.join('.cache', 'nhl_api')
os.makedirs(_CACHE_DIR, exist_ok=True)
_GAME_FEED_CACHE: Dict[str, Dict[str, Any]] = {}
_SHIFTS_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_TTL = 60 * 60  # default cache TTL: 1 hour
_CACHING_ENABLED = True

def set_caching(enabled: bool):
    """Enable or disable API caching globally."""
    global _CACHING_ENABLED
    _CACHING_ENABLED = enabled

# Basic throttling: ensure at least MIN_INTERVAL seconds between requests
_LAST_REQUEST_TIME = 0.0
_MIN_REQUEST_INTERVAL = 0.25  # seconds

# Simple API call accounting to avoid accidental excessive calls across the process.
# We implement a sliding window counter (timestamps deque) and a small default cap.
_API_CALL_TIMES = deque()  # timestamps (float seconds)
_API_CALL_WINDOW = 60.0  # window in seconds to count calls (default 1 minute)
_API_CALL_MAX = 60  # default max calls permitted in window


def _increment_api_call():
    """Record that an API call was just made (append timestamp and prune old entries)."""
    try:
        now = time.time()
        _API_CALL_TIMES.append(now)
        # prune older entries outside the window
        cutoff = now - _API_CALL_WINDOW
        while _API_CALL_TIMES and _API_CALL_TIMES[0] < cutoff:
            _API_CALL_TIMES.popleft()
    except Exception:
        pass


def _allow_api_call() -> bool:
    """Return True if a new API call is allowed under the current sliding-window cap.

    This inspects the timestamps deque, prunes entries older than the window, and
    returns True when the count is strictly less than _API_CALL_MAX.
    """
    try:
        now = time.time()
        cutoff = now - _API_CALL_WINDOW
        while _API_CALL_TIMES and _API_CALL_TIMES[0] < cutoff:
            _API_CALL_TIMES.popleft()
        return len(_API_CALL_TIMES) < int(_API_CALL_MAX)
    except Exception:
        return True


def _reset_api_counters():
    """Reset the internal API call counters (useful for tests or debug)."""
    try:
        _API_CALL_TIMES.clear()
    except Exception:
        pass


def _throttle():
    """Throttle requests to avoid hammering the API from tight loops."""
    global _LAST_REQUEST_TIME
    now = time.time()
    elapsed = now - _LAST_REQUEST_TIME
    if elapsed < _MIN_REQUEST_INTERVAL:
        time.sleep(_MIN_REQUEST_INTERVAL - elapsed + random.uniform(0, 0.05))
    _LAST_REQUEST_TIME = time.time()


def _cache_path(kind: str, key: str) -> str:
    safe = hashlib.sha1(str(key).encode('utf-8')).hexdigest()
    return os.path.join(_CACHE_DIR, f"{kind}_{safe}.json")


def _is_cacheable(kind: str, obj: Any) -> bool:
    """Return True if `obj` is worth caching for `kind`.

    Avoid caching error-like payloads (e.g. {'raw': None}, empty dicts, or
    lists with no useful content). Keep this conservative: if unsure, allow caching.
    """
    try:
        if obj is None:
            return False
        if kind == 'shifts':
            # expect a dict with either 'raw' containing JSON or 'all_shifts' non-empty
            if isinstance(obj, dict):
                raw = obj.get('raw') if 'raw' in obj else obj
                if raw is None:
                    # some callers set 'raw' to None on 403/denied
                    return False
                # if 'all_shifts' present and non-empty, cache
                all_shifts = obj.get('all_shifts')
                if isinstance(all_shifts, list) and len(all_shifts) == 0:
                    # empty shifts list likely not useful
                    return False
                return True
            # non-dict responses: be permissive
            return True
        elif kind == 'game_feed':
            # expect a dict with meaningful keys; empty dicts are not cacheable
            if isinstance(obj, dict):
                if not obj:
                    return False
                # some error responses are dicts with a single 'message' key; avoid caching
                if set(obj.keys()) <= {'message'}:
                    return False
                return True
        else:
            # default: cache
            return True
    except Exception:
        return True
    # defensive final return for static analyzers: ensure we always return a bool
    return True


def _cache_get(kind: str, key: str, ttl: int = None) -> Optional[Dict[str, Any]]:
    """Return cached JSON object if available and fresh, else None.

    Additionally, if the cached object looks like an error/no-data payload, remove
    the cache file and treat as a miss.
    """
    if not _CACHING_ENABLED:
        return None

    if ttl is None:
        ttl = _CACHE_TTL
    # check in-memory cache first
    if kind == 'game_feed':
        if key in _GAME_FEED_CACHE:
            cached = _GAME_FEED_CACHE[key]
            if _is_cacheable(kind, cached):
                return cached
            else:
                # purge in-memory and on-disk
                try:
                    _GAME_FEED_CACHE.pop(key, None)
                except Exception:
                    pass
                try:
                    os.remove(_cache_path(kind, key))
                except Exception:
                    pass
                return None
    elif kind == 'shifts':
        if key in _SHIFTS_CACHE:
            cached = _SHIFTS_CACHE[key]
            if _is_cacheable(kind, cached):
                return cached
            else:
                try:
                    _SHIFTS_CACHE.pop(key, None)
                except Exception:
                    pass
                try:
                    os.remove(_cache_path(kind, key))
                except Exception:
                    pass
                return None

    path = _cache_path(kind, key)
    try:
        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            if (time.time() - mtime) <= float(ttl):
                with open(path, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                # if the loaded data is not cacheable, remove file and return None
                if not _is_cacheable(kind, data):
                    try:
                        os.remove(path)
                    except Exception:
                        pass
                    return None
                # populate memory cache
                if kind == 'game_feed':
                    _GAME_FEED_CACHE[key] = data
                elif kind == 'shifts':
                    _SHIFTS_CACHE[key] = data
                return data
    except Exception:
        # any cache read error -> treat as cache miss
        return None
    return None


def _cache_put(kind: str, key: str, obj: Dict[str, Any]):
    """Save object to disk and memory cache. Ignore failures.

    Only persist objects that are likely useful on future calls (avoid caching
    permission-denied or empty payloads).
    """
    if not _CACHING_ENABLED:
        return
    try:
        if not _is_cacheable(kind, obj):
            # don't cache nothing/error responses
            return
        # serialize to JSON-friendly structure
        path = _cache_path(kind, key)
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(obj, fh)
        os.replace(tmp, path)
        if kind == 'game_feed':
            _GAME_FEED_CACHE[key] = obj
        elif kind == 'shifts':
            _SHIFTS_CACHE[key] = obj
    except Exception:
        pass


def purge_cached_game(game_id: Any):
    """Remove cached game_feed and shifts entries for a given game id.

    Accepts numeric ids or strings. Returns True if at least one file cleared.
    """
    key = str(game_id)
    removed = False
    try:
        # in-memory
        if key in _GAME_FEED_CACHE:
            _GAME_FEED_CACHE.pop(key, None)
            removed = True
        if key in _SHIFTS_CACHE:
            _SHIFTS_CACHE.pop(key, None)
            removed = True
        # on-disk
        for kind in ('game_feed', 'shifts'):
            p = _cache_path(kind, key)
            try:
                if os.path.exists(p):
                    os.remove(p)
                    removed = True
            except Exception:
                pass
    except Exception:
        pass
    return removed


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
                _throttle()
                resp = SESSION.get(url, timeout=3)
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

            # advance to the next week
            week_start_dt = week_start_dt + timedelta(days=7)

        return games

    TEAM_ABB = (team or 'PHI').upper()

    # The api-web endpoint for a club's schedule for a season
    url = (f"https://api-web.nhle.com/v1/club-schedule-season/"
           f"{TEAM_ABB}/{season}")

    # Attempt to use a cached response for schedules (helps reduce calls)
    cache_key = f'season_{TEAM_ABB}_{season}'
    cached = _cache_get('game_feed', cache_key)
    if cached is not None:
        try:
            games_raw = cached.get('games')
            if isinstance(games_raw, list):
                return [g for g in games_raw if isinstance(g, dict)]
        except Exception:
            pass

    _throttle()
    resp = SESSION.get(url, timeout=3)
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

    # cache season schedule response
    try:
        _cache_put('game_feed', cache_key, data)
    except Exception:
        pass

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


def get_game_feed(game_id: int, max_retries: int = 3, backoff_base: float =
1.0, max_backoff: float = 5.0, force_refresh: bool = False) -> Dict[str, Any]:
    """Fetch and return the play-by-play feed JSON for the requested game.

    The function retries on 429 responses honoring Retry-After headers when
    present, and performs exponential backoff on transient network errors.

    Parameters:
    - force_refresh: if True, bypass cache and fetch from API directly
    """
    url = f'https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play'

    # Attempt to use cached response
    if not force_refresh:
        cached = _cache_get('game_feed', str(game_id))
        if cached is not None:
            return cached

    backoff = backoff_base
    for attempt in range(1, max_retries + 1):
        try:
            _throttle()
            # increment API call counter for this live request
            try:
                _increment_api_call()
            except Exception:
                pass
            resp = SESSION.get(url, timeout=3)
            resp.raise_for_status()
            data = resp.json()

            # Update cache with fresh response
            _cache_put('game_feed', str(game_id), data)

            return data

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
                            # parsedate_to_datetime may return None; ensure it's a datetime
                            parsed_dt = parsed if isinstance(parsed, datetime) else None
                            if isinstance(parsed_dt, datetime):
                                if parsed_dt.tzinfo is None:
                                    parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
                                delta = (parsed_dt - datetime.now(timezone.utc)).total_seconds()
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


def get_shifts(game_id, force_refresh: bool = False):
    """Fetch and return the shifts feed JSON for the requested game.

    Hardened behavior notes:
    - Check cache and purge stale or clearly invalid entries (raw==None).
    - Retry multiple times with backoff on transient failures.
    - If the primary endpoint returns a payload without candidate shift lists, try an alternate api-web endpoint once.
    - Always return a dict with keys: 'game_id', 'raw', 'all_shifts', 'shifts_by_player'.
    - Avoid caching unhelpful payloads.
    """
    try:
        # Always fetch fresh data from the network; do not use or write cache.
        # force_refresh parameter is accepted for API compatibility but ignored.

        # Prepare endpoints to try (primary then fallback)
        primary_url = 'https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId=' + str(game_id)
        alt_url = f'https://api-web.nhle.com/v1/gamecenter/{game_id}/shiftcharts'

        urls_to_try = [primary_url]

        max_retries = 3
        backoff = 1.0
        data = None
        used_url = None

        for url in urls_to_try:
            for attempt in range(1, max_retries + 1):
                try:
                    _throttle()
                    try:
                        _increment_api_call()
                    except Exception:
                        pass
                    resp = SESSION.get(url, timeout=3)
                    status = getattr(resp, 'status_code', None)
                    if status == 403:
                        logging.warning('get_shifts: access denied (403) for game %s at %s', game_id, url)
                        # don't cache 403; move to next url or return
                        break
                    if status == 429:
                        # rate limited; parse Retry-After if present
                        ra = None
                        try:
                            ra = resp.headers.get('Retry-After')
                        except Exception:
                            ra = None
                        wait = backoff
                        if ra:
                            try:
                                wait = float(ra)
                            except Exception:
                                try:
                                    parsed = email_utils.parsedate_to_datetime(ra)
                                    # parsedate_to_datetime may return None; ensure it's a datetime
                                    parsed_dt = parsed if isinstance(parsed, datetime) else None
                                    if isinstance(parsed_dt, datetime):
                                        if parsed_dt.tzinfo is None:
                                            parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
                                        delta = (parsed_dt - datetime.now(timezone.utc)).total_seconds()
                                        wait = max(wait, max(0.0, delta))
                                except Exception:
                                    pass
                        sleep_time = min(300.0, wait) + random.uniform(0, 1.0)
                        logging.warning('get_shifts: 429 for %s at %s attempt %d/%d; sleeping %.1fs', game_id, url, attempt, max_retries, sleep_time)
                        time.sleep(sleep_time)
                        backoff = min(2.0, backoff * 2)
                        continue

                    # if non-200 but not handled above, raise for usual handling
                    resp.raise_for_status()

                    # try parse JSON; if decode fails, retry
                    try:
                        j = resp.json()
                    except Exception as e:
                        logging.warning('get_shifts: JSON parse failed for %s at %s: %s', game_id, url, e)
                        time.sleep(backoff + random.uniform(0, 0.5))
                        backoff = min(2.0, backoff * 2)
                        continue

                    # if JSON contains no candidate shift lists, we may retry (transient different shape)
                    # quick check for likely presence: look for lists/dicts containing 'player' or 'start'
                    def _has_candidate_shifts(obj):
                        if obj is None:
                            return False
                        if isinstance(obj, dict):
                            # common top-level keys
                            for k in ('data', 'shifts', 'shiftReports', 'rows'):
                                if isinstance(obj.get(k), list) and len(obj.get(k)) > 0:
                                    return True
                            # sometimes payload includes nested structures
                            for v in obj.values():
                                if isinstance(v, list) and v and isinstance(v[0], dict):
                                    keys = set(v[0].keys())
                                    if keys & {'playerId', 'player_id', 'personId', 'start', 'startTime'}:
                                        return True
                        elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
                            keys = set(obj[0].keys())
                            if keys & {'playerId', 'player_id', 'personId', 'start', 'startTime'}:
                                return True
                        return False

                    if not _has_candidate_shifts(j):
                        # bad/empty shape -> retry up to attempts, then try next URL
                        logging.info('get_shifts: response for %s at %s contains no candidate shifts; attempt %d/%d', game_id, url, attempt, max_retries)
                        time.sleep(backoff + random.uniform(0, 0.5))
                        backoff = min(2.0, backoff * 2)
                        continue

                    data = j
                    used_url = url
                    break

                except requests.exceptions.RequestException as re:
                    logging.warning('get_shifts: network error for %s at %s: %s (attempt %d/%d)', game_id, url, re, attempt, max_retries)
                    time.sleep(backoff + random.uniform(0, 0.5))
                    backoff = min(2.0, backoff * 2)
                    continue
                except Exception as e:
                    logging.exception('get_shifts: unexpected error for %s at %s: %s', game_id, url, e)
                    time.sleep(backoff + random.uniform(0, 0.5))
                    backoff = min(2.0, backoff * 2)
                    continue

            if data is not None:
                break

        if data is None:
            logging.warning('get_shifts: failed to retrieve usable shifts payload for %s from all endpoints', game_id)
            # Try NHL HTML report fallback before giving up
            try:
                backup = get_shifts_from_nhl_html(game_id, force_refresh=force_refresh, debug=True)
                if backup and isinstance(backup, dict) and backup.get('all_shifts'):
                    logging.info('get_shifts: obtained shifts for %s from NHL HTML fallback', game_id)
                    return backup
            except Exception:
                logging.exception('get_shifts: NHL HTML fallback failed for %s', game_id)
            return {'game_id': game_id, 'raw': None, 'all_shifts': [], 'shifts_by_player': {}}

        # Heuristic: collect lists of candidate shift entries from the JSON payload
        candidates = []
        def _collect_lists(obj):
            if isinstance(obj, list):
                if obj and all(isinstance(i, dict) for i in obj):
                    sample = obj[0]
                    keys = set(sample.keys())
                    if keys & {'playerId', 'player_id', 'personId', 'player'} or keys & {'start', 'startTime', 'start_time', 'startTimeUTC'}:
                        candidates.append(obj)
                for el in obj:
                    _collect_lists(el)
            elif isinstance(obj, dict):
                for v in obj.values():
                    _collect_lists(v)
        try:
            _collect_lists(data)
        except Exception:
            pass

        # If no candidates found, still attempt to probe a few top-level keys
        if not candidates:
            for k in ('shifts', 'data', 'shiftReports', 'shiftReport', 'rows'):
                try:
                    val = data.get(k)
                    if isinstance(val, list):
                        candidates.append(val)
                except Exception:
                    continue

        all_shifts = []
        def _parse_time_to_seconds(s: Any) -> Optional[float]:
            if s is None:
                return None
            if isinstance(s, (int, float)):
                return float(s)
            if isinstance(s, str):
                s = s.strip()
                import re
                m = re.match(r'^(\d+):(\d{2})$', s)
                if m:
                    try:
                        mm = int(m.group(1)); ss = int(m.group(2)); return float(mm*60+ss)
                    except Exception:
                        return None
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
                pid = None
                for k in ('playerId', 'player_id', 'personId', 'playerIdRef'):
                    if k in entry:
                        pid = entry.get(k); break
                if pid is None and 'player' in entry and isinstance(entry.get('player'), dict):
                    pid = entry.get('player').get('id') or entry.get('player').get('playerId')
                tid = None
                for k in ('teamId', 'team_id', 'teamIdRef'):
                    if k in entry:
                        tid = entry.get(k); break
                if tid is None and 'team' in entry and isinstance(entry.get('team'), dict):
                    tid = entry.get('team').get('id') or entry.get('team').get('teamId')
                period = entry.get('period') or entry.get('periodNumber') or entry.get('p')
                start_raw = entry.get('start') or entry.get('startTime') or entry.get('start_time') or entry.get('startTimeUTC')
                end_raw = entry.get('end') or entry.get('endTime') or entry.get('end_time') or entry.get('endTimeUTC')
                start_seconds = _parse_time_to_seconds(start_raw)
                end_seconds = _parse_time_to_seconds(end_raw)
                shift_parsed = {'game_id': game_id, 'player_id': pid, 'team_id': tid, 'period': period, 'start_raw': start_raw, 'end_raw': end_raw, 'start_seconds': start_seconds, 'end_seconds': end_seconds, 'raw': entry}
                all_shifts.append(shift_parsed)

        # Remove zero-length shifts where start and end seconds are identical
        try:
            before_len = len(all_shifts)
            filtered = []
            for s in all_shifts:
                ss = s.get('start_seconds')
                es = s.get('end_seconds')
                try:
                    if ss is not None and es is not None and float(ss) == float(es):
                        # drop this zero-length interval
                        continue
                except Exception:
                    pass
                filtered.append(s)
            removed_cnt = before_len - len(filtered)
            if removed_cnt:
                logging.info('get_shifts: removed %d zero-length shifts for game %s', removed_cnt, game_id)
            all_shifts = filtered
        except Exception:
            pass

        shifts_by_player = {}
        for s in all_shifts:
            keyp = s.get('player_id') or 'unknown'
            shifts_by_player.setdefault(keyp, []).append(s)

        result = {'game_id': game_id, 'raw': data, 'all_shifts': all_shifts, 'shifts_by_player': shifts_by_player}

        # Do not write to cache; always return fresh result
        return result

    except Exception as e:
        logging.exception('get_shifts: unexpected error for game %s: %s', game_id, e)
        return {'game_id': game_id, 'raw': None, 'all_shifts': [], 'shifts_by_player': {}}


def _get_roster_mapping(game_id: Any) -> Dict[str, Dict[int, int]]:
    """Extract jersey number -> player_id mapping from game feed.
    
    Returns a dict with 'home' and 'away' keys, each containing {jersey_number: player_id}.
    This allows team-specific mapping since jersey numbers can be reused across teams.
    
    Example return: {'home': {12: 8471675, 23: 8474564}, 'away': {12: 8475798, ...}}
    """
    try:
        feed = get_game_feed(game_id)
        if not feed or not isinstance(feed, dict):
            return {'home': {}, 'away': {}}
        
        # First, extract home and away team IDs so we can map teamId to 'home'/'away'
        home_id = None
        away_id = None
        if 'homeTeam' in feed and isinstance(feed['homeTeam'], dict):
            home_id = feed['homeTeam'].get('id')
        if 'awayTeam' in feed and isinstance(feed['awayTeam'], dict):
            away_id = feed['awayTeam'].get('id')
        
        # Convert to int if they're strings
        if home_id is not None:
            try:
                home_id = int(home_id)
            except (ValueError, TypeError):
                pass
        if away_id is not None:
            try:
                away_id = int(away_id)
            except (ValueError, TypeError):
                pass
        
        # Build mapping by walking the feed structure
        home_map = {}
        away_map = {}
        
        def walk_and_extract(obj, current_team=None):
            """Recursively walk the feed and extract player info."""
            if isinstance(obj, dict):
                # Check if this dict contains team info
                team = current_team
                
                # If we have a teamId field, use it to determine team (highest priority)
                if 'teamId' in obj:
                    try:
                        tid = int(obj.get('teamId'))
                        if tid == home_id:
                            team = 'home'
                        elif tid == away_id:
                            team = 'away'
                    except (ValueError, TypeError):
                        pass
                
                # Otherwise, detect team context from common keys
                if team is None or team not in ('home', 'away'):
                    if 'homeTeam' in obj or ('team' in obj and obj.get('team') == 'home'):
                        team = 'home'
                    elif 'awayTeam' in obj or ('team' in obj and obj.get('team') == 'away'):
                        team = 'away'
                
                # Try to extract player info from this dict
                pid = None
                num = None
                
                # Look for player ID in various common structures
                # {'person': {'id': ...}, 'jerseyNumber': '12'}
                if 'person' in obj and isinstance(obj.get('person'), dict):
                    p = obj.get('person')
                    pid = p.get('id') or p.get('personId') or p.get('playerId')
                # {'player': {'id': ...}, 'number': '12'}
                elif 'player' in obj and isinstance(obj.get('player'), dict):
                    p = obj.get('player')
                    pid = p.get('id') or p.get('playerId') or p.get('personId')
                # Direct ID field
                else:
                    for k in ('playerId', 'personId', 'id'):
                        if pid is None and k in obj and isinstance(obj.get(k), (int, str)):
                            try:
                                pid = int(obj.get(k))
                                break
                            except Exception:
                                pass
                
                # Look for jersey number
                for k in ('sweaterNumber', 'jerseyNumber', 'jersey', 'number'):
                    if k in obj:
                        try:
                            num = int(str(obj.get(k)).strip())
                            break
                        except Exception:
                            pass
                
                # If we found both pid and number, add to appropriate team map
                if pid is not None and num is not None and team in ('home', 'away'):
                    target_map = home_map if team == 'home' else away_map
                    target_map[num] = int(pid)
                    logging.debug('_get_roster_mapping: adding %s team jersey %d -> player_id %d', team, num, pid)
                
                # Recurse into nested structures
                for key, value in obj.items():
                    # Propagate team context for known structure keys
                    next_team = team
                    if key in ('homeTeam', 'home'):
                        next_team = 'home'
                    elif key in ('awayTeam', 'away'):
                        next_team = 'away'
                    walk_and_extract(value, next_team)
                    
            elif isinstance(obj, list):
                for el in obj:
                    walk_and_extract(el, current_team)
        
        walk_and_extract(feed)
        
        return {'home': home_map, 'away': away_map}
        
    except Exception as e:
        logging.warning('_get_roster_mapping: failed for game %s: %s', game_id, e)
        return {'home': {}, 'away': {}}


def _get_team_ids(game_id: Any) -> Dict[str, Optional[int]]:
    """Extract team IDs for home and away teams from game feed.
    
    Returns a dict with 'home' and 'away' keys mapping to team IDs.
    Example return: {'home': 1, 'away': 6}
    """
    try:
        feed = get_game_feed(game_id)
        if not feed or not isinstance(feed, dict):
            return {'home': None, 'away': None}
        
        home_id = None
        away_id = None
        
        # Try to extract from common structures
        if 'homeTeam' in feed and isinstance(feed['homeTeam'], dict):
            home_id = feed['homeTeam'].get('id')
        if 'awayTeam' in feed and isinstance(feed['awayTeam'], dict):
            away_id = feed['awayTeam'].get('id')
        
        # Fallback: search for teams in boxscore, linescore, or plays structures
        if home_id is None or away_id is None:
            for key in ['boxscore', 'linescore', 'plays']:
                if key in feed and isinstance(feed[key], dict):
                    obj = feed[key]
                    if home_id is None and 'home' in obj and isinstance(obj['home'], dict):
                        home_id = obj['home'].get('team', {}).get('id')
                    if away_id is None and 'away' in obj and isinstance(obj['away'], dict):
                        away_id = obj['away'].get('team', {}).get('id')
        
        # Convert to int if they're strings
        if home_id is not None:
            try:
                home_id = int(home_id)
            except (ValueError, TypeError):
                pass
        if away_id is not None:
            try:
                away_id = int(away_id)
            except (ValueError, TypeError):
                pass
        
        return {'home': home_id, 'away': away_id}
        
    except Exception as e:
        logging.warning('_get_team_ids: failed for game %s: %s', game_id, e)
        return {'home': None, 'away': None}


def _normalize_name(n: str) -> str:
    """Normalize a player name for matching purposes.
    
    Removes punctuation, extra whitespace, converts to lowercase, and handles accented characters.
    """
    if not n:
        return ''
    s = str(n).lower().strip()
    
    # Normalize Unicode characters (decompose accented chars)
    # NFD decomposes characters like 'ü' into 'u' + combining diaeresis
    try:
        s = unicodedata.normalize('NFD', s)
        # Remove combining characters (accents, diacritics)
        s = ''.join(c for c in s if not unicodedata.combining(c))
    except Exception:
        pass
    
    # Remove common punctuation and special characters
    s = _PUNCTUATION_PATTERN.sub('', s)
    # Collapse multiple spaces to single space
    s = _WHITESPACE_PATTERN.sub(' ', s)
    return s


def _build_name_to_id_map(game_id: Any) -> Dict[str, int]:
    """Build a mapping from normalized player name to player_id from game feed.
    
    Returns a dict mapping normalized names to player IDs.
    Example: {'john smith': 8471675, ...}
    """
    try:
        feed = get_game_feed(game_id)
        if not feed or not isinstance(feed, dict):
            return {}
        
        name_map = {}
        
        def walk_and_extract_names(obj, depth=0):
            """Recursively walk the feed and extract player names and IDs.
            
            Args:
                obj: The object to walk (dict, list, or other)
                depth: Current recursion depth (for limiting traversal)
            """
            # Limit recursion depth to prevent excessive traversal
            if depth > 20:
                return
            
            if isinstance(obj, dict):
                # Try to extract player info from this dict
                pid = None
                name = None
                
                # Look for player ID
                if 'person' in obj and isinstance(obj.get('person'), dict):
                    p = obj.get('person')
                    pid = p.get('id') or p.get('personId') or p.get('playerId')
                    # Try to get full name
                    fname = p.get('firstName') or p.get('first_name')
                    lname = p.get('lastName') or p.get('last_name')
                    if fname and lname:
                        name = f"{fname} {lname}"
                    elif 'fullName' in p:
                        name = p.get('fullName')
                    elif 'name' in p:
                        name = p.get('name')
                elif 'player' in obj and isinstance(obj.get('player'), dict):
                    p = obj.get('player')
                    pid = p.get('id') or p.get('playerId') or p.get('personId')
                    fname = p.get('firstName') or p.get('first_name')
                    lname = p.get('lastName') or p.get('last_name')
                    if fname and lname:
                        name = f"{fname} {lname}"
                    elif 'fullName' in p:
                        name = p.get('fullName')
                    elif 'name' in p:
                        name = p.get('name')
                else:
                    # Direct fields
                    for k in ('playerId', 'personId', 'id'):
                        if pid is None and k in obj:
                            try:
                                pid = int(obj.get(k))
                                break
                            except Exception:
                                pass
                    # Try to get name
                    if 'fullName' in obj:
                        name = obj.get('fullName')
                    elif 'name' in obj and isinstance(obj.get('name'), str):
                        name = obj.get('name')
                    elif 'firstName' in obj and 'lastName' in obj:
                        fname = obj.get('firstName')
                        lname = obj.get('lastName')
                        if fname and lname:
                            name = f"{fname} {lname}"
                
                # If we found both pid and name, add to map
                if pid is not None and name:
                    try:
                        normalized = _normalize_name(name)
                        if normalized:
                            name_map[normalized] = int(pid)
                            logging.debug('_build_name_to_id_map: adding %s -> %d', normalized, pid)
                    except Exception:
                        pass
                
                # Recurse into nested structures
                for value in obj.values():
                    walk_and_extract_names(value, depth + 1)
                    
            elif isinstance(obj, list):
                for el in obj:
                    walk_and_extract_names(el, depth + 1)
        
        walk_and_extract_names(feed)
        
        return name_map
        
    except Exception as e:
        logging.warning('_build_name_to_id_map: failed for game %s: %s', game_id, e)
        return {}


def get_shifts_from_nhl_html(game_id: Any, force_refresh: bool = False, debug: bool = False) -> Dict[str, Any]:
    """Fallback: obtain shift information by scraping NHL official HTML reports.

    This implementation parses per-player shift detail tables and, when those are
    not available, falls back to parsing event-style tables that list "On Ice"
    jersey numbers for Home and Away. The output mirrors `get_shifts()`:
    {'game_id', 'raw', 'all_shifts', 'shifts_by_player'} and may include a
    'debug' dict when debug=True.
    """
    key = str(game_id)
    if not force_refresh:
        cached = _cache_get('shifts_html', key)
        if cached:
            return cached

    try:
        gid = str(game_id)
        if len(gid) >= 10:
            year = int(gid[:4])
            season = f"{year}{year+1}"
        else:
            season = '20252026'

        report_suffix = gid[4:] if len(gid) > 4 else gid
        venue_texts = []
        tried_urls = []
        venue_tags = [('H', 'home'), ('V', 'away')]
        for venue_tag, _ in venue_tags:
            venue_url = f"https://www.nhl.com/scores/htmlreports/{season}/T{venue_tag}{report_suffix}.HTM"
            tried_urls.append(venue_url)
            try:
                _throttle()
                resp = SESSION.get(venue_url, timeout=12)
                resp.raise_for_status()
                venue_texts.append((venue_tag, resp.text))
                continue
            except Exception:
                try:
                    if venue_url.startswith('https://'):
                        alt = 'http://' + venue_url.split('://', 1)[1]
                        _throttle()
                        resp = SESSION.get(alt, timeout=12)
                        resp.raise_for_status()
                        venue_texts.append((venue_tag, resp.text))
                        continue
                except Exception:
                    continue

        if not venue_texts:
            if debug:
                return {'game_id': game_id, 'raw': None, 'all_shifts': [], 'shifts_by_player': {}, 'debug': {'error': 'no venue pages fetched', 'tried_urls': tried_urls}}
            return {'game_id': game_id, 'raw': None, 'all_shifts': [], 'shifts_by_player': {}}

        from bs4 import BeautifulSoup
        import re

        def clean_cell_text(el):
            if el is None:
                return ''
            return re.sub(r"\s+", ' ', el.get_text(' ', strip=True)).strip()

        def mmss_to_seconds(mmss: str) -> Optional[int]:
            if mmss is None:
                return None
            m = re.search(r"(\d{1,2}:\d{2})", str(mmss))
            if not m:
                return None
            try:
                mm, ss = m.group(1).split(':')
                return int(mm) * 60 + int(ss)
            except Exception:
                return None

        def period_time_to_total(period_raw, time_str) -> Optional[int]:
            try:
                per = None
                if period_raw is None:
                    return None
                pr = str(period_raw).strip().upper()
                if pr == 'OT':
                    per = 4
                else:
                    m = re.search(r"(\d+)", pr)
                    if m:
                        per = int(m.group(1))
                if per is None:
                    return None
                per_secs = mmss_to_seconds(time_str)
                if per_secs is None:
                    return None
                if per <= 3:
                    return (per - 1) * 1200 + per_secs
                else:
                    # OT and beyond: treat OT as 5-minute
                    return 3 * 1200 + (per - 4) * 300 + per_secs
            except Exception:
                return None

        # helper to decide whether a heading looks like a player heading
        def looks_like_player_heading(s: str) -> bool:
            if not s:
                return False
            s = s.strip()
            low = s.lower()
            # filter out summary header lines that contain 'shf', 'avg', 'toi' etc
            if any(x in low for x in ('shf', 'avg', 'toi', 'ev tot', 'pp tot', 'sh tot', 'per shf', 'avg toi')):
                return False
            # avoid lines that look like mm:ss tables
            if re.search(r"\d{1,2}:\d{2}", s):
                return False
            if not re.search(r"[A-Za-z]", s):
                return False
            if len(s) > 150:
                return False
            return True

        player_heading_patterns = [
            re.compile(r"^\s*(\d{1,3})\s+([^,\n\r]+),\s*(.+)$"),  # '12 LAST, FIRST'
            re.compile(r"^\s*([^,\n\r]+),\s*(.+)\s+(\d{1,3})$"),   # 'LAST, FIRST 12'
            re.compile(r"^\s*(\d{1,3})\s+([A-Za-z\-\'\.\s]{2,40})$"),                    # '12 NAME'
        ]

        all_shifts = []
        shifts_by_player = {}
        raw_combined = []
        tables_scanned = 0
        players_scanned = 0

        # First pass: try to find per-player detail tables
        for venue_tag, text in venue_texts:
            raw_combined.append(text or '')
            soup = None
            for parser in ('lxml', 'html.parser', 'html5lib'):
                try:
                    soup = BeautifulSoup(text, parser)
                    if soup is not None:
                        break
                except Exception:
                    soup = None
                    continue
            if soup is None:
                continue

            for tbl in soup.find_all('table'):
                rows = tbl.find_all('tr')
                if not rows:
                    continue

                # look for header row within first few rows
                header_row = None
                header_cells = []
                for i, r in enumerate(rows[:8]):
                    txts = [clean_cell_text(c).lower() for c in r.find_all(['td', 'th'])]
                    if not txts:
                        continue
                    joined = ' '.join(txts)
                    # header likely contains '#' or 'shift' and a 'start' or 'duration' or 'end'
                    if (any('shift' in t or t.strip().startswith('#') or t.strip() == '#' for t in txts)
                            and any(k in joined for k in ('start', 'end', 'duration', 'time', 'elapsed', 'per'))):
                        header_row = i
                        header_cells = txts
                        break
                if header_row is None:
                    continue

                tables_scanned += 1

                # skip per-player summary tables which tend to include 'shf' and 'avg' or 'toi' in headers
                hj = ' '.join(header_cells).replace('\u00a0', ' ')
                hj_norm = re.sub(r"\s+", ' ', hj.lower())
                # stronger summary detection: if header contains metrics like SHF, AVG, TOI, or TOT columns, skip
                if (('shf' in hj_norm and 'avg' in hj_norm) or ('ev tot' in hj_norm) or ('pp tot' in hj_norm) or ('sh tot' in hj_norm) or ('toi' in hj_norm and 'avg' in hj_norm) or any(x in hj_norm for x in ('shf', 'avg', 'toi')) and ('tot' in hj_norm or 'pp' in hj_norm)):
                    # informational skip
                    continue

                # build index map
                idx_map = {}
                for i, h in enumerate(header_cells):
                    if 'per' in h and 'ev' not in h:
                        idx_map['period'] = i
                    elif 'start' in h and 'elapsed' not in h:
                        idx_map['start'] = i
                    elif 'end' in h:
                        idx_map['end'] = i
                    elif 'duration' in h:
                        idx_map['duration'] = i
                    elif 'shift' in h or h.strip().startswith('#') or '#' in h:
                        idx_map['#'] = i
                    elif 'elapsed' in h:
                        idx_map.setdefault('start', i)
                    elif 'time' in h and 'on ice' not in h:
                        # sometimes header uses 'Time' rather than 'Start'
                        idx_map.setdefault('start', i)

                if not any(k in idx_map for k in ('#', 'start', 'end', 'duration')):
                    continue

                # infer player heading by looking at text rows immediately before the table
                pnum = None
                player_name = None
                heading_text = None
                # check rows inside table before header (sometimes player heading is a prior row)
                for r in rows[:header_row]:
                    cs = [clean_cell_text(c) for c in r.find_all(['td', 'th'])]
                    for s in cs:
                        ss = re.sub(r"\s+", ' ', s)
                        if not looks_like_player_heading(ss):
                            continue
                        for pat in player_heading_patterns:
                            m = pat.match(ss)
                            if not m:
                                continue
                            groups = m.groups()
                            if len(groups) == 3:
                                g1, g2, g3 = groups[0].strip(), groups[1].strip(), groups[2].strip()
                                if g1.isdigit():
                                    pnum = int(g1); last = g2; first = g3; player_name = (first + ' ' + last).strip()
                                elif g3.isdigit():
                                    pnum = int(g3); last = g1; first = g2; player_name = (first + ' ' + last).strip()
                            elif len(groups) == 2:
                                g1, g2 = groups[0].strip(), groups[1].strip()
                                if g1.isdigit():
                                    pnum = int(g1); player_name = g2
                                else:
                                    player_name = (g1 + ' ' + g2).strip()
                            heading_text = ss
                            break
                    if pnum is not None or player_name is not None:
                        break

                # as fallback, look for preceding siblings/text near the table
                if pnum is None and player_name is None:
                    walker = tbl.previous_elements
                    collected = []
                    for j, el in enumerate(walker):
                        if j > 160:
                            break
                        txt = None
                        if getattr(el, 'string', None):
                            txt = str(el.string).strip()
                        elif isinstance(el, str):
                            txt = el.strip()
                        if not txt:
                            continue
                        if len(txt) > 200:
                            continue
                        low = txt.lower()
                        if any(x in low for x in ('shf', 'avg', 'toi', 'ev tot', 'pp tot', 'sh tot')):
                            continue
                        collected.append(txt)
                        if re.search(r"\d{1,3}", txt) and re.search(r"[A-Za-z]", txt):
                            break
                    for s in collected:
                        ss = re.sub(r"\s+", ' ', s)
                        if not looks_like_player_heading(ss):
                            continue
                        for pat in player_heading_patterns:
                            m = pat.match(ss)
                            if not m:
                                continue
                            groups = m.groups()
                            if len(groups) == 3:
                                g1, g2, g3 = groups[0].strip(), groups[1].strip(), groups[2].strip()
                                if g1.isdigit():
                                    pnum = int(g1); last = g2; first = g3; player_name = (first + ' ' + last).strip()
                                elif g3.isdigit():
                                    pnum = int(g3); last = g1; first = g2; player_name = (first + ' ' + last).strip()
                            elif len(groups) == 2:
                                g1, g2 = groups[0].strip(), groups[1].strip()
                                if g1.isdigit():
                                    pnum = int(g1); player_name = g2
                                else:
                                    player_name = (g1 + ' ' + g2).strip()
                            heading_text = ss
                            break
                        if pnum is not None or player_name is not None:
                            break

                # still nothing -> attempt immediate previous tags like <b> or <strong>
                if pnum is None and player_name is None:
                    prev_tag = tbl.find_previous(['b', 'strong', 'font', 'p', 'div', 'caption', 'pre'])
                    if prev_tag is not None:
                        s = clean_cell_text(prev_tag)
                        if looks_like_player_heading(s):
                            for pat in player_heading_patterns:
                                m = pat.match(s)
                                if not m:
                                    continue
                                groups = m.groups()
                                if len(groups) == 3:
                                    g1, g2, g3 = groups[0].strip(), groups[1].strip(), groups[2].strip()
                                    if g1.isdigit():
                                        pnum = int(g1); last = g2; first = g3; player_name = (first + ' ' + last).strip()
                                    elif g3.isdigit():
                                        pnum = int(g3); last = g1; first = g2; player_name = (first + ' ' + last).strip()
                                elif len(groups) == 2:
                                    g1, g2 = groups[0].strip(), groups[1].strip()
                                    if g1.isdigit():
                                        pnum = int(g1); player_name = g2
                                    else:
                                        player_name = (g1 + ' ' + g2).strip()
                                heading_text = s
                                break

                # If still no identity, skip this table - likely a summary or unrelated table
                if pnum is None and player_name is None:
                    continue

                players_scanned += 1

                # parse rows after header_row for numbered shifts
                # Improved parsing: iterate rows sequentially to handle tables that include
                # multiple players and repeated player headings inside the same table.
                # Seed the current player from the previously-inferred pnum/player_name so
                # the first player's shift rows (which may immediately follow the header)
                # are attributed correctly.
                current_pnum = pnum
                current_player_name = player_name
                current_heading_text = heading_text
                current_idx_map = None
                for r in rows:
                    cells = [clean_cell_text(c) for c in r.find_all(['td', 'th'])]
                    if not cells:
                        continue

                    joined = ' '.join([c for c in cells if c])
                    lowjoined = joined.lower()

                    # detect summary rows and skip
                    if re.search(r"per\s+shf|avg|toi|ev tot|pp tot|sh tot", lowjoined):
                        # reset header mapping when encountering summaries
                        current_idx_map = None
                        continue

                    # detect player heading within the table
                    heading_found = False
                    for s in cells:
                        ss = re.sub(r"\s+", ' ', s).strip()
                        if not looks_like_player_heading(ss):
                            continue
                        for pat in player_heading_patterns:
                            m = pat.match(ss)
                            if not m:
                                continue
                            groups = m.groups()
                            pnum = None
                            pname = None
                            if len(groups) == 3:
                                g1, g2, g3 = groups[0].strip(), groups[1].strip(), groups[2].strip()
                                if g1.isdigit():
                                    pnum = int(g1); last = g2; first = g3; pname = (first + ' ' + last).strip()
                                elif g3.isdigit():
                                    pnum = int(g3); last = g1; first = g2; pname = (first + ' ' + last).strip()
                            elif len(groups) == 2:
                                g1, g2 = groups[0].strip(), groups[1].strip()
                                if g1.isdigit():
                                    pnum = int(g1); pname = g2
                                else:
                                    pname = (g1 + ' ' + g2).strip()
                            if pnum is not None or pname is not None:
                                current_pnum = pnum
                                current_player_name = pname
                                current_heading_text = ss
                                heading_found = True
                                break
                    if heading_found:
                        # a heading resets header mapping; header likely follows
                        current_idx_map = None
                        continue

                    # detect header row inside the table
                    # header likely contains '#' or 'shift' and some time columns
                    is_header = False
                    hdr_cells = [c.lower() for c in cells]
                    if (any('shift' in t or t.strip().startswith('#') or t.strip() == '#' for t in hdr_cells)
                            and any(k in ' '.join(hdr_cells) for k in ('start', 'end', 'duration', 'time', 'elapsed', 'per'))):
                        # build index map
                        m_idx = {}
                        for i, h in enumerate(hdr_cells):
                            if 'per' in h and 'ev' not in h:
                                m_idx['period'] = i
                            elif 'start' in h and 'elapsed' not in h:
                                m_idx['start'] = i
                            elif 'end' in h:
                                m_idx['end'] = i
                            elif 'duration' in h:
                                m_idx['duration'] = i
                            elif 'shift' in h or h.strip().startswith('#') or '#' in h:
                                m_idx['#'] = i
                            elif 'elapsed' in h:
                                m_idx.setdefault('start', i)
                            elif 'time' in h and 'on ice' not in h:
                                m_idx.setdefault('start', i)
                        # require at least one of the core indices
                        if any(k in m_idx for k in ('#', 'start', 'end', 'duration')):
                            current_idx_map = m_idx
                            continue

                    # if we have a header mapping and a current player, parse numeric shift rows
                    first = cells[0] if cells else ''
                    if current_idx_map and re.search(r"^\s*\d+\s*$", first):
                        # skip summary rows inside
                        lowjoin = ' '.join([c.lower() for c in cells])
                        if re.search(r"per\s+shf|avg|toi|ev tot|pp tot|sh tot", lowjoin):
                            continue

                        period_txt = cells[current_idx_map.get('period')] if current_idx_map.get('period') is not None and current_idx_map.get('period') < len(cells) else None
                        start_txt = cells[current_idx_map.get('start')] if current_idx_map.get('start') is not None and current_idx_map.get('start') < len(cells) else None
                        end_txt = cells[current_idx_map.get('end')] if current_idx_map.get('end') is not None and current_idx_map.get('end') < len(cells) else None

                        # Fallback mm:ss search
                        def find_first_mmss(seq):
                            for c in seq:
                                if not c:
                                    continue
                                m = re.search(r"(\d{1,2}:\d{2})", c)
                                if m:
                                    return m.group(1)
                            return None
                        def find_last_mmss(seq):
                            for c in reversed(seq):
                                if not c:
                                    continue
                                m = re.search(r"(\d{1,2}:\d{2})", c)
                                if m:
                                    return m.group(1)
                            return None

                        start_mmss = None
                        end_mmss = None
                        if start_txt:
                            m = re.search(r"(\d{1,2}:\d{2})", start_txt)
                            start_mmss = m.group(1) if m else None
                        if end_txt:
                            m = re.search(r"(\d{1,2}:\d{2})", end_txt)
                            end_mmss = m.group(1) if m else None

                        if start_mmss is None:
                            start_mmss = find_first_mmss(cells)
                        if end_mmss is None:
                            end_mmss = find_last_mmss(cells)

                        start_secs = mmss_to_seconds(start_mmss) if start_mmss else None
                        end_secs = mmss_to_seconds(end_mmss) if end_mmss else None

                        # if end missing, try duration
                        if end_secs is None and current_idx_map.get('duration') is not None and current_idx_map.get('duration') < len(cells):
                            dur_txt = cells[current_idx_map.get('duration')]
                            dur_secs = mmss_to_seconds(dur_txt)
                            if start_secs is not None and dur_secs is not None:
                                end_secs = start_secs + dur_secs
                                try:
                                    mm = int(end_secs) // 60
                                    ss = int(end_secs) % 60
                                    end_mmss = f"{mm:01d}:{ss:02d}"
                                except Exception:
                                    pass

                        # infer start when missing but end and duration present
                        if start_secs is None and end_secs is not None and current_idx_map.get('duration') is not None and current_idx_map.get('duration') < len(cells):
                            dur_txt = cells[current_idx_map.get('duration')]
                            dur_secs = mmss_to_seconds(dur_txt)
                            if dur_secs is not None:
                                inferred_start = end_secs - dur_secs
                                if inferred_start is not None and inferred_start >= 0:
                                    start_secs = inferred_start
                                    try:
                                        mm = int(start_secs) // 60
                                        ss = int(start_secs) % 60
                                        start_mmss = f"{mm:01d}:{ss:02d}"
                                    except Exception:
                                        start_mmss = None

                        per_val = None
                        try:
                            if period_txt is not None:
                                pt = str(period_txt).strip().upper()
                                if pt == 'OT':
                                    per_val = 4
                                else:
                                    mm = re.search(r"(\d+)", pt)
                                    if mm:
                                        per_val = int(mm.group(1))
                        except Exception:
                            per_val = None

                        total_start = period_time_to_total(per_val, start_mmss) if start_mmss and per_val is not None else None
                        total_end = period_time_to_total(per_val, end_mmss) if end_mmss and per_val is not None else None

                        # format numeric seconds into mm:ss when missing
                        try:
                            if start_secs is not None and (not start_mmss):
                                mm = int(start_secs) // 60
                                ss = int(start_secs) % 60
                                start_mmss = f"{mm:01d}:{ss:02d}"
                            if end_secs is not None and (not end_mmss):
                                mm = int(end_secs) // 60
                                ss = int(end_secs) % 60
                                end_mmss = f"{mm:01d}:{ss:02d}"
                        except Exception:
                            pass

                        shift = {
                            'game_id': game_id,
                            'player_id': current_pnum if current_pnum is not None else None,
                            'player_number': current_pnum,
                            'player_name': current_player_name,
                            'team_id': None,
                            'team_side': 'home' if venue_tag == 'H' else 'away',
                            'period': per_val,
                            'start_raw': start_mmss,
                            'end_raw': end_mmss,
                            'start_seconds': start_secs,
                            'end_seconds': end_secs,
                            'start_total_seconds': total_start,
                            'end_total_seconds': total_end,
                            'raw': {'player_heading': current_heading_text, 'row_cells': cells}
                        }
                        all_shifts.append(shift)
                        continue
                    # otherwise, nothing to do for this row
                    continue

        # If we didn't find any per-player shifts, fall back to event-style parsing
        if not all_shifts:
            for venue_tag, text in venue_texts:
                soup = None
                for parser in ('lxml', 'html.parser', 'html5lib'):
                    try:
                        soup = BeautifulSoup(text, parser)
                        if soup is not None:
                            break
                    except Exception:
                        soup = None
                        continue
                if soup is None:
                    continue

                # Find candidate tables that have both home and away on-ice columns
                for tbl in soup.find_all('table'):
                    rows = tbl.find_all('tr')
                    if not rows:
                        continue
                    header_row = None
                    header_cells = []
                    for i, r in enumerate(rows[:8]):
                        txts = [clean_cell_text(c).lower() for c in r.find_all(['td', 'th'])]
                        if not txts:
                            continue
                        joined = ' '.join(txts)
                        if (('on ice' in joined) or (('home' in joined and 'away' in joined and ('time' in joined or 'elapsed' in joined or 'event' in joined or 'str' in joined or 'strength' in joined)))):
                            header_row = i
                            header_cells = txts
                            break
                    if header_row is None:
                        continue

                    # map indices heuristically
                    idx_map = {}
                    for i, h in enumerate(header_cells):
                        if 'per' in h and 'ev' not in h:
                            idx_map['period'] = i
                        elif 'time' in h or 'elapsed' in h:
                            idx_map['time'] = i
                        elif 'home on' in h or ('home' in h and 'on' in h) or 'home on ice' in h:
                            idx_map['home_on'] = i
                        elif 'away on' in h or ('away' in h and 'on' in h) or 'away on ice' in h:
                            idx_map['away_on'] = i
                        elif 'description' in h or 'event' in h:
                            idx_map['desc'] = i

                    # if no on-ice columns, try to find home/away by presence of 'home'/'away' tokens
                    joined = ' '.join(header_cells)
                    if 'home' in joined and 'away' in joined and ('on' in joined or 'on ice' in joined):
                        for i, h in enumerate(header_cells):
                            if 'home' in h and 'on' in h:
                                idx_map.setdefault('home_on', i)
                            if 'away' in h and 'on' in h:
                                idx_map.setdefault('away_on', i)

                    if 'home_on' not in idx_map and 'away_on' not in idx_map:
                        continue

                    # collect event rows
                    events = []
                    for dr in rows[header_row + 1:]:
                        cells = [clean_cell_text(c) for c in dr.find_all(['td', 'th'])]
                        if not cells:
                            continue
                        # skip obvious summary lines
                        lowfirst = cells[0].lower() if cells else ''
                        if re.search(r"per\s+shf|avg|toi|ev tot|pp tot|sh tot", lowfirst):
                            continue

                        period_txt = cells[idx_map.get('period')] if idx_map.get('period') is not None and idx_map.get('period') < len(cells) else None
                        time_txt = cells[idx_map.get('time')] if idx_map.get('time') is not None and idx_map.get('time') < len(cells) else None
                        home_on_txt = cells[idx_map.get('home_on')] if idx_map.get('home_on') is not None and idx_map.get('home_on') < len(cells) else None
                        away_on_txt = cells[idx_map.get('away_on')] if idx_map.get('away_on') is not None and idx_map.get('away_on') < len(cells) else None

                        mmss = None
                        if time_txt:
                            m = re.search(r"(\d{1,2}:\d{2})", time_txt)
                            mmss = m.group(1) if m else None

                        def parse_nums(cell_text):
                            if not cell_text:
                                return []
                            nums = re.findall(r"\d{1,3}", cell_text)
                            out = []
                            for n in nums:
                                try:
                                    out.append(int(n))
                                except Exception:
                                    pass
                            return sorted(set(out))

                        home_on = parse_nums(home_on_txt)
                        away_on = parse_nums(away_on_txt)

                        per_val = None
                        try:
                            if period_txt:
                                pt = str(period_txt).strip().upper()
                                if pt == 'OT':
                                    per_val = 4
                                else:
                                    mm = re.search(r"(\d+)", pt)
                                    if mm:
                                        per_val = int(mm.group(1))
                        except Exception:
                            per_val = None

                        start_secs = mmss_to_seconds(mmss) if mmss else None
                        total_secs = period_time_to_total(per_val, mmss) if mmss and per_val is not None else None

                        events.append({'period': per_val, 'mmss': mmss, 'start_seconds': start_secs, 'total_seconds': total_secs, 'home_on': home_on, 'away_on': away_on, 'raw_cells': cells})

                    # sort events by total_seconds
                    events = [e for e in events if e.get('total_seconds') is not None]
                    events.sort(key=lambda e: e.get('total_seconds') or 0)
                    if not events:
                        continue

                    # infer shifts by tracking presence across events
                    all_nums = set()
                    for ev in events:
                        all_nums.update(ev.get('home_on', []))
                        all_nums.update(ev.get('away_on', []))

                    current_on = {}
                    per_player_shifts = {}
                    for ev in events:
                        t = ev.get('total_seconds')
                        per = ev.get('period')
                        for num in list(all_nums):
                            on_h = num in ev.get('home_on', [])
                            on_a = num in ev.get('away_on', [])
                            key_h = (num, 'home')
                            key_a = (num, 'away')
                            # home side
                            if on_h and key_h not in current_on:
                                current_on[key_h] = {'start_total': t, 'period': per, 'start_mmss': ev.get('mmss')}
                            if not on_h and key_h in current_on:
                                srec = current_on.pop(key_h)
                                if srec and srec.get('start_total') is not None and t is not None:
                                    per_player_shifts.setdefault(key_h, []).append({'start_total': srec.get('start_total'), 'end_total': t, 'period': srec.get('period'), 'start_mmss': srec.get('start_mmss'), 'end_mmss': ev.get('mmss')})
                            # away side
                            if on_a and key_a not in current_on:
                                current_on[key_a] = {'start_total': t, 'period': per, 'start_mmss': ev.get('mmss')}
                            if not on_a and key_a in current_on:
                                srec = current_on.pop(key_a)
                                if srec and srec.get('start_total') is not None and t is not None:
                                    per_player_shifts.setdefault(key_a, []).append({'start_total': srec.get('start_total'), 'end_total': t, 'period': srec.get('period'), 'start_mmss': srec.get('start_mmss'), 'end_mmss': ev.get('mmss')})

                    # close open shifts at last event
                    last = events[-1]
                    last_t = last.get('total_seconds')
                    last_mmss = last.get('mmss')
                    last_per = last.get('period')
                    for key, srec in list(current_on.items()):
                        if srec and srec.get('start_total') is not None:
                            per_player_shifts.setdefault(key, []).append({'start_total': srec.get('start_total'), 'end_total': last_t, 'period': srec.get('period') or last_per, 'start_mmss': srec.get('start_mmss'), 'end_mmss': last_mmss})

                    # convert to shifts
                    for (num, side), intervals in per_player_shifts.items():
                        for it in intervals:
                            per = it.get('period')
                            start_seconds = None
                            end_seconds = None
                            if it.get('start_total') is not None and per is not None:
                                if per <= 3:
                                    start_seconds = it.get('start_total') - ((per - 1) * 1200)
                                else:
                                    start_seconds = it.get('start_total') - (3 * 1200) - ((per - 4) * 300)
                            if it.get('end_total') is not None and per is not None:
                                if per <= 3:
                                    end_seconds = it.get('end_total') - ((per - 1) * 1200)
                                else:
                                    end_seconds = it.get('end_total') - (3 * 1200) - ((per - 4) * 300)

                            # ensure raw mm:ss strings exist when we have numeric seconds
                            s_raw = it.get('start_mmss')
                            e_raw = it.get('end_mmss')
                            try:
                                if start_seconds is not None and not s_raw:
                                    mm = int(start_seconds) // 60
                                    ss = int(start_seconds) % 60
                                    s_raw = f"{mm:01d}:{ss:02d}"
                                if end_seconds is not None and not e_raw:
                                    mm = int(end_seconds) // 60
                                    ss = int(end_seconds) % 60
                                    e_raw = f"{mm:01d}:{ss:02d}"
                            except Exception:
                                pass

                            shift = {
                                'game_id': game_id,
                                'player_id': num,
                                'player_number': num,
                                'player_name': None,
                                'team_id': None,
                                'team_side': side,
                                'period': per,
                                'start_raw': s_raw,
                                'end_raw': e_raw,
                                'start_seconds': start_seconds,
                                'end_seconds': end_seconds,
                                'start_total_seconds': it.get('start_total'),
                                'end_total_seconds': it.get('end_total'),
                                'raw': {'inferred_from_event_table': True}
                            }
                            all_shifts.append(shift)

                    if all_shifts:
                        break
                if all_shifts:
                    break

        # Map jersey numbers to canonical player_id values and set team_id
        roster_map = _get_roster_mapping(game_id)
        team_ids = _get_team_ids(game_id)
        name_map = _build_name_to_id_map(game_id)
        mapped_count = 0
        unmapped_count = 0
        unmapped_players = set()
        team_id_set_count = 0
        
        for shift in all_shifts:
            team_side = shift.get('team_side')
            player_number = shift.get('player_number')
            player_name = shift.get('player_name')
            
            # Set team_id based on team_side if available
            if team_side in ('home', 'away'):
                team_id = team_ids.get(team_side)
                if team_id is not None:
                    shift['team_id'] = team_id
                    team_id_set_count += 1
            
            # Try to map player_number to canonical player_id
            canonical_id = None
            
            # First: Try jersey number mapping for detected team_side
            if player_number is not None and team_side in ('home', 'away'):
                team_roster = roster_map.get(team_side, {})
                canonical_id = team_roster.get(player_number)
            
            # Second: Try name-based mapping if jersey mapping failed
            if canonical_id is None and player_name:
                normalized = _normalize_name(player_name)
                canonical_id = name_map.get(normalized)
            
            # Third: Try other team's roster (jersey number might be wrongly attributed to team_side)
            if canonical_id is None and player_number is not None:
                other_side = 'away' if team_side == 'home' else 'home'
                other_roster = roster_map.get(other_side, {})
                alt = other_roster.get(player_number)
                if alt is not None:
                    canonical_id = alt
                    # Correct the team_side and team_id
                    shift['team_side'] = other_side
                    other_team_id = team_ids.get(other_side)
                    if other_team_id is not None:
                        shift['team_id'] = other_team_id
            
            # If we found a canonical ID, set it
            if canonical_id is not None:
                shift['player_id'] = int(canonical_id)
                mapped_count += 1
                
                # If team_id is still not set, try to infer it from the canonical_id
                if not shift.get('team_id'):
                    # Check which team's roster contains this player_id
                    if canonical_id in roster_map.get('home', {}).values():
                        shift['team_id'] = team_ids.get('home')
                    elif canonical_id in roster_map.get('away', {}).values():
                        shift['team_id'] = team_ids.get('away')
            else:
                # Keep jersey number as player_id if mapping not found
                unmapped_count += 1
                unmapped_players.add((team_side, player_number, player_name))
        
        # Log mapping statistics for debugging
        if debug or unmapped_count > 0:
            total_roster_players = len(roster_map.get('home', {})) + len(roster_map.get('away', {}))
            home_players = len(roster_map.get('home', {}))
            away_players = len(roster_map.get('away', {}))
            
            logging.info(
                'get_shifts_from_nhl_html game %s: roster has %d players (home: %d, away: %d), '
                'name_map has %d entries, mapped %d/%d shifts, team_id set for %d/%d shifts', 
                game_id, total_roster_players, home_players, away_players, len(name_map),
                mapped_count, len(all_shifts), team_id_set_count, len(all_shifts)
            )
            if unmapped_count > 0:
                logging.warning('get_shifts_from_nhl_html game %s: %d unmapped shifts for players: %s', 
                               game_id, unmapped_count, unmapped_players)
        
        # Build shifts_by_player mapping from all_shifts (populate for both per-player and event-derived)
        # Now use canonical player_id values
        shifts_by_player = {}
        for s in all_shifts:
            keyp = s.get('player_id') if s.get('player_id') is not None else s.get('player_number') if s.get('player_number') is not None else 'unknown'
            shifts_by_player.setdefault(keyp, []).append(s)

        result = {'game_id': game_id, 'raw': '\n\n'.join(raw_combined), 'all_shifts': all_shifts, 'shifts_by_player': shifts_by_player}
        if debug:
            result['debug'] = {
                'urls_tried': tried_urls, 
                'tables_scanned': tables_scanned, 
                'players_scanned': players_scanned, 
                'found_shifts': len(all_shifts),
                'team_ids': team_ids,
                'team_id_set_count': team_id_set_count,
                'roster_mapping': {
                    'home_players': len(roster_map.get('home', {})),
                    'away_players': len(roster_map.get('away', {})),
                    'mapped_shifts': mapped_count,
                    'unmapped_shifts': unmapped_count,
                    'unmapped_players': list(unmapped_players)
                }
            }
        
        if all_shifts:
            _cache_put('shifts_html', key, result)

        return result

    except Exception as e:
        logging.exception('get_shifts_from_nhl_html: failed for %s: %s', game_id, e)
        return {'game_id': game_id, 'raw': None, 'all_shifts': [], 'shifts_by_player': {}, 'debug': {'error': str(e)}}


def compare_shifts(game_id: Any, debug: bool = False) -> Dict[str, Any]:
    """Compare outputs from get_shifts and get_shifts_from_nhl_html for a game_id.

    Returns a dict containing both raw results and a small "diff" summary useful
    for debugging parsing discrepancies.
    """
    res = {'game_id': game_id}
    j1 = get_shifts(game_id, force_refresh=True)
    j2 = get_shifts_from_nhl_html(game_id, force_refresh=True, debug=debug)

    all1 = j1.get('all_shifts') or []
    all2 = j2.get('all_shifts') or []

    # helper to extract player identifiers
    def players_from_shifts(lst):
        s = set()
        for sh in lst:
            pid = sh.get('player_id') if isinstance(sh, dict) else None
            pnum = sh.get('player_number') if isinstance(sh, dict) else None
            if pid is None and pnum is not None:
                s.add(('num', pnum))
            elif pid is not None:
                s.add(('id', pid))
        return s

    p1 = players_from_shifts(all1)
    p2 = players_from_shifts(all2)

    only1 = p1 - p2
    only2 = p2 - p1

    res['api'] = {'raw': j1, 'count': len(all1), 'players': p1}
    res['html'] = {'raw': j2, 'count': len(all2), 'players': p2}
    res['diff'] = {'only_api_players': list(only1), 'only_html_players': list(only2), 'api_count': len(all1), 'html_count': len(all2)}

    # sample mismatched shifts (by player id/number)
    sample_diffs = []
    # look for players present in one but not the other and include a couple sample shifts
    for tag, val in list(only1)[:5]:
        rec = {'source': 'api_only', 'player': (tag, val), 'sample_shifts': []}
        for sh in all1:
            pid = sh.get('player_id')
            pnum = sh.get('player_number')
            if (pid is None and tag == 'num' and pnum == val) or (pid == val and tag == 'id'):
                rec['sample_shifts'].append({'period': sh.get('period'), 'start_raw': sh.get('start_raw'), 'end_raw': sh.get('end_raw')})
                if len(rec['sample_shifts']) >= 3:
                    break
        sample_diffs.append(rec)
    for tag, val in list(only2)[:5]:
        rec = {'source': 'html_only', 'player': (tag, val), 'sample_shifts': []}
        for sh in all2:
            pid = sh.get('player_id')
            pnum = sh.get('player_number')
            if (pid is None and tag == 'num' and pnum == val) or (pid == val and tag == 'id'):
                rec['sample_shifts'].append({'period': sh.get('period'), 'start_raw': sh.get('start_raw'), 'end_raw': sh.get('end_raw')})
                if len(rec['sample_shifts']) >= 3:
                    break
        sample_diffs.append(rec)

    res['diff']['sample_diffs'] = sample_diffs

    if debug:
        logging.debug('compare_shifts: %s', res['diff'])
    return res


if __name__ == '__main__':
    # quick CLI to compare for a provided game id when run directly
    import sys
    if len(sys.argv) > 1:
        gid = sys.argv[1]
        try:
            gid = int(gid)
        except Exception:
            pass
        out = compare_shifts(gid, debug=True)
        print('API shifts count:', out['api']['count'])
        print('HTML shifts count:', out['html']['count'])
        print('Only in API players:', out['diff']['only_api_players'])
        print('Only in HTML players:', out['diff']['only_html_players'])
    else:
        print('Usage: python nhl_api.py <game_id>')
