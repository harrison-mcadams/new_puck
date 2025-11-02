"""Parse various NHL game-feed shapes and extract shot/goal events.

This module exposes `_game` which normalizes api-web play objects into a
consistent list of event dictionaries. It also provides `_season` which
fetches a season's games and returns a concatenated pandas.DataFrame
suitable for ML.
"""

import math
import logging
import time
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import random
import email.utils as email_utils

import pandas as pd
import requests

import nhl_api


logging.basicConfig(level=logging.INFO)


def _game(game_feed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract shot and goal events with coordinates from an api-web game feed.

    This parser intentionally supports the api-web/play-by-play shape where a
    top-level `plays` (or `playByPlay.play`) list contains events with
    `details` and `coordinates`.

    Returns a list of event dicts with keys:
      - event: 'SHOT' or 'GOAL'
      - x, y: float coordinates
      - period, periodTime
      - playerID, teamID, teamAbbrev
      - homeTeamDefendingSide, gameID
    """
    events: List[Dict[str, Any]] = []

    plays = None
    if isinstance(game_feed, dict):
        plays = game_feed.get('plays') or game_feed.get('playByPlay', {}).get('plays')

    if not isinstance(plays, list):
        # nothing we can parse
        logging.debug('_game(): no plays list found in feed; keys=%s', list(game_feed.keys()) if isinstance(game_feed, dict) else type(game_feed))
        return events

    for p in plays:
        if not isinstance(p, dict):
            continue

        # api-web often uses a textual descriptor key such as 'typeDescKey'
        ev_type = p.get('typeDescKey') or (p.get('type') or {}).get('description') or p.get('typeCode')
        if not isinstance(ev_type, str):
            continue
        r = ev_type.strip().lower()
        if r in ('shot-on-goal', 'shot_on_goal', 'shot', 'shot on goal'):
            ev_norm = 'SHOT'
        elif r == 'goal':
            ev_norm = 'GOAL'
        else:
            continue

        details = p.get('details') or p.get('detail') or {}
        coords = p.get('coordinates') or {}

        x = None
        y = None
        if isinstance(details, dict):
            x = details.get('xCoord') if x is None else x
            y = details.get('yCoord') if y is None else y
        if x is None:
            x = coords.get('x')
        if y is None:
            y = coords.get('y')
        if x is None or y is None:
            continue

        # period info may be at different keys
        period = None
        if isinstance(p.get('periodDescriptor'), dict):
            period = p.get('periodDescriptor', {}).get('number')
        else:
            period = p.get('period') or p.get('periodNumber')
        period_time = p.get('timeRemaining') or p.get('timeInPeriod') or None

        player_id = None
        if isinstance(details, dict):
            player_id = details.get('shootingPlayerId') or details.get('playerId')

        team_id = details.get('eventOwnerTeamId') if isinstance(details, dict) else None
        team_abbrev = None
        team_obj = p.get('team') or {}
        if isinstance(team_obj, dict):
            team_abbrev = team_obj.get('triCode') or team_obj.get('abbrev') or team_obj.get('name')
            team_id = team_id or team_obj.get('id')

        home_side = p.get('homeTeamDefendingSide') or p.get('home_team_defending_side')

        try:
            events.append({
                'event': ev_norm,
                'x': float(x),
                'y': float(y),
                'period': period,
                'periodTime': period_time,
                'playerID': player_id,
                'teamID': team_id,
                'teamAbbrev': team_abbrev,
                'homeTeamDefendingSide': home_side,
                'gameID': game_feed.get('id') or game_feed.get('gamePk')
            })
        except Exception:
            # skip rows that fail numeric conversion
            continue

    return events


def _period_time_to_seconds(t: Optional[str]) -> Optional[int]:
    """Convert a period time like '12:34' to seconds remaining or elapsed.

    If the input is None or not parseable, return None.
    """
    if t is None:
        return None
    if isinstance(t, (int, float)):
        return int(t)
    try:
        parts = str(t).split(':')
        if len(parts) == 2:
            mm, ss = parts
            return int(mm) * 60 + int(ss)
        return int(float(t))
    except Exception:
        return None


def _season(season: str = '20252026', team: str = 'all', out_path: Optional[str] = None,
            use_cache: bool = False, cache_limit_files: Optional[int] = 200,
            min_delay: float = 0.5, jitter: float = 0.2, max_workers: int = 4,
            verbose: bool = True) -> pd.DataFrame:
    """Fetch a full season (or a team's season) and return a concatenated
    pandas.DataFrame of shot and goal events suitable for ML.

    Improvements over the previous version:
    - caching is optional (use_cache=False by default to avoid disk usage)
    - limited cache size when enabled (cache_limit_files)
    - concurrent fetching using a ThreadPoolExecutor (configurable max_workers)
    - lightweight per-task retry/backoff; per-thread small delays (min_delay+jitter)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    games = nhl_api.get_season(team=team, season=season)
    records: List[Dict[str, Any]] = []

    # Prepare cache directory only if requested
    cache_dir = None
    if use_cache:
        if out_path:
            cache_dir = Path(out_path) / 'cache'
        else:
            cache_dir = Path('static') / 'cache'
        cache_dir.mkdir(parents=True, exist_ok=True)

    max_retries = 5
    max_backoff = 120.0

    def _prune_cache_if_needed():
        if not cache_dir or cache_limit_files is None:
            return
        try:
            files = sorted([p for p in cache_dir.iterdir() if p.is_file()], key=lambda p: p.stat().st_mtime)
            while len(files) > int(cache_limit_files):
                old = files.pop(0)
                try:
                    old.unlink()
                except Exception:
                    pass
        except Exception:
            pass

    def fetch_game_feed_task(gm: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fetch a game feed for a single game dict (gm). Returns a dict or None.

        This function is intentionally lighter-weight and suitable for running
        in a thread pool. It honors optional caching when enabled.
        """
        game_id = gm.get('id') or gm.get('gamePk') or gm.get('gameID')
        if game_ID is None:
            return None

        cache_file = cache_dir / f'game_{game_id}.json' if cache_dir else None

        # Try cache first
        if cache_file and cache_file.exists():
            try:
                return json.loads(cache_file.read_text(encoding='utf-8'))
            except Exception:
                pass

        # small per-thread delay to spread requests
        try:
            time.sleep(float(min_delay) + random.uniform(0, float(jitter)))
        except Exception:
            pass

        backoff = 1.0
        for attempt in range(1, max_retries + 1):
            try:
                feed = nhl_api.get_game_feed(game_ID)
                # save to cache if requested
                if cache_file:
                    try:
                        cache_file.write_text(json.dumps(feed), encoding='utf-8')
                        _prune_cache_if_needed()
                    except Exception:
                        pass
                return feed
            except requests.exceptions.HTTPError as he:
                resp = getattr(he, 'response', None)
                status = getattr(resp, 'status_code', None)
                if status == 429:
                    # honor Retry-After if present, otherwise backoff
                    ra = None
                    try:
                        ra_hdr = (resp.headers.get('Retry-After') if resp is not None and resp.headers is not None else None)
                    except Exception:
                        ra_hdr = None
                    if ra_hdr:
                        try:
                            ra = float(ra_hdr)
                        except Exception:
                            try:
                                parsed = email_utils.parsedate_to_datetime(ra_hdr)
                                if isinstance(parsed, datetime):
                                    if parsed.tzinfo is None:
                                        parsed = parsed.replace(tzinfo=timezone.utc)
                                    ra = max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())
                            except Exception:
                                ra = None
                    sleep_for = ra if ra is not None else backoff
                    sleep_for = min(max_backoff, float(sleep_for)) + random.uniform(0, 1.0)
                    logging.warning('429 fetching game %s; attempt %d/%d; sleeping %.1fs', game_ID, attempt, max_retries, sleep_for)
                    time.sleep(sleep_for)
                    backoff = min(max_backoff, backoff * 2)
                    continue
                else:
                    logging.debug('Non-retryable HTTP error for game %s: %s', game_ID, he)
                    return None
            except requests.exceptions.RequestException as re:
                logging.warning('Network error fetching game %s: %s — attempt %d/%d, sleeping %.1fs', game_ID, re, attempt, max_retries, backoff)
                time.sleep(backoff + random.uniform(0, 0.5))
                backoff = min(max_backoff, backoff * 2)
                continue
            except Exception as e:
                logging.warning('Unexpected error fetching game %s: %s', game_ID, e)
                return None
        return None

    # Inform start
    if verbose:
        print(f"_season: starting fetch for season={season}, team={team}, games={len(games)}, workers={max_workers}, use_cache={use_cache}", flush=True)
    else:
        logging.info('Starting _season: season=%s team=%s games=%d workers=%d use_cache=%s', season, team, len(games), max_workers, use_cache)

    # Fetch feeds concurrently with a simple console progress indicator
    feeds = []
    total_games = len(games)
    fetched_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        future_to_game = {exe.submit(fetch_game_feed_task, gm): gm for gm in games}
        for fut in as_completed(future_to_game):
            gm = future_to_game[fut]
            try:
                feed = fut.result()
            except Exception as e:
                logging.warning('Threaded fetch failed for game %s: %s', gm.get('id') or gm.get('gamePk'), e)
                feed = None
            fetched_count += 1
            # progress update
            if verbose:
                print(f'Fetching feeds: {fetched_count}/{total_games}', flush=True)
            else:
                logging.info('Fetching feeds: %d/%d', fetched_count, total_games)
            if feed:
                feeds.append((gm, feed))
    # final progress message
    if verbose:
        print(f'Finished fetching feeds: {fetched_count}/{total_games}', flush=True)
    else:
        logging.info('Finished fetching feeds: %d/%d', fetched_count, total_games)

    # Parse and elaborate feeds (single-threaded parsing, which is fast)
    total_feeds = len(feeds)
    parsed_count = 0
    for gm, game_feed in feeds:
        try:
            events = _game(game_feed)
        except Exception as e:
            logging.warning('Parser error for game %s: %s', gm.get('id') or gm.get('gamePk'), e)
            events = []
        parsed_count += 1
        if verbose:
            print(f'Parsing feeds: {parsed_count}/{total_feeds}', flush=True)
        else:
            logging.info('Parsing feeds: %d/%d', parsed_count, total_feeds)
        if not events:
            continue
        try:
            elaborated = _elaborate(events)
        except Exception as e:
            logging.warning('Elaboration error for game %s: %s', gm.get('id') or gm.get('gamePk'), e)
            continue
        records.extend(elaborated)
    if total_feeds:
        if verbose:
            print(f'Finished parsing feeds: {parsed_count}/{total_feeds}', flush=True)
        else:
            logging.info('Finished parsing feeds: %d/%d', parsed_count, total_feeds)

    # Build DataFrame and optionally save
    df = pd.DataFrame.from_records(records) if records else pd.DataFrame()
    if out_path and not df.empty:
        out_file = Path(out_path) / f'{season}.csv'
        try:
            df.to_csv(out_file, index=False)
            logging.info('Saved season data to %s', out_file)
        except Exception as e:
            logging.warning('Failed to save CSV %s: %s', out_file, e)
    logging.info('Finished _season: feeds=%d, records=%d', len(feeds), len(records))
    return df


def _elaborate(game_feed: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Take a list of event dicts (from `_game`) and derive ML-friendly features.

    Returns a new list of dicts with numeric fields (x,y,dist_center,angle_deg,periodTime_seconds)
    and other metadata.
    """

    elaborated_game_feed: List[Dict[str, Any]] = []

    for ev in game_feed:
        rec = dict(ev)  # shallow copy

        # Book-keeping stuff
        # ensure numeric period when possible
        try:
            rec['period'] = int(rec['period']) if rec.get('period') is not None else None
        except Exception:
            rec['period'] = None
        rec['periodTime_seconds'] = _period_time_to_seconds(rec.get('periodTime'))
        # x/y to numeric
        try:
            x = float(rec.get('x'))
            y = float(rec.get('y'))
        except Exception:
            x = None
            y = None
        rec['x'] = x
        rec['y'] = y

        ## Basic analysis stuff

        # Calculate distance from goal

        # Calculate angle

        # Normalize and derive helpful ML features
        rec['is_goal'] = 1 if rec.get('event') == 'GOAL' else 0




        if x is not None and y is not None:
            rec['dist_center'] = math.hypot(x, y)
            rec['angle_deg'] = math.degrees(math.atan2(y, x))
        else:
            rec['dist_center'] = None
            rec['angle_deg'] = None

        elaborated_game_feed.append(rec)

    return elaborated_game_feed


if __name__ == '__main__':
    df = _season(out_path='/Users/harrisonmcadams/PycharmProjects/new_puck/static')
    print('Season dataframe shape:', df.shape)
    if not df.empty:
        print(df.head())
    else:
        print('No events found for season')
