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
from rink import rink_goal_xs


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

    home_id = game_feed['homeTeam']['id']
    away_id = game_feed['awayTeam']['id']

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
        # Capture period time and whether it's 'remaining' or 'elapsed'. The
        # api-web feeds sometimes expose either `timeRemaining` or
        # `timeInPeriod`; keep track of which was present so downstream logic
        # can interpret the value correctly.
        time_remaining = p.get('timeRemaining') if 'timeRemaining' in p else None
        time_in_period = p.get('timeInPeriod') if 'timeInPeriod' in p else None
        period_time = time_remaining or time_in_period or None
        if time_remaining is not None:
            period_time_type = 'remaining'
        elif time_in_period is not None:
            period_time_type = 'elapsed'
        else:
            period_time_type = None

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

        # figure out game situation (e.g. even-strength, PP, SH) if possible
        situation_code = p['situationCode'] if 'situationCode' in p else None

        home_skaters = situation_code[2]
        home_goalie_in_net = situation_code[3]
        away_skaters = situation_code[1]
        away_goalie_in_net = situation_code[0]

        # define game state (initialize to None for safety)
        game_state = None
        if team_id == home_id:
            game_state = f"{home_skaters}v{away_skaters}"
        elif team_id == away_id:
            game_state = f"{away_skaters}v{home_skaters}"

        # define is_net_empty
        is_net_empty = 0
        if team_id == home_id and away_goalie_in_net == '0':
            is_net_empty = 1

        if team_id == away_id and home_goalie_in_net == '0':
            is_net_empty = 1


        try:
            events.append({
                'event': ev_norm,
                'x': float(x),
                'y': float(y),
                'game_state': game_state,
                'is_net_empty': is_net_empty,
                'period': period,
                'period_time': period_time,
                'player_id': player_id,
                'team_id': team_id,
                'home_id': home_id,
                'away_id': away_id,
                'home_team_defending_side': home_side,
                'game_id': game_feed.get('id') or game_feed.get('gamePk'),
                'periodTimeType': period_time_type
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
        if game_id is None:
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
                feed = nhl_api.get_game_feed(game_id)
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
                    logging.warning('429 fetching game %s; attempt %d/%d; sleeping %.1fs', game_id, attempt, max_retries, sleep_for)
                    time.sleep(sleep_for)
                    backoff = min(max_backoff, backoff * 2)
                    continue
                else:
                    logging.debug('Non-retryable HTTP error for game %s: %s', game_id, he)
                    return None
            except requests.exceptions.RequestException as re:
                logging.warning('Network error fetching game %s: %s — attempt %d/%d, sleeping %.1fs', game_id, re, attempt, max_retries, backoff)
                time.sleep(backoff + random.uniform(0, 0.5))
                backoff = min(max_backoff, backoff * 2)
                continue
            except Exception as e:
                logging.warning('Unexpected error fetching game %s: %s', game_id, e)
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

    # canonical goal positions from rink.py
    try:
        left_goal_x, right_goal_x = rink_goal_xs()
    except Exception:
        # fallback to historical constants if rink helper unavailable
        left_goal_x, right_goal_x = -89.0, 89.0

    for ev in game_feed:
        rec = dict(ev)  # shallow copy

        # Book-keeping stuff
        # normalize period to integer when possible
        try:
            rec['period'] = int(rec.get('period')) if rec.get('period') is not None else None
        except Exception:
            rec['period'] = None

        # Compute two time-derived columns:
        # - periodTime_seconds_elapsed: seconds elapsed within the period
        # - total_time_elapsed_seconds: seconds elapsed since game start
        #
        # Assumptions:
        # * Regulation periods (1-3) are 20 minutes (1200s)
        # * Overtime periods (4+) are treated as 5 minutes (300s)
        period_time_str = rec.get('periodTime')
        period_time_type = rec.get('periodTimeType')  # 'remaining' or 'elapsed' or None

        period_elapsed = None
        if period_time_str is not None:
            # numeric input
            if isinstance(period_time_str, (int, float)):
                try:
                    val = int(period_time_str)
                    if period_time_type == 'remaining':
                        per_len = 1200 if (rec.get('period') is None or rec.get('period') <= 3) else 300
                        period_elapsed = max(0, per_len - val)
                    else:
                        period_elapsed = val
                except Exception:
                    period_elapsed = None
            else:
                # parse mm:ss style strings or plain numeric strings
                try:
                    s = str(period_time_str).strip()
                    if ':' in s:
                        mm, ss = s.split(':')
                        seconds = int(mm) * 60 + int(ss)
                    else:
                        seconds = int(float(s))
                    if period_time_type == 'remaining':
                        per_len = 1200 if (rec.get('period') is None or rec.get('period') <= 3) else 300
                        period_elapsed = max(0, per_len - seconds)
                    else:
                        period_elapsed = seconds
                except Exception:
                    period_elapsed = None

        rec['periodTime_seconds_elapsed'] = int(period_elapsed) if period_elapsed is not None else None

        # compute total time elapsed since game start (seconds)
        total_elapsed = None
        if rec.get('period') is not None and rec.get('period') >= 1 and period_elapsed is not None:
            p = int(rec['period'])
            if p <= 3:
                total_elapsed = (p - 1) * 1200 + period_elapsed
            else:
                # period 4 == first overtime: add three regulation periods + (p-4)*OT + elapsed
                total_elapsed = 3 * 1200 + (p - 4) * 300 + period_elapsed

        rec['total_time_elapsed_seconds'] = int(total_elapsed) if total_elapsed is not None else None

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

        if x is not None and y is not None:
            # calculate distance to the attacked goal
            # Determine which goal the shooter is attacking. The feed includes
            # `home_team_defending_side` which indicates the side the home team
            # is defending; the attacking goal is the opposite side.
            if rec.get('team_id') == rec.get('home_id'):
                # shooter is home -> attacking goal is the side opposite what home defends
                if rec.get('home_team_defending_side') == 'left':
                    goal_x = right_goal_x
                elif rec.get('home_team_defending_side') == 'right':
                    goal_x = left_goal_x
                else:
                    goal_x = right_goal_x
            elif rec.get('team_id') == rec.get('away_id'):
                # shooter is away -> attacking goal is the side not defended by home
                if rec.get('home_team_defending_side') == 'left':
                    goal_x = left_goal_x
                elif rec.get('home_team_defending_side') == 'right':
                    goal_x = right_goal_x
                else:
                    goal_x = left_goal_x
            else:
                # fallback to right goal
                goal_x = right_goal_x

            goal_y = 0.0
            distance = math.hypot(x - goal_x, y - goal_y)
            rec['distance'] = distance

            # Calculate angle such that:
            # - angle = 0° along the goal-line vector pointing toward the
            #   goalie's left (i.e. along +y for left-side goal, along -y for right-side goal)
            # - angle increases clockwise
            #
            # Vector from goal center to shot:
            vx = x - goal_x
            vy = y - goal_y

            # Reference vector along goal line pointing toward goalie's left:
            # If goal_x < 0 (left goal), goalie faces +x so his left is +y.
            # If goal_x > 0 (right goal), goalie faces -x so his left is -y.
            if goal_x < 0:
                rx, ry = 0.0, 1.0
            else:
                rx, ry = 0.0, -1.0

            # Signed angle from ref r to vector v (CCW positive): atan2(cross, dot)
            cross = rx * vy - ry * vx
            dot = rx * vx + ry * vy
            angle_rad_ccw = math.atan2(cross, dot)

            # We want clockwise positive, so invert sign, convert to degrees
            angle_deg = ( -math.degrees(angle_rad_ccw) ) % 360.0

            rec['angle_deg'] = angle_deg

        else:
            rec['dist_center'] = None
            rec['angle_deg'] = None
        # Calculate distance from goal

        # Calculate angle

        # Normalize and derive helpful ML features
        rec['is_goal'] = 1 if rec.get('event') == 'GOAL' else 0






        # periodTimeType was only needed to compute elapsed seconds earlier;
        # remove it so final records do not include this transient field.
        rec.pop('periodTimeType', None)

        elaborated_game_feed.append(rec)

    return elaborated_game_feed

def events_to_df(events):
    import pandas as pd
    # convert events list to pandas DataFrame
    df = pd.DataFrame.from_records(events)


if __name__ == '__main__':

    debug_season = True
    if debug_season:
        df = _season(out_path='/Users/harrisonmcadams/PycharmProjects/new_puck/static')
        print('Season dataframe shape:', df.shape)
        if not df.empty:
            print(df.head())
        else:
            print('No events found for season')
