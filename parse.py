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


def _game(game_feed: Dict[str, Any]) -> pd.DataFrame:
    """Extract shot and goal events with coordinates from an api-web game feed.

    This parser intentionally supports the api-web/play-by-play shape where a
    top-level `plays` (or `playByPlay.play`) list contains events with
    `details` and `coordinates`.

    Returns a pandas.DataFrame where each row is an event and columns include:
      - event: 'SHOT' or 'GOAL'
      - x, y: float coordinates
      - period, period_time
      - player_id, team_id, team_abbrev
      - home_team_defending_side, game_id
    """
    events: List[Dict[str, Any]] = []
    # collect synthetic events to append later (e.g., predicted penalty_end)
    synthetic_events: List[Dict[str, Any]] = []

    plays = None
    # defensive initialize
    ev_type = None
    if isinstance(game_feed, dict):
        plays = game_feed.get('plays') or game_feed.get('playByPlay', {}).get('plays')

    if not isinstance(plays, list):
        # nothing we can parse
        logging.debug('_game(): no plays list found in feed; keys=%s', list(game_feed.keys()) if isinstance(game_feed, dict) else type(game_feed))
        # return an empty DataFrame for consistency
        return pd.DataFrame()

    # Safely extract home/away info; feeds can vary in structure
    home_team_obj = game_feed.get('homeTeam') or game_feed.get('home') or {}
    away_team_obj = game_feed.get('awayTeam') or game_feed.get('away') or {}
    try:
        home_id = home_team_obj.get('id') if isinstance(home_team_obj, dict) else None
    except Exception:
        home_id = None
    try:
        away_id = away_team_obj.get('id') if isinstance(away_team_obj, dict) else None
    except Exception:
        away_id = None

    try:
        home_abb = home_team_obj.get('abbrev') if isinstance(home_team_obj, dict) else None
    except Exception:
        home_abb = None
    try:
        away_abb = away_team_obj.get('abbrev') if isinstance(away_team_obj, dict) else None
    except Exception:
        away_abb = None

    for idx, p in enumerate(plays):
        if not isinstance(p, dict):
            continue

        # api-web often uses a textual descriptor key such as 'typeDescKey'
        ev_type = p.get('typeDescKey') or (p.get('type') or {}).get('description') or p.get('typeCode')


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
        # If this is not a coordinate-bearing event we still want to
        # continue parsing (we'll append synthetic events later) but for
        # the purpose of adding the main event we skip those without coords
        if x is None or y is None:
            # Before skipping, detect penalty starts that may not have coords
            # and still want to schedule a synthetic penalty_end event.
            # We'll fall through to continue skipping adding the main row,
            # but synthetic events will be added later.
            pass

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

        # Determine the home team's defending side robustly using helper.
        try:
            home_side = infer_home_defending_side_from_play(p, game_feed=game_feed)
        except Exception:
            # fallback to legacy direct keys if helper fails for some reason
            home_side = p.get('homeTeamDefendingSide') or p.get('home_team_defending_side')

        # figure out game situation (e.g. even-strength, PP, SH) if possible
        situation_code = p['situationCode'] if 'situationCode' in p else None

        # protect against malformed situation codes
        try:
            home_skaters = situation_code[2]
            home_goalie_in_net = situation_code[3]
            away_skaters = situation_code[1]
            away_goalie_in_net = situation_code[0]
        except Exception:
            home_skaters = None
            away_skaters = None
            home_goalie_in_net = '1'
            away_goalie_in_net = '1'

        # define game state (initialize to None for safety)
        game_state = None
        if team_id == home_id:
            if home_skaters is not None and away_skaters is not None:
                game_state = f"{home_skaters}v{away_skaters}"
        elif team_id == away_id:
            if home_skaters is not None and away_skaters is not None:
                game_state = f"{away_skaters}v{home_skaters}"

        # define is_net_empty
        is_net_empty = 0
        try:
            if team_id == home_id and away_goalie_in_net == '0':
                is_net_empty = 1

            if team_id == away_id and home_goalie_in_net == '0':
                is_net_empty = 1
        except Exception:
            is_net_empty = 0

        # Define whether the event is a shot attempt:
        shot_attempt_types = ['shot-on-goal', 'missed-shot', 'blocked-shot',
                              'goal']
        if ev_type in shot_attempt_types:
            is_shot_attempt = True
        else:
            is_shot_attempt = False

        # Compute elapsed seconds for this event (useful for synthetic events)
        # convert period_time to seconds and then to total elapsed since game start
        try:
            per_num = int(period) if period is not None else None
        except Exception:
            per_num = None
        per_len = 1200 if (per_num is None or per_num <= 3) else 300
        per_secs = None
        try:
            if period_time is not None:
                # attempt to parse mm:ss or numeric
                raw = _period_time_to_seconds(period_time)
                if raw is not None:
                    if period_time_type == 'remaining':
                        per_secs = max(0, per_len - int(raw))
                    else:
                        per_secs = int(raw)
        except Exception:
            per_secs = None

        total_elapsed = None
        try:
            if per_num is not None and per_num >= 1 and per_secs is not None:
                per_num_int = int(per_num)
                if per_num_int <= 3:
                    total_elapsed = (per_num_int - 1) * 1200 + int(per_secs)
                else:
                    total_elapsed = 3 * 1200 + (per_num_int - 4) * 300 + int(per_secs)
        except Exception:
            total_elapsed = None

        # Detect penalty starts and schedule synthetic penalty_end events.
        # Heuristic: ev_type contains 'penalty' or details indicate penalty.
        try:
            ev_lc = str(ev_type).strip().lower() if ev_type is not None else ''
        except Exception:
            ev_lc = ''
        is_penalty_start = False
        if 'penal' in ev_lc or 'penalty' in ev_lc:
            # some penalty events are labeled 'penalty' or contain 'penalty' in description
            is_penalty_start = True
        # Also inspect details/description for common penalty markers
        if not is_penalty_start and isinstance(details, dict):
            desc = (details.get('description') or details.get('typeDescription') or '')
            try:
                if 'penal' in str(desc).lower() or 'penalty' in str(desc).lower():
                    is_penalty_start = True
            except Exception:
                pass

        if is_penalty_start and total_elapsed is not None:
            # try to extract explicit duration (prefer seconds if present)
            dur_secs = None
            try:
                # common fields to check
                for k in ('penaltyDuration', 'penalty_seconds', 'penaltySeconds', 'penaltySecondsDuration', 'penaltyMinutes', 'penaltyMinutesDuration', 'penaltyMin'):
                    if isinstance(details, dict) and k in details and details.get(k) is not None:
                        val = details.get(k)
                        # numeric minutes -> seconds
                        try:
                            f = float(val)
                            if 'min' in k.lower() or 'minutes' in k.lower():
                                dur_secs = int(f * 60)
                            else:
                                dur_secs = int(f)
                            break
                        except Exception:
                            # try parsing MM:SS
                            try:
                                parsed = _period_time_to_seconds(val)
                                if parsed is not None:
                                    dur_secs = int(parsed)
                                    break
                            except Exception:
                                continue
                # some feeds encode penalty type text e.g. 'Minor' or 'Major'
                if dur_secs is None and isinstance(details, dict):
                    ptype = (details.get('penaltyType') or details.get('penaltySeverity') or details.get('type'))
                    if ptype:
                        pt = str(ptype).lower()
                        if 'major' in pt:
                            dur_secs = 300
                        elif 'minor' in pt:
                            dur_secs = 120
                        elif 'double' in pt:
                            dur_secs = 240
                        elif 'misconduct' in pt:
                            dur_secs = 600
            except Exception:
                dur_secs = None

            # Fallback default: minor penalty 2 minutes
            if dur_secs is None:
                dur_secs = 120

            # Important: penalty time actually starts at the ensuing faceoff,
            # not necessarily at the moment the penalty is recorded. Look ahead
            # in the plays list for the next faceoff and use its timestamp as
            # the penalty_start_total when available. Fall back to the current
            # event's total_elapsed when no faceoff timestamp is found.
            penalty_start_total = total_elapsed
            try:
                for j in range(idx + 1, len(plays)):
                    np = plays[j]
                    if not isinstance(np, dict):
                        continue
                    # derive next play's event type
                    np_ev = np.get('typeDescKey') or (np.get('type') or {}).get('description') or np.get('typeCode')
                    try:
                        np_ev_lc = str(np_ev).strip().lower() if np_ev is not None else ''
                    except Exception:
                        np_ev_lc = ''
                    # consider common faceoff descriptors
                    if 'faceoff' in np_ev_lc or 'face-off' in np_ev_lc:
                        # compute this play's total_elapsed similarly
                        try:
                            np_period = None
                            if isinstance(np.get('periodDescriptor'), dict):
                                np_period = np.get('periodDescriptor', {}).get('number')
                            else:
                                np_period = np.get('period') or np.get('periodNumber')
                            try:
                                np_per_num = int(np_period) if np_period is not None else None
                            except Exception:
                                np_per_num = None
                            np_per_len = 1200 if (np_per_num is None or np_per_num <= 3) else 300
                            np_time_remaining = np.get('timeRemaining') if 'timeRemaining' in np else None
                            np_time_in_period = np.get('timeInPeriod') if 'timeInPeriod' in np else None
                            np_period_time = np_time_remaining or np_time_in_period or None
                            np_period_time_type = 'remaining' if np_time_remaining is not None else ('elapsed' if np_time_in_period is not None else None)
                            np_per_secs = None
                            if np_period_time is not None:
                                parsed = _period_time_to_seconds(np_period_time)
                                if parsed is not None:
                                    if np_period_time_type == 'remaining':
                                        np_per_secs = max(0, np_per_len - int(parsed))
                                    else:
                                        np_per_secs = int(parsed)
                            if np_per_num is not None and np_per_num >= 1 and np_per_secs is not None:
                                pnum = int(np_per_num)
                                if pnum <= 3:
                                    penalty_start_total = (pnum - 1) * 1200 + int(np_per_secs)
                                else:
                                    penalty_start_total = 3 * 1200 + (pnum - 4) * 300 + int(np_per_secs)
                                break
                        except Exception:
                            # if computing next play time fails, continue scanning
                            continue
            except Exception:
                pass

            scheduled_end_total = int(penalty_start_total) + int(dur_secs)

            # Convert scheduled_end_total back to period and period_time (elapsed)
            sched_p = None
            sched_period_elapsed = None
            try:
                if scheduled_end_total < 3 * 1200:
                    sched_p = int(scheduled_end_total // 1200) + 1
                    sched_period_elapsed = int(scheduled_end_total - (sched_p - 1) * 1200)
                else:
                    # overtime periods
                    extra = scheduled_end_total - 3 * 1200
                    ot_index = int(extra // 300)
                    sched_p = 4 + ot_index
                    sched_period_elapsed = int(extra - ot_index * 300)
            except Exception:
                sched_p = None
                sched_period_elapsed = None

            # format mm:ss
            sched_period_time_str = None
            try:
                if sched_period_elapsed is not None:
                    mm = int(sched_period_elapsed // 60)
                    ss = int(sched_period_elapsed % 60)
                    sched_period_time_str = f"{mm:02d}:{ss:02d}"
            except Exception:
                sched_period_time_str = None

            # Build synthetic event dict for penalty_end
            synth = {
                'event': 'penalty_end',
                'x': None,
                'y': None,
                # optimistic reversion to even strength when penalty expires
                'game_state': '5v5',
                'is_net_empty': 0,
                'period': sched_p,
                'period_time': sched_period_time_str,
                'player_id': player_id,
                'team_id': team_id,
                'home_id': home_id,
                'away_id': away_id,
                'home_abb': home_abb,
                'away_abb': away_abb,
                'home_team_defending_side': home_side,
                'game_id': game_feed.get('id') or game_feed.get('gamePk'),
                'periodTimeType': 'elapsed',
                'synthetic': True,
                'predicted_penalty_duration_seconds': int(dur_secs),
                'predicted_penalty_end_total_seconds': int(scheduled_end_total),
            }
            synthetic_events.append(synth)

        try:
            # Only append main event if it has coordinates; otherwise skip
            if x is not None and y is not None:
                events.append({
                    'event': ev_type,
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
                    'home_abb': home_abb,
                    'away_abb': away_abb,
                    'home_team_defending_side': home_side,
                    'game_id': game_feed.get('id') or game_feed.get('gamePk'),
                    'periodTimeType': period_time_type
                })
            else:
                # no coords: skip main event (but synthetic events may exist)
                pass
        except Exception:
            # skip rows that fail numeric conversion
            continue

    # Append synthetic events (e.g., predicted penalty_end) to the events list
    if synthetic_events:
        try:
            events.extend(synthetic_events)
        except Exception:
            pass

    try:
        return pd.DataFrame.from_records(events)
    except Exception:
        return pd.DataFrame()


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
    # Save out the fetched game feeds.

    # If an output path was provided, persist the raw feeds to disk (JSON + CSV)
    if out_path and feeds:
        try:
            out_dir_path = Path(out_path)
            out_dir_path.mkdir(parents=True, exist_ok=True)
            raw_json_path = out_dir_path / f'{season}_raw_game_feeds.json'
            # write JSON backup with game metadata and feed
            try:
                with raw_json_path.open('w', encoding='utf-8') as jfh:
                    json.dump([{'game_meta': gm, 'feed': feed} for gm, feed in feeds], jfh, ensure_ascii=False)
                logging.info('Saved raw game feeds JSON to %s', raw_json_path)
            except Exception as e:
                logging.warning('Failed to write raw feeds JSON %s: %s', raw_json_path, e)

            # also write a compact CSV (game_id, feed_json)
            try:
                csv_path = out_dir_path / f'{season}_raw_game_feeds.csv'
                import csv as _csv
                with csv_path.open('w', encoding='utf-8', newline='') as fh:
                    writer = _csv.writer(fh)
                    writer.writerow(['game_id', 'feed'])
                    for gm, feed in feeds:
                        gid = gm.get('id') or gm.get('gamePk') or gm.get('gameID')
                        try:
                            writer.writerow([gid, json.dumps(feed, ensure_ascii=False)])
                        except Exception:
                            writer.writerow([gid, '{}'])
                logging.info('Saved raw game feeds CSV to %s', csv_path)
            except Exception as e:
                logging.warning('Failed to write raw feeds CSV: %s', e)
        except Exception as e:
            logging.warning('Failed to persist raw game feeds to out_path %s: %s', out_path, e)


    # Parse and elaborate feeds (single-threaded parsing, which is fast)
    total_feeds = len(feeds)
    parsed_count = 0
    for gm, game_feed in feeds:
        try:
            events_df = _game(game_feed)
        except Exception as e:
            logging.warning('Parser error for game %s: %s', gm.get('id') or gm.get('gamePk'), e)
            events_df = pd.DataFrame()
        parsed_count += 1
        if verbose:
            print(f'Parsing feeds: {parsed_count}/{total_feeds}', flush=True)
        else:
            logging.info('Parsing feeds: %d/%d', parsed_count, total_feeds)
        if events_df is None or events_df.empty:
            continue
        try:
            elaborated_df = _elaborate(events_df)
        except Exception as e:
            logging.warning('Elaboration error for game %s: %s', gm.get('id') or gm.get('gamePk'), e)
            continue
        # extend records with dictionaries
        try:
            records.extend(elaborated_df.to_dict('records'))
        except Exception:
            # fallback: if elaborated_df isn't a DataFrame, try iterating
            try:
                for r in elaborated_df:
                    records.append(r)
            except Exception:
                pass
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


def _elaborate(game_feed: pd.DataFrame) -> pd.DataFrame:
    """Take a DataFrame produced by `_game` and derive ML-friendly features.

    Returns a DataFrame with numeric fields (x,y,distance,angle_deg,periodTime_seconds_elapsed,
    total_time_elapsed_seconds) and other metadata.
    """

    elaborated_game_feed: List[Dict[str, Any]] = []

    # canonical goal positions from rink.py
    try:
        left_goal_x, right_goal_x = rink_goal_xs()
    except Exception:
        # fallback to historical constants if rink helper unavailable
        left_goal_x, right_goal_x = -89.0, 89.0

    # Defensive: handle None or empty inputs quickly
    if game_feed is None:
        return pd.DataFrame()

    # Accept either a DataFrame or an iterable of dict-like rows
    try:
        if isinstance(game_feed, pd.DataFrame):
            rows = game_feed.to_dict('records')
        else:
            rows = list(game_feed)
    except Exception as e:
        logging.warning('_elaborate(): failed to materialize rows from input: %s', e)
        return pd.DataFrame()

    if not rows:
        # nothing to do
        return pd.DataFrame()

    # Build a temporary DataFrame of the raw rows so we can run per-period
    # heuristics (we have access to the whole game's rows here, unlike in
    # `_game`). Use this to infer `home_team_defending_side` for periods
    # where the feed lacks explicit metadata. The inference prefers shot
    # attempts in the period and requires at least a couple points to be
    # confident.
    try:
        raw_df = pd.DataFrame(rows)
    except Exception:
        raw_df = None

    period_side_map: Dict[Any, Optional[str]] = {}
    shot_types = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
    if raw_df is not None and not raw_df.empty and 'period' in raw_df.columns:
        try:
            raw_df['period'] = pd.to_numeric(raw_df['period'], errors='coerce')
        except Exception:
            pass
        try:
            periods = sorted(raw_df['period'].dropna().unique().tolist())
        except Exception:
            periods = []
        for per in periods:
            try:
                period_slice = raw_df[raw_df['period'] == per]
                if period_slice.empty:
                    period_side_map[per] = None
                    continue

                # prefer shot attempts when available
                if 'event' in period_slice.columns:
                    shots_slice = period_slice[period_slice['event'].isin(shot_types)]
                else:
                    shots_slice = period_slice

                # representative play dict for helper input
                rep = period_slice.iloc[0].to_dict()

                # require at least 2 points for coordinate-based inference
                try_df = shots_slice if len(shots_slice) >= 2 else (period_slice if len(period_slice) >= 2 else None)
                if try_df is None:
                    period_side_map[per] = None
                    continue

                try:
                    side = infer_home_defending_side_from_play(rep, game_feed=None, events_df=try_df)
                except Exception:
                    side = None
                period_side_map[per] = side
            except Exception:
                period_side_map[per] = None
    else:
        period_side_map = {}

    for ev in rows:
        try:
            rec = dict(ev)  # shallow copy

            # Book-keeping stuff
            # normalize period to integer when possible
            try:
                rec['period'] = int(rec.get('period')) if rec.get('period') is not None else None
            except Exception:
                rec['period'] = None

            # If the row lacks an explicit `home_team_defending_side`, fill it
            # from the per-period inference mapping computed above.
            if rec.get('home_team_defending_side') is None:
                inferred_side = period_side_map.get(rec.get('period'))
                if inferred_side is not None:
                    rec['home_team_defending_side'] = inferred_side

            # Compute two time-derived columns:
            # - periodTime_seconds_elapsed: seconds elapsed within the period
            # - total_time_elapsed_seconds: seconds elapsed since game start
            #
            # Assumptions:
            # * Regulation periods (1-3) are 20 minutes (1200s)
            # * Overtime periods (4+) are treated as 5 minutes (300s)
            period_time_str = rec.get('periodTime') or rec.get('period_time')
            period_time_type = rec.get('periodTimeType') or rec.get('periodTimeType')

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
            # Call to infer_home_defending_side around here, before any of
            # htat information is subsequently operated on

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

            # periodTimeType was only needed to compute elapsed seconds earlier;
            # remove it so final records do not include this transient field.
            rec.pop('periodTimeType', None)

            elaborated_game_feed.append(rec)

        except Exception as e:
            logging.warning('Error elaborating event: %s', e)
            # skip malformed event but continue processing others
            continue

    # return as DataFrame
    try:
        return pd.DataFrame.from_records(elaborated_game_feed)
    except Exception as e:
        logging.warning('Failed to build elaborated DataFrame: %s', e)
        return pd.DataFrame()


def _scrape(season: str = '20252026', team: str = 'all', out_dir: str = 'data', use_cache: bool = True,
             max_games: Optional[int] = None, max_workers: int = 8, verbose: bool = True,
             # New behavior flags
             save_raw: bool = True,
             save_csv: bool = True,
             save_json: bool = True,
             per_game_files: bool = False,
             process_elaborated: bool = False,
             save_elaborated: bool = False,
             return_feeds: bool = False,
             return_elaborated_df: bool = False
             ) -> Any:
    """Scrape and optionally persist and process raw game feeds for a season.

    This function fetches game feeds using the `nhl_api` helpers and provides
    flexible options for how fetched data are saved and/or processed.

    High-level behavior:
      1. Obtain a list of games via `nhl_api.get_season(team, season)`.
      2. Concurrently fetch each game's feed via `nhl_api.get_game_feed(game_id)`.
      3. Optionally save raw feeds to disk (combined CSV, combined JSON, and/or
         per-game JSON files).
      4. Optionally run the parser pipeline (`_game` + `_elaborate`) and
         aggregate results into an "elaborated" pandas.DataFrame suitable for
         downstream ML workflows.

    Important parameters (new/extended options):
      - save_raw (bool, default True): If True, persist fetched raw feeds to
        disk according to the CSV/JSON/per-game flags below.
      - save_csv (bool, default True): Write a single CSV file at
        ``{out_dir}/{season}/{season}_game_feeds.csv`` with columns
        ("game_id","feed") where `feed` is a JSON-encoded string for each row.
      - save_json (bool, default True): Write a combined JSON file at
        ``{out_dir}/{season}/{season}_game_feeds.json`` containing a list of
        ``{"game_id": id, "feed": <feed dict>}`` entries.
      - per_game_files (bool, default False): If True, write one JSON file per
        game as ``{out_dir}/{season}/game_{game_id}.json``. Useful for
        incremental processing and avoiding huge CSV cells.
      - process_elaborated (bool, default False): If True, for each fetched
        feed run `_game(feed)` then `_elaborate(...)` and aggregate the results
        into a single pandas.DataFrame (the "elaborated" DataFrame).
      - save_elaborated (bool, default False): If True (and
        `process_elaborated` is True and output exists), save the elaborated
        DataFrame to ``{out_dir}/{season}/{season}.csv``.
      - return_feeds (bool, default False): If True, include the list of
        raw feeds in the returned result (see return behavior below).
      - return_elaborated_df (bool, default False): If True, include the
        elaborated pandas.DataFrame in the returned result.

    Caching and backward compatibility:
      - `use_cache` combined with `save_csv` preserves the legacy behavior:
        if ``{out_dir}/{season}/{season}_raw_game_feeds.csv`` exists and the
        caller did not request returned data (neither `return_raw_feeds` nor
        `return_elaborated_df`), the function returns the existing CSV path and
        skips fetching. This keeps existing callers working as before.

    Return value(s):
      - Legacy default behaviour (when neither `return_feeds` nor
        `return_elaborated_df` are True): returns a string path to the saved
        CSV if present, else to the saved JSON, else an empty string.
      - When either `return_feeds` or `return_elaborated_df` is True, the
        function returns a dict with keys:
          * 'saved_paths' -> dict with keys 'csv','json','per_game' listing any
            saved file paths (or None/empty list)
          * 'feeds' -> list of fetched feeds (present only if
            `return_feeds` True)
          * 'elaborated_df' -> pandas.DataFrame (present only if
            `return_elaborated_df` True and `process_elaborated` was run)

    Notes and implementation details:
      - Fetching is done concurrently using ThreadPoolExecutor; set
        `max_workers` to control concurrency.
      - The worker `_fetch_task` returns None on failure; failures are logged
        and skipped so a partial result can still be produced.
      - When serializing feeds to CSV, feeds that fail `json.dumps` are written
        as the empty JSON object string `'{}'` to avoid breaking the whole file.
      - When processing elaborated data, rows with no usable events are
        skipped.
      - For large seasons or repeated runs, consider using `per_game_files`
        to reduce memory/CSV cell size and to allow incremental re-use.
      - The function favors robustness over strict failure: disk or
        serialization errors are logged but do not raise.

    Example usage:
      - Legacy (quick cache-aware save):
          parse._scrape(season='20252026', out_dir='data', use_cache=True)

      - Save per-game files and also return raw feeds for immediate use:
          res = parse._scrape(season='20252026', out_dir='data', per_game_files=True,
                              return_feeds=True)
          feeds = res['feeds']

      - Fetch, process into an elaborated DataFrame, save it, and return it:
          res = parse._scrape(season='20252026', out_dir='data',
                              process_elaborated=True, save_elaborated=True,
                              return_elaborated_df=True)
          df = res['elaborated_df']

    """
    # Prepare output directory
    base = Path(out_dir)
    season_dir = base / season
    season_dir.mkdir(parents=True, exist_ok=True)

    csv_path = season_dir / f'{season}_game_feeds.csv'
    json_path = season_dir / f'{season}_game_feeds.json'

    # preserve existing cache behavior (if csv exists and caller wants the default csv)
    if use_cache and save_csv and csv_path.exists():
        if verbose:
            print(f'_scrape: cache hit, returning existing file: {csv_path}')
        # maintain backward-compatibility by returning the csv path string
        if not (return_feeds or return_elaborated_df):
            return str(csv_path)
        # otherwise let the function continue and load the cached CSV/JSON as needed

    # Get season games list
    games = nhl_api.get_season(team=team, season=season) or []
    if max_games is not None and isinstance(max_games, int) and max_games > 0:
        games = games[:max_games]

    if verbose:
        print(f'_scrape: found {len(games)} games for team={team} season={season}')

    # helper to extract game id reliably
    def _game_id_from_gm(gm: Dict[str, Any]) -> Optional[int]:
        try:
            gid = gm.get('id') or gm.get('gamePk') or gm.get('gameID')
            return int(gid)
        except Exception:
            return None

    # prepare concurrent fetch
    from concurrent.futures import ThreadPoolExecutor, as_completed

    feeds = []

    # If caller specifically requested the elaborated DataFrame, ensure we
    # process elaborated records even if process_elaborated was False.
    if return_elaborated_df:
        process_elaborated = True

    # Preload cached feeds mapping (game_id -> feed dict) when use_cache is enabled
    cached_feeds: Dict[int, Dict[str, Any]] = {}
    if use_cache:
        # per-game files take precedence (fast lookup)
        try:
            season_dir = Path(out_dir) / season
            # CSV cache
            csv_cache_path = season_dir / f'{season}_game_feeds.csv'
            if csv_cache_path.exists():
                try:
                    import csv as _csv
                    with csv_cache_path.open('r', encoding='utf-8') as ch:
                        reader = _csv.reader(ch)
                        headers = next(reader, None)
                        for row in reader:
                            if not row:
                                continue
                            try:
                                gid = int(row[0])
                                feed = json.loads(row[1]) if len(row) > 1 and row[1] else {}
                                cached_feeds[gid] = feed
                            except Exception:
                                continue
                except Exception:
                    pass
            else:
                # try combined JSON cache
                json_cache_path = season_dir / f'{season}_game_feeds.json'
                if json_cache_path.exists():
                    try:
                        with json_cache_path.open('r', encoding='utf-8') as jfh:
                            lst = json.load(jfh)
                            # Accept either list of {'game_id','feed'} or [{'game_meta','feed'}]
                            for it in lst:
                                try:
                                    if isinstance(it, dict) and 'game_id' in it and 'feed' in it:
                                        gid = int(it['game_id'])
                                        cached_feeds[gid] = it['feed']
                                    elif isinstance(it, dict) and 'game_meta' in it and 'feed' in it:
                                        gm = it.get('game_meta') or {}
                                        gid = gm.get('id') or gm.get('gamePk') or gm.get('gameID')
                                        try:
                                            gid = int(gid)
                                            cached_feeds[gid] = it['feed']
                                        except Exception:
                                            continue
                                except Exception:
                                    continue
                    except Exception:
                        pass
        except Exception:
            pass

    def _fetch_task(gm: Dict[str, Any]):
        gid = _game_id_from_gm(gm)
        if gid is None:
            return None
        # If we preloaded a cached feed, return it immediately
        if use_cache and gid in cached_feeds:
            return {'game_id': gid, 'feed': cached_feeds[gid]}
        # If per-game files exist on disk, prefer them when use_cache is enabled
        try:
            per_path = season_dir / f'game_{gid}.json'
            if use_cache and per_path.exists():
                try:
                    with per_path.open('r', encoding='utf-8') as pfh:
                        return {'game_id': gid, 'feed': json.load(pfh)}
                except Exception:
                    pass
        except Exception:
            pass
        try:
            feed = nhl_api.get_game_feed(gid)
            if not isinstance(feed, dict) or not feed:
                return None
            return {'game_id': gid, 'feed': feed}
        except Exception as e:
            if verbose:
                print(f'_scrape: failed to fetch game {gid}: {e}')
            return None

    if verbose:
        print(f'_scrape: fetching up to {len(games)} feeds with {max_workers} workers...')

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(_fetch_task, gm): gm for gm in games}
        completed = 0
        total = len(futures)
        for fut in as_completed(futures):
            completed += 1
            try:
                res = fut.result()
            except Exception as e:
                res = None
                if verbose:
                    print(f'_scrape: task exception: {e}')
            if res:
                feeds.append(res)
            if verbose:
                print(f'_scrape: progress {completed}/{total}', end='\r')

    if verbose:
        print(f'\n_scrape: fetched {len(feeds)} feeds')
    # Persist raw feeds according to flags
    saved_paths: Dict[str, Any] = {'csv': '', 'json': '', 'per_game': []}
    if save_raw and feeds:
        # Ensure output directory exists
        try:
            season_dir = Path(out_dir) / season
            season_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            season_dir = Path(out_dir)

        if save_csv:
            try:
                import csv as _csv
                with (season_dir / f'{season}_game_feeds.csv').open('w', encoding='utf-8', newline='') as fh:
                    writer = _csv.writer(fh)
                    writer.writerow(['game_id', 'feed'])
                    for item in feeds:
                        try:
                            writer.writerow([item['game_id'], json.dumps(item['feed'], ensure_ascii=False)])
                        except Exception:
                            writer.writerow([item.get('game_id'), '{}'])
                saved_paths['csv'] = str(season_dir / f'{season}_game_feeds.csv')
                if verbose:
                    print(f'_scrape: saved CSV -> {saved_paths["csv"]}')
            except Exception as e:
                logging.warning('_scrape: failed to write CSV %s: %s', csv_path, e)

        if save_json:
            try:
                with (season_dir / f'{season}_game_feeds.json').open('w', encoding='utf-8') as jfh:
                    json.dump([{'game_id': f['game_id'], 'feed': f['feed']} for f in feeds], jfh, ensure_ascii=False)
                saved_paths['json'] = str(season_dir / f'{season}_game_feeds.json')
                if verbose:
                    print(f'_scrape: saved JSON -> {saved_paths["json"]}')
            except Exception as e:
                logging.warning('_scrape: failed to write JSON backup %s: %s', json_path, e)

        if per_game_files:
            for item in feeds:
                try:
                    pid = int(item['game_id'])
                except Exception:
                    pid = item.get('game_id')
                try:
                    ppath = season_dir / f'game_{pid}.json'
                    with ppath.open('w', encoding='utf-8') as pfh:
                        json.dump(item['feed'], pfh, ensure_ascii=False)
                    saved_paths['per_game'].append(str(ppath))
                except Exception:
                    continue

    # Optionally process into elaborated DataFrame. Initialize to empty
    # DataFrame so we always return a DataFrame object when requested.
    elaborated_df: Optional[pd.DataFrame] = pd.DataFrame()
    if process_elaborated and feeds:
        records: List[Dict[str, Any]] = []
        for item in feeds:
            try:
                ev_df = _game(item['feed'])
                if ev_df is None or ev_df.empty:
                    continue
                try:
                    edf = _elaborate(ev_df)
                except Exception:
                    continue
                if edf is None or edf.empty:
                    continue
                records.extend(edf.to_dict('records'))
            except Exception:
                continue
        if records:
            elaborated_df = pd.DataFrame.from_records(records)
            out_file = Path(out_dir) / season / f'{season}_df.csv'
            try:
                out_file.parent.mkdir(parents=True, exist_ok=True)
                elaborated_df.to_csv(out_file, index=False)
                if verbose:
                    print(f'_scrape: saved elaborated season CSV -> {out_file}')
            except Exception as e:
                logging.warning('_scrape: failed to save elaborated CSV %s: %s', str(out_file), e)

    # Build return value – maintain backward compatibility by default
    if not (return_feeds or return_elaborated_df):
        # Default legacy return: path to CSV if saved, else JSON path or empty string
        if saved_paths.get('csv'):
            return saved_paths['csv']
        if saved_paths.get('json'):
            return saved_paths['json']
        return ''
    # If the caller only requested the elaborated DataFrame, return it
    # directly (this provides the convenience the caller asked for).
    if return_elaborated_df and not return_feeds:
        # Return the DataFrame object (possibly empty)
        return elaborated_df

    # Otherwise return a dict preserving previous richer return structure
    result: Dict[str, Any] = {}
    result['saved_paths'] = saved_paths
    result['feeds'] = feeds if return_feeds else None
    result['elaborated_df'] = elaborated_df if return_elaborated_df else None
    return result


def _normalize_side(s: Optional[str]) -> Optional[str]:
    """Normalize a side string into 'left' or 'right' or None."""
    if s is None:
        return None
    try:
        s2 = str(s).strip().lower()
    except Exception:
        return None
    if not s2:
        return None
    if s2.startswith('l'):
        return 'left'
    if s2.startswith('r'):
        return 'right'
    # sometimes the feed uses 'home'/'away' in odd contexts; ignore those
    return None


def infer_home_defending_side_from_play(p: dict, game_feed: Optional[dict] = None, events_df=None) -> Optional[str]:
    """Infer the home team's defending side ('left' or 'right').

    Strategy (in order):
      1. Look for explicit keys on the play dict `p` that directly indicate
         the home defending side (common key: 'homeTeamDefendingSide').
      2. If not present, try common alternative keys on `p` or `game_feed`.
      3. As a robust fallback, if `events_df` (pandas DataFrame) for the
         same game/period/team is provided, infer from coordinates: compute
         mean distance to left/right canonical goal positions and choose the
         attacked goal that gives the smaller mean distance; from that
         determine the defended side.

    Parameters:
      - p: the single play dictionary (as received from the NHL feed)
      - game_feed: optional full game feed dict (for alternate metadata locations)
      - events_df: optional pandas.DataFrame of events for the same game/period
                   (helps infer side when explicit keys are missing)

    Returns: 'left' or 'right', or None if undetermined.
    """
    # 1) explicit keys on the play
    candidates = [
        'homeTeamDefendingSide',
        'home_team_defending_side',
        'homeTeamDefendSide',
        'homeDefendingSide',
    ]
    for k in candidates:
        if isinstance(p, dict) and k in p:
            s = _normalize_side(p.get(k))
            if s:
                return s

    # 2) sometimes encoded in nested structures
    # check p.get('team') but this usually doesn't contain home side
    # check top-level game_feed locations that sometimes exist
    if isinstance(game_feed, dict):
        # common locations
        try_keys = [
            'homeTeamDefendingSide',
            'home_team_defending_side',
            # older or nested shapes
            ('liveData', 'plays', 'homeTeamDefendingSide'),
            ('gameData', 'teams', 'home', 'defendingSide'),
            ('liveData', 'linescore', 'teams', 'home', 'defendingSide'),
        ]
        for tk in try_keys:
            if isinstance(tk, str) and tk in game_feed:
                s = _normalize_side(game_feed.get(tk))
                if s:
                    return s
            elif isinstance(tk, tuple):
                cur = game_feed
                ok = True
                for part in tk:
                    if not isinstance(cur, dict) or part not in cur:
                        ok = False
                        break
                    cur = cur.get(part)
                if ok:
                    s = _normalize_side(cur)
                    if s:
                        return s

    # 3) Fallback heuristic based on coordinates and events_df when available.
    # If we have a small DataFrame of events for the same (game_id, period)
    # we can estimate which goal each team is attacking by comparing mean
    # distance to canonical left/right goals. This is a per-period heuristic
    # (teams switch ends each period) so prefer passing the per-period slice.
    try:
        import pandas as _pd
        if events_df is not None and hasattr(events_df, 'empty') and not events_df.empty:
            # attempt to find the home id from the play or the df
            home_id = p.get('home_id') or p.get('homeId') or None
            # if not in play, try from df
            if home_id is None and 'home_id' in events_df.columns:
                # take the first non-null
                vals = events_df['home_id'].dropna().unique()
                if len(vals) > 0:
                    home_id = vals[0]

            # need team membership: prefer rows where team_id == home_id
            if home_id is None and 'team_id' in events_df.columns:
                # try to infer which team is home by majority of events marked home
                # if dataset contains home_id per row use that; otherwise skip
                pass

            # require numeric x/y and the rink helper
            try:
                from rink import rink_goal_xs
                left_x, right_x = rink_goal_xs()
            except Exception:
                left_x, right_x = -89.0, 89.0

            # build candidate rows for the home team (or all rows if home_id unknown)
            if home_id is not None and 'team_id' in events_df.columns:
                cand = events_df[events_df['team_id'] == home_id]
            else:
                cand = events_df

            # require at least a couple points to make a decision
            if len(cand) >= 2 and 'x' in cand.columns and 'y' in cand.columns:
                candx = pd.to_numeric(cand['x'], errors='coerce')
                candy = pd.to_numeric(cand['y'], errors='coerce')
                mask = candx.notna() & candy.notna()
                if mask.sum() >= 2:
                    dx_left = ((candx[mask] - left_x)**2 + (candy[mask])**2)**0.5
                    dx_right = ((candx[mask] - right_x)**2 + (candy[mask])**2)**0.5
                    mean_left = float(dx_left.mean())
                    mean_right = float(dx_right.mean())
                    # if mean_right < mean_left -> these events are closer to right goal -> attacking right
                    if mean_right < mean_left:
                        # attacking right => home is defending left
                        return 'left'
                    else:
                        return 'right'
    except Exception:
        pass

    # 4) if nothing works, return None
    return None

def _timing(df):
    # For a given game or games, extract timing intervals and totals for a
    # simple boolean condition applied to rows of the dataframe.
    #
    # This implementation focuses on `game_state` conditions (e.g. '5v5') and
    # uses the column `total_time_elapsed_seconds` as the timeline. It is
    # intentionally simple and easy to read; it returns both per-game interval
    # lists and per-game totals, plus an overall aggregate.
    #
    # Signature (kept simple for callers):
    #   _timing(df, condition_col='game_state', condition_value='5v5',
    #           time_col='total_time_elapsed_seconds', game_col='game_id')
    #
    # Returns: (intervals_per_game, totals_per_game, aggregate_totals)
    #  - intervals_per_game: dict game_id -> list of (start_sec, end_sec)
    #  - totals_per_game: dict game_id -> dict with keys 'condition_seconds', 'total_seconds'
    #  - aggregate_totals: dict with same keys aggregated across games

    # Allow callers to override via keyword args for flexibility while
    # maintaining backward compatibility with no-arg calls.

    # NOTE: keep the implementation conservative: if required columns are
    # missing, return empty/zero structures rather than raising.

    return _timing_impl(df)


def _timing_impl(df, condition_col: str = 'game_state', condition_value: str = '5v5',
                 time_col: str = 'total_time_elapsed_seconds', game_col: str = 'game_id'):
    """Core implementation used by `_timing`.

    Parameters
    - df: pandas.DataFrame containing events across one or more games
    - condition_col: column to test (default 'game_state')
    - condition_value: value that counts as "in condition" (default '5v5')
    - time_col: numeric column containing seconds elapsed since game start
    - game_col: column that identifies distinct games

    Returns (intervals_per_game, totals_per_game, aggregate_totals)
    """
    import pandas as _pd

    intervals_per_game = {}
    totals_per_game = {}
    agg_condition_seconds = 0.0
    agg_total_seconds = 0.0

    # Basic validation
    if df is None:
        return intervals_per_game, totals_per_game, {'condition_seconds': 0.0, 'total_seconds': 0.0}
    if not isinstance(df, _pd.DataFrame):
        try:
            df = _pd.DataFrame.from_records(list(df))
        except Exception:
            return intervals_per_game, totals_per_game, {'condition_seconds': 0.0, 'total_seconds': 0.0}

    if game_col not in df.columns:
        # try to treat entire df as single game with id None
        df = df.copy()
        df[game_col] = None

    # If time column missing, we cannot compute durations reliably
    if time_col not in df.columns:
        # return empty results rather than erroring
        return intervals_per_game, totals_per_game, {'condition_seconds': 0.0, 'total_seconds': 0.0}

    # Ensure numeric time
    df = df.copy()
    df[time_col] = _pd.to_numeric(df[time_col], errors='coerce')

    # Process per game
    for gid, gdf in df.groupby(game_col):
        # sort by time to ensure chronological order
        gdf = gdf.sort_values(by=time_col).reset_index(drop=True)
        # drop rows without time
        gdf = gdf[gdf[time_col].notna()]
        if gdf.shape[0] == 0:
            intervals_per_game[gid] = []
            totals_per_game[gid] = {'condition_seconds': 0.0, 'total_seconds': 0.0}
            continue

        # boolean mask for condition
        cond = (gdf.get(condition_col) == condition_value)

        # label contiguous blocks where cond value is same
        change_points = cond.ne(cond.shift(fill_value=False)).cumsum()
        intervals = []
        condition_seconds = 0.0

        # For each group where cond is True, compute start and end times
        grouped = gdf.groupby(change_points)
        for _, block in grouped:
            # block is contiguous with same cond value
            val = bool((block.get(condition_col) == condition_value).iloc[0])
            if not val:
                continue
            start_time = float(block[time_col].iloc[0])
            end_time = float(block[time_col].iloc[-1])
            # If there is only a single event in block, duration is 0; that's OK
            intervals.append((start_time, end_time))
            condition_seconds += max(0.0, end_time - start_time)

        # estimate total time span for game as last_time - first_time
        total_seconds = float(gdf[time_col].iloc[-1] - gdf[time_col].iloc[0]) if gdf.shape[0] >= 2 else 0.0

        intervals_per_game[gid] = intervals
        totals_per_game[gid] = {'condition_seconds': condition_seconds, 'total_seconds': total_seconds}

        agg_condition_seconds += condition_seconds
        agg_total_seconds += total_seconds

    aggregate = {'condition_seconds': agg_condition_seconds, 'total_seconds': agg_total_seconds}
    return intervals_per_game, totals_per_game, aggregate


if __name__ == '__main__':

    # let's scrape from seasons starting in 2014
    years = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023,
             2024, 2025]
    # Example debug invocation (only when run as a script)
    res = None
    for year in years:
        season = f'{year}{year+1}'
        print(f'\n=== Scraping season {season} ===')
        res = _scrape(
            season=season, team='all',
            out_dir='data', use_cache=True,
            max_games=None, max_workers=8,
            verbose=True,
            # New behavior flags
            save_raw=False,
            save_csv=False,
            save_json=False,
            per_game_files=False,
            process_elaborated=True,
            save_elaborated=True,
            return_feeds=False,
            return_elaborated_df=True,
        )

    # Print a readable summary of the result
    if isinstance(res, dict):
        print('\n_scrape result:')
        sp = res.get('saved_paths') or {}
        print('  saved_paths:')
        for k, v in sp.items():
            print(f'    {k}: {v}')
        raw = res.get('feeds')
        if raw is not None:
            print(f'  feeds: {len(raw)} items')
        edf = res.get('elaborated_df')
        if edf is not None:
            try:
                print(f"  elaborated_df: shape={edf.shape}")
                print(edf.head().to_string())
            except Exception:
                print('  elaborated_df: (unable to display preview)')
    else:
        print('\n_scrape returned:', res)

    # Optionally run a quick season debug (kept for backwards compatibility)
    debug_season = False
    if debug_season:
        df = _season(out_path='/Users/harrisonmcadams/PycharmProjects/new_puck/static')
        print('\nSeason dataframe shape:', getattr(df, 'shape', None))
        if isinstance(df, pd.DataFrame) and not df.empty:
            print(df.head())
        else:
            print('No events found for season')
