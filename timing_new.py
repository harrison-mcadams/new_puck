"""
timing_new.py

New timing utilities that compute game-state intervals using shift-chart
data as the primary source of truth. The module focuses on clarity and
simplicity while providing the necessary building blocks for downstream
analysis (plotting, xG maps, etc.).

Design:
 - Use the NHL shift charts (via `nhl_api.get_shifts` and `parse._shifts`) to
   derive player on-ice intervals.
 - Compute `game_state` intervals by counting skaters per team (skaters =
   players_on_ice - 1 heuristic to account for goalies).
 - Compute `is_net_empty` intervals by detecting goalie shifts and marking
   times where no goalie is on ice for a team.
 - Compute `player_id(s)` intervals using `parse._shifts(..., combine='intersection')`.
 - Provide simple utilities to merge, union, and intersect interval lists.
 - Integrated HTML fallback: _get_shifts_df() and get_shifts_with_html_fallback()
   automatically fall back to HTML parsing when API shifts are empty/minimal.

Public API (minimal, explicit):
 - get_shifts_with_html_fallback(game_id, min_rows_threshold=5)
 - compute_intervals_for_game(game_id, condition)
 - demo_for_export(df, condition)

The code is intentionally simple and documented to be a good foundation
for integration into the rest of the repo.
"""

from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
from pathlib import Path
import logging
import json
import os

import pandas as pd
import numpy as np

# Local imports (these modules are part of the workspace)
import nhl_api
import parse
# Re-use get_game_state helper if available
try:
    from scripts.get_game_state import get_game_state
except Exception:
    get_game_state = None


Interval = Tuple[float, float]


def get_shifts_with_html_fallback(game_id: int, min_rows_threshold: int = 5) -> Dict[str, Any]:
    """Wrapper to get shifts with automatic HTML fallback when API returns empty/minimal data.
    
    This is a convenience function that can be used in place of nhl_api.get_shifts()
    when you want automatic fallback to HTML parsing if the API response is insufficient.
    
    Parameters:
        game_id: NHL game ID
        min_rows_threshold: Minimum number of shifts required to consider API response valid (default 5)
    
    Returns:
        Dict with keys: 'game_id', 'raw', 'all_shifts', 'shifts_by_player'
        Same format as nhl_api.get_shifts() and nhl_api.get_shifts_from_nhl_html()
    """
    try:
        shifts_res = nhl_api.get_shifts(game_id)
    except Exception as e:
        logging.exception('get_shifts_with_html_fallback: get_shifts failed for %s: %s', game_id, e)
        shifts_res = {'game_id': game_id, 'raw': None, 'all_shifts': [], 'shifts_by_player': {}}
    
    # Check if response is empty or minimal
    all_shifts = shifts_res.get('all_shifts', []) if isinstance(shifts_res, dict) else []
    
    if not all_shifts or len(all_shifts) < min_rows_threshold:
        logging.info(
            'get_shifts_with_html_fallback: API shifts %s for game %s (%d rows); using HTML fallback',
            'empty' if not all_shifts else 'minimal', game_id, len(all_shifts)
        )
        try:
            html_res = nhl_api.get_shifts_from_nhl_html(game_id, force_refresh=True, debug=True)
            if html_res and isinstance(html_res, dict) and html_res.get('all_shifts'):
                logging.info(
                    'get_shifts_with_html_fallback: HTML fallback successful for game %s (%d shifts)',
                    game_id, len(html_res.get('all_shifts', []))
                )
                return html_res
            else:
                logging.warning('get_shifts_with_html_fallback: HTML fallback also returned empty for game %s', game_id)
        except Exception as e:
            logging.exception('get_shifts_with_html_fallback: HTML fallback failed for %s: %s', game_id, e)
    
    return shifts_res


def _merge_intervals(intervals: List[Interval], eps: float = 1e-9) -> List[Interval]:
    """Merge sorted or unsorted intervals, coalescing touching/overlapping ones.

    intervals: list of (start, end)
    returns merged list sorted by start
    """
    if not intervals:
        return []
    ints = sorted(((float(s), float(e)) for s, e in intervals if s is not None and e is not None and e > s), key=lambda x: x[0])
    merged: List[Interval] = []
    cur_s, cur_e = ints[0]
    for s, e in ints[1:]:
        if s <= cur_e + eps:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def _intersect_two(a: List[Interval], b: List[Interval], eps: float = 1e-9) -> List[Interval]:
    """Intersect two lists of non-overlapping, sorted intervals.

    Result is sorted list of intersections.
    """
    if not a or not b:
        return []
    i = j = 0
    res: List[Interval] = []
    while i < len(a) and j < len(b):
        s1, e1 = a[i]
        s2, e2 = b[j]
        s = max(s1, s2)
        e = min(e1, e2)
        if e > s + eps:
            res.append((s, e))
        if e1 < e2 - eps:
            i += 1
        elif e2 < e1 - eps:
            j += 1
        else:
            i += 1
            j += 1
    return res


def _union_lists_of_intervals(list_of_lists: List[List[Interval]]) -> List[Interval]:
    """Union multiple interval lists into a merged list.

    Each input list may be unsorted/overlapping; the function merges them
    all into a single coalesced list.
    """
    all_intervals: List[Interval] = []
    for lst in list_of_lists:
        if lst:
            all_intervals.extend(lst)
    return _merge_intervals(all_intervals)


def _intersect_multiple(list_of_lists: List[List[Interval]]) -> List[Interval]:
    """Intersect many lists of intervals (sequential pairwise intersection).

    If any list is empty -> intersection is empty.
    """
    if not list_of_lists:
        return []
    res = list_of_lists[0]
    for lst in list_of_lists[1:]:
        res = _intersect_two(res, lst)
        if not res:
            break
    return res


def _get_shifts_df(game_id: int, min_rows_threshold: int = 5) -> pd.DataFrame:
    """Fetch and parse shift chart for a game into a DataFrame (using parse._shifts).

    Returns DataFrame with columns including 'start_total_seconds' and 'end_total_seconds'.
    This helper is defensive: if parsing fails it will try a force_refresh and
    will save debug payloads to the nhl_api cache directory for offline analysis.
    
    Integrated HTML Fallback:
    If the API response is empty or contains fewer than min_rows_threshold shifts,
    this function will automatically fall back to get_shifts_from_nhl_html.
    
    Parameters:
        game_id: NHL game ID
        min_rows_threshold: Minimum number of shifts required to consider API response valid (default 5)
    """
    try:
        shifts_res = nhl_api.get_shifts(game_id)
    except Exception as e:
        # log and return empty DataFrame
        logging.exception('timing_new._get_shifts_df: get_shifts failed for %s: %s', game_id, e)
        return pd.DataFrame()

    # If get_shifts returns a non-dict, coerce to dict wrapper
    if shifts_res is None:
        shifts_res = {}
    if not isinstance(shifts_res, dict):
        shifts_res = {'raw': shifts_res}

    # Try parsing using parse._shifts; if empty, attempt a force_refresh retry
    try:
        df_shifts = parse._shifts(shifts_res)
    except Exception as e:
        logging.exception('timing_new._get_shifts_df: parse._shifts threw for game %s: %s', game_id, e)
        df_shifts = None

    if df_shifts is None or (hasattr(df_shifts, 'empty') and df_shifts.empty):
        # Attempt a force-refresh from the NHL API
        try:
            shifts_res2 = nhl_api.get_shifts(game_id, force_refresh=True)
        except Exception as e:
            logging.exception('timing_new._get_shifts_df: force_refresh get_shifts failed for %s: %s', game_id, e)
            shifts_res2 = None

        # Save raw payloads for investigation
        try:
            cache_dir = getattr(nhl_api, '_CACHE_DIR', '.cache/nhl_api')
            os.makedirs(cache_dir, exist_ok=True)
            debug_path = os.path.join(cache_dir, f'debug_shifts_{game_id}.json')
            with open(debug_path, 'w', encoding='utf-8') as fh:
                json.dump({'first': shifts_res, 'force': shifts_res2}, fh, default=str)
            logging.info('timing_new._get_shifts_df: wrote debug shifts JSON to %s', debug_path)
        except Exception:
            logging.exception('timing_new._get_shifts_df: failed to write debug shifts JSON for %s', game_id)

        # Try parsing the refreshed payload
        try:
            df_shifts = parse._shifts(shifts_res2) if shifts_res2 is not None else None
        except Exception as e:
            logging.exception('timing_new._get_shifts_df: parse._shifts threw on refreshed payload for %s: %s', game_id, e)
            df_shifts = None
    
    # Check if we have minimal/insufficient data and need HTML fallback
    need_html_fallback = False
    if df_shifts is None or (hasattr(df_shifts, 'empty') and df_shifts.empty):
        need_html_fallback = True
        logging.info('timing_new._get_shifts_df: API shifts empty for game %s; will try HTML fallback', game_id)
    elif len(df_shifts) < min_rows_threshold:
        need_html_fallback = True
        logging.info('timing_new._get_shifts_df: API shifts minimal (%d rows) for game %s; will try HTML fallback', 
                    len(df_shifts), game_id)
    
    # If API response is insufficient, try HTML fallback
    if need_html_fallback:
        try:
            logging.info('timing_new._get_shifts_df: attempting HTML fallback for game %s', game_id)
            html_shifts_res = nhl_api.get_shifts_from_nhl_html(game_id, force_refresh=False, debug=True)
            if html_shifts_res and isinstance(html_shifts_res, dict):
                df_html = parse._shifts(html_shifts_res)
                if df_html is not None and not df_html.empty:
                    logging.info('timing_new._get_shifts_df: HTML fallback successful for game %s (%d shifts)', 
                                game_id, len(df_html))
                    df_shifts = df_html
                else:
                    logging.warning('timing_new._get_shifts_df: HTML fallback returned empty DataFrame for game %s', game_id)
            else:
                logging.warning('timing_new._get_shifts_df: HTML fallback returned invalid result for game %s', game_id)
        except Exception as e:
            logging.exception('timing_new._get_shifts_df: HTML fallback failed for game %s: %s', game_id, e)

    # Normalize result to DataFrame and ensure expected columns exist
    if df_shifts is None:
        return pd.DataFrame()

    # If parse._shifts returned a dict-like structure, coerce to DataFrame
    if isinstance(df_shifts, dict):
        try:
            df_shifts = pd.DataFrame.from_records(df_shifts.get('all_shifts', []) if 'all_shifts' in df_shifts else df_shifts)
        except Exception:
            try:
                df_shifts = pd.DataFrame(df_shifts)
            except Exception:
                return pd.DataFrame()

    if not isinstance(df_shifts, pd.DataFrame):
        try:
            df_shifts = pd.DataFrame(df_shifts)
        except Exception:
            return pd.DataFrame()

    # Try to harmonize column names to expected 'start_total_seconds' and 'end_total_seconds'
    if 'start_total_seconds' not in df_shifts.columns:
        # common alternatives
        for alt in ('start_seconds', 'start_seconds_total', 'start_seconds_parsed', 'start_seconds_rel'):
            if alt in df_shifts.columns:
                df_shifts = df_shifts.rename(columns={alt: 'start_total_seconds'})
                break
        # if still missing, try to compute from 'start_raw' if it's an mm:ss string
        if 'start_total_seconds' not in df_shifts.columns and 'start_raw' in df_shifts.columns:
            def _parse_mmss(x):
                try:
                    if x is None:
                        return None
                    if isinstance(x, (int, float)):
                        return float(x)
                    s = str(x).strip()
                    if ':' in s:
                        mm, ss = s.split(':')
                        return float(int(mm) * 60 + int(ss))
                except Exception:
                    return None
            df_shifts['start_total_seconds'] = df_shifts['start_raw'].apply(_parse_mmss)

    if 'end_total_seconds' not in df_shifts.columns:
        for alt in ('end_seconds', 'end_seconds_total', 'end_seconds_parsed'):
            if alt in df_shifts.columns:
                df_shifts = df_shifts.rename(columns={alt: 'end_total_seconds'})
                break
        if 'end_total_seconds' not in df_shifts.columns and 'end_raw' in df_shifts.columns:
            def _parse_mmss_end(x):
                try:
                    if x is None:
                        return None
                    if isinstance(x, (int, float)):
                        return float(x)
                    s = str(x).strip()
                    if ':' in s:
                        mm, ss = s.split(':')
                        return float(int(mm) * 60 + int(ss))
                except Exception:
                    return None
            df_shifts['end_total_seconds'] = df_shifts['end_raw'].apply(_parse_mmss_end)

    # Coerce numeric types
    try:
        df_shifts['start_total_seconds'] = pd.to_numeric(df_shifts['start_total_seconds'], errors='coerce')
        df_shifts['end_total_seconds'] = pd.to_numeric(df_shifts['end_total_seconds'], errors='coerce')
    except Exception:
        pass

    # Drop rows missing bounds
    try:
        df_shifts = df_shifts.dropna(subset=['start_total_seconds', 'end_total_seconds'])
    except Exception:
        pass

    # Final check
    if df_shifts is None or (hasattr(df_shifts, 'empty') and df_shifts.empty):
        return pd.DataFrame()

    return df_shifts


def _detect_goalies_from_shifts_df(df_shifts: pd.DataFrame) -> List[Any]:
    """(Deprecated) kept for backward compatibility — use _classify_player_roles instead."""
    return _classify_player_roles(df_shifts).get('G', []) if df_shifts is not None else []


def _classify_player_roles(df_shifts: pd.DataFrame, game_length_seconds: Optional[float] = None) -> Dict[str, Any]:
    """Classify players into roles ('G' or 'S') using shift raw metadata and heuristics.

    Improvements over previous implementation:
      - Prefer explicit position fields when present.
      - Compute per-team candidate selection: choose likely goalie(s) per team by
        total time on ice and long single shifts rather than global thresholds.
      - Return conservative defaults (mark as 'S' when unsure).
    """
    out_roles: Dict[str, str] = {}
    stats: Dict[str, Dict[str, float]] = {}

    if df_shifts is None or df_shifts.empty:
        return {'roles': out_roles, 'by_player': stats}

    # Normalize and coerce
    df = df_shifts.copy()
    df['player_id_str'] = df['player_id'].astype(str)
    df['team_id_str'] = df['team_id'].astype(str)

    # gather per-player stats and explicit positions
    explicit_pos_map: Dict[str, str] = {}
    for pid, grp in df.groupby('player_id_str'):
        total = 0.0
        max_shift = 0.0
        n_shifts = 0
        explicit_pos = None
        for _, r in grp.iterrows():
            s = r.get('start_total_seconds')
            e = r.get('end_total_seconds')
            if s is None or e is None:
                continue
            try:
                dur = float(e) - float(s)
                if dur <= 0:
                    continue
            except Exception:
                continue
            total += dur
            max_shift = max(max_shift, dur)
            n_shifts += 1

            raw = r.get('raw') or {}
            try:
                if isinstance(raw, dict):
                    p = raw.get('player') or raw.get('person') or None
                    cand = None
                    if isinstance(p, dict):
                        cand = p.get('primaryPosition') or p.get('position') or p.get('pos')
                    cand = cand or raw.get('position') or raw.get('primaryPosition') or raw.get('pos')
                    if cand:
                        explicit_pos = str(cand).upper()
            except Exception:
                explicit_pos = explicit_pos

        stats[pid] = {'total_seconds': float(total), 'max_shift': float(max_shift), 'n_shifts': int(n_shifts)}
        if explicit_pos is not None:
            explicit_pos_map[pid] = explicit_pos

    # estimate game_length if not provided
    if game_length_seconds is None:
        try:
            starts = pd.to_numeric(df['start_total_seconds'], errors='coerce').dropna()
            ends = pd.to_numeric(df['end_total_seconds'], errors='coerce').dropna()
            if len(starts) and len(ends):
                game_length_seconds = float(max(ends.max(), starts.max()) - min(starts.min(), ends.min()))
            else:
                game_length_seconds = None
        except Exception:
            game_length_seconds = None

    # thresholds
    ABSOLUTE_GOALIE_SECONDS = 1500.0  # 25 minutes
    LONG_SHIFT_THRESHOLD = 900.0  # 15 minutes
    FRACTION_OF_GAME_CUTOFF = 0.3  # 30% of game time

    # Build per-team player lists to select goalie candidates
    per_team_players: Dict[str, List[str]] = defaultdict(list)
    for pid in stats.keys():
        # discover team for player via first row
        try:
            t = df.loc[df['player_id_str'] == pid, 'team_id_str'].iloc[0]
        except Exception:
            t = 'UNK'
        per_team_players[t].append(pid)

    # First, assign explicit positions
    for pid, pos in explicit_pos_map.items():
        try:
            s = pos.upper()
            if s.startswith('G') or 'GOAL' in s:
                out_roles[pid] = 'G'
            else:
                out_roles[pid] = 'S'
        except Exception:
            out_roles[pid] = 'S'

    # For each team, if no explicit goalie found, pick candidate(s)
    for team, pids in per_team_players.items():
        # check how many already assigned G in this team
        assigned_g = [p for p in pids if out_roles.get(p) == 'G']
        if assigned_g:
            # ensure any remaining unassigned are S
            for p in pids:
                if p not in out_roles:
                    out_roles[p] = 'S'
            continue

        # compute totals for ranking
        candidates = []
        for p in pids:
            st = stats.get(p, {})
            tot = st.get('total_seconds', 0.0)
            maxs = st.get('max_shift', 0.0)
            candidates.append((p, float(tot), float(maxs)))
        # sort by total seconds descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        if not candidates:
            continue
        top_pid, top_total, top_max = candidates[0]
        # determine if top is sufficiently goalie-like
        is_goalie = False
        if top_total >= ABSOLUTE_GOALIE_SECONDS:
            is_goalie = True
        elif game_length_seconds is not None and top_total >= FRACTION_OF_GAME_CUTOFF * game_length_seconds and top_max >= LONG_SHIFT_THRESHOLD:
            is_goalie = True
        elif top_max >= 1800.0 and candidates[0][1] > 0:
            # very long single shift
            is_goalie = True
        # assign
        if is_goalie:
            out_roles[top_pid] = 'G'
            # optionally assign backup if second is substantial fraction
            if len(candidates) > 1 and candidates[1][1] >= 0.5 * candidates[0][1]:
                out_roles[candidates[1][0]] = 'G'
            # mark all others as skaters
            for p in pids:
                if p not in out_roles:
                    out_roles[p] = 'S'
        else:
            # fallback: mark all as skaters
            for p in pids:
                if p not in out_roles:
                    out_roles[p] = 'S'

    # ensure every player in stats has a role
    for pid in stats.keys():
        if pid not in out_roles:
            out_roles[pid] = 'S'

    return {'roles': out_roles, 'by_player': stats}


def _intervals_from_goalie_presence(df_shifts: pd.DataFrame, team_id_map: Dict[Any, Any]) -> Dict[str, List[Interval]]:
    """Compute intervals for goalie presence (per team key 'home' and 'away').

    Now uses explicit player-role classification to robustly identify goalie shifts.
    """
    out = {'home': [], 'away': []}
    if df_shifts is None or df_shifts.empty:
        return out

    # classify players
    cls = _classify_player_roles(df_shifts)
    roles = cls.get('roles', {})
    goalie_pids = {k for k, v in roles.items() if v == 'G'}

    # Fallback: if classification found no goalies, try explicit position fields in raw shift rows
    if not goalie_pids:
        cand = set()
        for _, r in df_shifts.iterrows():
            raw = r.get('raw') or {}
            pos = None
            try:
                if isinstance(raw, dict):
                    p = raw.get('player') or raw.get('person') or None
                    if isinstance(p, dict):
                        pos = p.get('primaryPosition') or p.get('position')
                    pos = pos or raw.get('position') or raw.get('primaryPosition')
            except Exception:
                pos = None
            if pos:
                try:
                    s = str(pos).upper()
                    if s.startswith('G') or 'GOAL' in s:
                        pid = r.get('player_id')
                        if pid is not None:
                            cand.add(str(pid))
                except Exception:
                    continue
        if cand:
            goalie_pids = cand

    if not goalie_pids:
        # cannot detect goalies robustly; return empty presence intervals
        return out

    # build events for goalie shifts only grouped by team
    events = []  # (time, 'start', 'end', team_label)
    # If team_id_map doesn't include concrete ids, derive a local mapping from df_shifts
    local_map = {}
    try:
        hid = team_id_map.get('home_id') if isinstance(team_id_map, dict) else None
        aid = team_id_map.get('away_id') if isinstance(team_id_map, dict) else None
    except Exception:
        hid = aid = None
    if hid in (None, '') or aid in (None, ''):
        # derive from df_shifts order: first unique team_id -> home, second -> away
        seen = []
        for _, r in df_shifts.iterrows():
            tid = r.get('team_id')
            if tid is None:
                continue
            if tid not in seen:
                seen.append(tid)
            if len(seen) >= 2:
                break
        if seen:
            local_map[str(seen[0])] = 'home'
            if len(seen) > 1:
                local_map[str(seen[1])] = 'away'
    # else local_map remains empty
    for _, r in df_shifts.iterrows():
        pid = str(r.get('player_id'))
        if pid not in goalie_pids:
            continue
        tid = r.get('team_id')
        s = r.get('start_total_seconds')
        e = r.get('end_total_seconds')
        if s is None or e is None:
            continue
        # map team id to 'home'/'away' using provided map
        team_label = None
        try:
            if isinstance(team_id_map, dict) and team_id_map.get('home_id') is not None and str(tid) == str(team_id_map.get('home_id')):
                team_label = 'home'
            elif isinstance(team_id_map, dict) and team_id_map.get('away_id') is not None and str(tid) == str(team_id_map.get('away_id')):
                team_label = 'away'
            elif str(tid) in local_map:
                team_label = local_map.get(str(tid))
            else:
                # fallback: if only one team seen assign to home, otherwise assign away
                team_label = 'home' if not local_map else 'away'
        except Exception:
            team_label = 'home'
        events.append((float(s), 'start', team_label))
        events.append((float(e), 'end', team_label))

    if not events:
        return out

    # sort events (end before start at same time)
    events.sort(key=lambda x: (x[0], 0 if x[1] == 'end' else 1))

    active = {'home': 0, 'away': 0}
    prev_t = events[0][0]
    for t, typ, team_label in events:
        if t > prev_t:
            # record presence interval for teams with active >0
            for tl in ('home', 'away'):
                if active.get(tl, 0) > 0:
                    out[tl].append((float(prev_t), float(t)))
            prev_t = t
        # apply event
        if typ == 'start':
            active[team_label] = active.get(team_label, 0) + 1
        else:
            active[team_label] = max(0, active.get(team_label, 0) - 1)

    # merged
    out['home'] = _merge_intervals(out['home'])
    out['away'] = _merge_intervals(out['away'])
    return out


def debug_print_game_intervals(res: Dict[str, Any], max_intervals_display: int = 10):
    """Pretty-print the intervals result produced by compute_intervals_for_game.

    Parameters
    - res: the dict returned by compute_intervals_for_game
    - max_intervals_display: limit how many intervals to print per condition
    """
    if not res:
        print("<no timing result>")
        return
    gid = res.get('game_id')
    print(f"\nTiming summary for game {gid}:")
    print("-" * 60)
    ipc = res.get('intervals_per_condition', {})
    pooled = res.get('pooled_seconds_per_condition', {})
    for key, ivals in ipc.items():
        total = pooled.get(key, 0.0)
        print(f"Condition: {key}  (intervals={len(ivals)}, total_seconds={total:.1f})")
        if not ivals:
            print("  (no intervals)")
            continue
        # print first few intervals
        for i, (s, e) in enumerate(ivals[:max_intervals_display]):
            print(f"  {i+1:2d}. {s:.1f} -> {e:.1f}   ({(e-s):.1f}s)")
        if len(ivals) > max_intervals_display:
            print(f"  ... (+{len(ivals)-max_intervals_display} more intervals)")
    # intersection
    inter = res.get('intersection_intervals', [])
    inter_sec = res.get('intersection_seconds', 0.0)
    print("-" * 60)
    print(f"Intersection intervals (count={len(inter)}, total_seconds={inter_sec:.1f}):")
    if inter:
        for i, (s, e) in enumerate(inter[:max_intervals_display]):
            print(f"  {i+1:2d}. {s:.1f} -> {e:.1f}   ({(e-s):.1f}s)")
        if len(inter) > max_intervals_display:
            print(f"  ... (+{len(inter)-max_intervals_display} more intervals)")
    else:
        print("  (none)")
    print(f"Total observed seconds: {res.get('total_observed_seconds', 0.0):.1f}")
    print("-" * 60)



def load_season_df(season: str = '20252026', data_dir: str = 'data') -> pd.DataFrame:
    """Load a season-level CSV from likely locations and return DataFrame.

    The function tries several conventional locations then falls back to a
    recursive search under `data/`. It prints which file was loaded (if
    any) and returns an empty DataFrame when none is found.
    """
    candidates = [
        Path(data_dir) / season / f"{season}_df.csv",
        Path(data_dir) / season / f"{season}.csv",
        Path(data_dir) / f"{season}_df.csv",
        Path(data_dir) / f"{season}.csv",
    ]
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_csv(p)
                print(f"Loaded season dataframe from {p} -> shape={df.shape}", flush=True)
                return df
            except Exception as e:
                print(f"Failed to read {p}: {e}", flush=True)
                continue
    # fallback: search under data/ for matching filenames
    data_dir_p = Path(data_dir)
    if data_dir_p.exists():
        found = list(data_dir_p.rglob(f'*{season}*.csv'))
        if found:
            try:
                df = pd.read_csv(found[0])
                print(f"Loaded season dataframe from {found[0]} -> shape={df.shape}", flush=True)
                return df
            except Exception:
                pass
    print("No season CSV found among candidates; returning empty DataFrame", flush=True)
    return pd.DataFrame()


def select_team_game(df: pd.DataFrame, team: str) -> Optional[Any]:
    """Return reasonable game id(s) for `team`.

    This helper is intentionally forgiving: it will attempt to match the
    provided `team` string against both team abbreviations (home_abb /
    away_abb) and numeric IDs (home_id / away_id). It returns:
      - None when no match is found.
      - A single game id (int or str) when only one found in fallback.
      - A list of game ids (preferably numeric and sorted) when multiple
        matching games are present.

    The function is defensive against missing columns and mixed types.
    """
    if df is None or df.empty:
        return None
    t = str(team).strip().upper()

    # defensive column access
    home_abb = df.get('home_abb')
    away_abb = df.get('away_abb')
    home_id = df.get('home_id')
    away_id = df.get('away_id')

    mask = pd.Series(False, index=df.index)
    try:
        if home_abb is not None:
            mask = mask | (home_abb.astype(str).str.upper() == t)
    except Exception:
        pass
    try:
        if away_abb is not None:
            mask = mask | (away_abb.astype(str).str.upper() == t)
    except Exception:
        pass

    # numeric id match
    tid = None
    try:
        tid = int(t)
    except Exception:
        tid = None
    if tid is not None:
        try:
            if home_id is not None:
                mask = mask | (home_id.astype(str) == str(tid))
        except Exception:
            pass
        try:
            if away_id is not None:
                mask = mask | (away_id.astype(str) == str(tid))
        except Exception:
            pass

    if not mask.any():
        return None

    if 'game_id' in df.columns:
        try:
            games = df.loc[mask, 'game_id'].dropna().unique().tolist()
        except Exception:
            games = []
    else:
        games = []

    if not games:
        try:
            last = df.loc[mask].tail(1)
            if 'game_id' in last.columns:
                return last['game_id'].iloc[0]
        except Exception:
            return None

    # try returning numeric-sorted game ids when possible
    try:
        numeric_games = sorted([int(g) for g in games])
        return numeric_games
    except Exception:
        return games


def compute_intervals_for_game(game_id: int, condition: Dict[str, Any],
 verbose: bool = False, net_empty_mode: str = 'either') -> Dict[str, Any]:
    """Compute per-condition intervals for a single game using shifts as source.

    condition: dict of keys -> values where values are scalars or lists.
      - supported keys (initial): 'game_state', 'is_net_empty', 'player_id' or 'player_ids'
    Returns structure:
      {
        'game_id': <id>,
        'intervals_per_condition': { key: [(s,e), ...] },
        'pooled_seconds_per_condition': { key: seconds },
        'intersection_intervals': [(s,e), ...],
        'intersection_seconds': float,
        'total_observed_seconds': float
      }
    """
    # normalize condition values to lists
    cond_norm: Dict[str, List[Any]] = {}
    for k, v in (condition or {}).items():
        if v is None:
            continue
        if isinstance(v, (list, tuple, set)):
            cond_norm[k] = list(v)
        else:
            cond_norm[k] = [v]

    # get shifts df
    df_shifts = _get_shifts_df(int(game_id))

    # obtain game metadata from play-by-play to map home/away ids
    feed = nhl_api.get_game_feed(int(game_id)) or {}
    home_id = None; away_id = None
    try:
        h = feed.get('homeTeam') or feed.get('home') or {}
        a = feed.get('awayTeam') or feed.get('away') or {}
        if isinstance(h, dict):
            home_id = h.get('id') or h.get('teamId')
        if isinstance(a, dict):
            away_id = a.get('id') or a.get('teamId')
    except Exception:
        pass

    team_id_map = {'home_id': home_id, 'away_id': away_id}

    intervals_per_condition: Dict[str, List[Interval]] = {}
    pooled_seconds_per_condition: Dict[str, float] = {}

    # Precompute some helpers
    # 1) game_state timeline via get_game_state if available (preferred)
    gs_timeline: List[Dict[str, Any]] = []
    if get_game_state is not None:
        try:
            gs_df, _ = get_game_state(game_id, condition=None, return_df=True, df_shifts=df_shifts)[:2]
            # get_game_state returns (df_intervals, filtered) when return_df=True
            # sometimes the function returns (None, None)
            if isinstance(gs_df, pd.DataFrame) and not gs_df.empty:
                gs_timeline = gs_df.to_dict('records')
        except Exception:
            gs_timeline = []

    # 2) goalie presence timeline
    goalie_presence = _intervals_from_goalie_presence(df_shifts, team_id_map)

    # compute total observed seconds for game (use shifts span if available)
    if df_shifts is not None and not df_shifts.empty:
        starts = pd.to_numeric(df_shifts['start_total_seconds'], errors='coerce').dropna().astype(float)
        ends = pd.to_numeric(df_shifts['end_total_seconds'], errors='coerce').dropna().astype(float)
        if len(starts) and len(ends):
            total_observed = float(max(ends.max(), starts.max()) - min(starts.min(), ends.min()))
        else:
            total_observed = 0.0
    else:
        total_observed = 0.0


    # Helper to flip game state string (e.g. "5v4" -> "4v5")
    def _flip_game_state(s):
        try:
            if 'v' in s:
                parts = s.split('v')
                if len(parts) == 2:
                    return f"{parts[1]}v{parts[0]}"
        except Exception:
            pass
        return s

    # Check if we need to flip game states (if team is specified and is AWAY)
    flip_states = False
    home_abb = feed.get('homeTeam', {}).get('abbrev') or feed.get('home', {}).get('abbrev')
    away_abb = feed.get('awayTeam', {}).get('abbrev') or feed.get('away', {}).get('abbrev')

    if 'team' in cond_norm:
        # Check if the requested team matches the away team
        # We need to be careful: cond_norm['team'] is a list.
        # Usually it's a single team.
        try:
            req_team = cond_norm['team'][0]
            # Check against away_id / away_abb
            is_away = False
            if away_id is not None and str(req_team) == str(away_id):
                is_away = True
            elif away_abb is not None and str(req_team).upper() == str(away_abb).upper():
                is_away = True
            
            if is_away:
                flip_states = True
        except Exception:
            pass

    # For each condition key compute intervals
    for key, vals in cond_norm.items():
        key_intervals_all: List[Interval] = []
        if key == 'game_state':
            # For each requested state, collect intervals where label == state
            if gs_timeline:
                for state in vals:
                    # If we are looking for relative state for Away team, flip the state string
                    # e.g. User asks for "5v4" (Team 5, Opp 4). If Team is Away, we look for "4v5" (Home 4, Away 5).
                    target_state = _flip_game_state(str(state)) if flip_states else str(state)

                    for r in gs_timeline:
                        try:
                            if str(r.get('label')) == target_state:
                                s = float(r.get('start'))
                                e = float(r.get('end'))
                                if e > s:
                                    key_intervals_all.append((s, e))
                        except Exception:
                            continue
            else:
                # fallback: attempt to compute skater counts directly from shifts (if gs_timeline missing)
                # Build intervals by counting active players per team similar to get_game_state
                # Reuse code from scripts.get_game_state logic in a compact form
                if df_shifts is None or df_shifts.empty:
                    key_intervals_all = []
                else:
                    # build events
                    evs = []
                    for _, r in df_shifts.iterrows():
                        s = r.get('start_total_seconds'); e = r.get('end_total_seconds'); tid = r.get('team_id'); pid = r.get('player_id')
                        if s is None or e is None:
                            continue
                        evs.append((float(s), 'start', tid, pid))
                        evs.append((float(e), 'end', tid, pid))
                    if evs:
                        evs.sort(key=lambda x: (x[0], 0 if x[1] == 'end' else 1))
                        # classify players once so we can exclude goalies when counting
                        roles = _classify_player_roles(df_shifts).get('roles', {})
                        active = defaultdict(set)
                        prev_t = evs[0][0]
                        i = 0
                        while i < len(evs):
                            t0 = evs[i][0]
                            # The interval [prev_t, t0) reflects the current on-ice composition
                            if t0 > prev_t:
                                # helper to count skaters for a team excluding goalies
                                def count_skaters_for_team(tid_val):
                                    sset = active.get(str(tid_val), set()) if tid_val is not None else set()
                                    if not sset:
                                        return 0
                                    return sum(1 for pid in sset if roles.get(str(pid), 'S') != 'G')

                                if home_id is not None:
                                    home_count = count_skaters_for_team(home_id)
                                else:
                                    home_count = sum(count_skaters_for_team(k) for k in active.keys())
                                if away_id is not None:
                                    away_count = count_skaters_for_team(away_id)
                                else:
                                    away_count = 0

                                # evaluate requested states against this interval
                                for state in vals:
                                    try:
                                        # Flip state if needed
                                        target_state = _flip_game_state(str(state)) if flip_states else str(state)

                                        if target_state == f"{home_count}v{away_count}":
                                            key_intervals_all.append((prev_t, t0))
                                    except Exception:
                                        continue
                                prev_t = t0

                            # apply all events at this timestamp
                            while i < len(evs) and evs[i][0] == t0:
                                _, typ, tid, pid = evs[i]
                                key_tid = str(tid)
                                if typ == 'start':
                                    active[key_tid].add(str(pid))
                                else:
                                    active[key_tid].discard(str(pid))
                                i += 1
        elif key == 'is_net_empty':
            # Simple, robust handling of goalie-presence -> net-empty logic.
            # goalie_presence contains lists of (s,e) for 'home' and 'away'.
            try:
                intervals = _build_is_net_empty_intervals(goalie_presence, vals, net_empty_mode, verbose)
                key_intervals_all.extend(intervals)
            except Exception:
                key_intervals_all = []
        elif key in ('player_id', 'player_ids'):
            # For player(s) use parse._shifts with combine='intersection' to compute
            # intervals when all provided players are simultaneously on ice.
            pids = []
            if key == 'player_id':
                pids = [vals[0]]
            else:
                pids = vals
            try:
                # Use robust fetch with fallback
                shifts_payload = get_shifts_with_html_fallback(game_id)
                shift_df = parse._shifts(shifts_payload, player_ids=pids, combine='intersection')
                for _, r in shift_df.iterrows():
                    s = r.get('start_total_seconds'); e = r.get('end_total_seconds')
                    if s is not None and e is not None and e > s:
                        key_intervals_all.append((float(s), float(e)))
            except Exception:
                key_intervals_all = []
            except Exception:
                key_intervals_all = []
        elif key == 'team':
            # Team filtering is handled at game selection level.
            # If we are processing this game, it matches the team.
            # Return full observed duration.
            if total_observed > 0:
                key_intervals_all = [(0.0, total_observed)]
            else:
                key_intervals_all = []
        else:
            # Unsupported condition -> empty
            key_intervals_all = []

        # union all intervals for the condition key
        merged = _merge_intervals(key_intervals_all)
        intervals_per_condition[str(key)] = merged
        pooled_seconds_per_condition[str(key)] = sum((e - s) for s, e in merged) if merged else 0.0

    # intersect across condition keys to find times where ALL conditions hold
    intersection_list_of_lists = [intervals_per_condition[k] for k in intervals_per_condition.keys() if intervals_per_condition.get(k)]
    if intersection_list_of_lists:
        inter = _intersect_multiple(intersection_list_of_lists)
    else:
        inter = []
    intersection_seconds = sum((e - s) for s, e in inter) if inter else 0.0

    # compute total observed seconds for game (use shifts span if available)
    # (Moved up)

    result = {
        'game_id': int(game_id),
        'intervals_per_condition': intervals_per_condition,
        'pooled_seconds_per_condition': pooled_seconds_per_condition,
        'intersection_intervals': inter,
        'intersection_seconds': float(intersection_seconds),
        'total_observed_seconds': float(total_observed),
    }

    # if verbose, print a tidy debug summary
    if verbose:
        try:
            debug_print_game_intervals(result)
        except Exception:
            # don't allow debug printing to break function
            print("(debug print failed)")

    return result


def demo_for_export(df: pd.DataFrame, condition: Dict[str, Any], verbose: bool = False, net_empty_mode: str = 'either') -> Dict[str, Any]:
    """Compute per-game timing information for all games referenced in `df`.

    df: a DataFrame that contains a 'game_id' column (events-level or season-level).
    condition: same shape as timing.intervals_for_conditions (dict of key->values).

    Returns a dict with 'per_game' mapping game_id -> compute_intervals_for_game(...) output.
    """
    if df is None or df.empty:
        return {'per_game': {}, 'aggregate': {}}

    if 'game_id' not in df.columns:
        return {'per_game': {}, 'aggregate': {}}

    gids = pd.unique(df['game_id'].dropna().astype(int)).tolist()

    # If caller provided a team in the condition, filter the input df so we only
    # process games where that team played. Accept either numeric team id or
    # abbreviation (e.g. 'PHI'). This reduces unnecessary work and matches the
    # user's request to filter by team prior to per-game processing.
    df_for_gids = df
    team_filter = None
    if isinstance(condition, dict) and 'team' in condition and condition.get('team') is not None:
        team_filter = condition.get('team')
        try:
            # attempt integer team id
            tid = int(team_filter) if (isinstance(team_filter, (int, float)) or (isinstance(team_filter, str) and str(team_filter).strip().isdigit())) else None
        except Exception:
            tid = None
        if tid is not None:
            try:
                df_for_gids = df.loc[(df.get('home_id').astype(str) == str(tid)) | (df.get('away_id').astype(str) == str(tid))].copy()
            except Exception:
                # robust fallback if columns missing
                try:
                    df_for_gids = df.loc[(df.get('home_id') == tid) | (df.get('away_id') == tid)].copy()
                except Exception:
                    df_for_gids = df
        else:
            tstr = str(team_filter).strip().upper()
            try:
                df_for_gids = df.loc[(df.get('home_abb', pd.Series(dtype=object)).astype(str).str.upper() == tstr) | (df.get('away_abb', pd.Series(dtype=object)).astype(str).str.upper() == tstr)].copy()
            except Exception:
                df_for_gids = df

    gids = pd.unique(df_for_gids['game_id'].dropna().astype(int)).tolist()

    per_game: Dict[Any, Any] = {}
    # Prepare tasks for parallel execution
    tasks = []
    for gid in gids:
        try:
            # Build a per-game condition dict so we can enforce a team selection.
            if isinstance(condition, dict):
                cond_for_game = condition.copy()
            else:
                cond_for_game = {} if condition is None else dict(condition)

            # If no explicit team provided, attempt to infer the game's home team
            if 'team' not in cond_for_game or cond_for_game.get('team') is None:
                try:
                    # Subset input df to this game_id
                    df_game_rows = df.loc[df['game_id'].astype(int) == int(gid)]
                    home_team_val = None
                    if 'home_abb' in df_game_rows.columns:
                        vals = pd.unique(df_game_rows['home_abb'].dropna().astype(str))
                        if len(vals) > 0:
                            home_team_val = vals[0]
                    if home_team_val is None and 'home_id' in df_game_rows.columns:
                        vals2 = pd.unique(df_game_rows['home_id'].dropna())
                        if len(vals2) > 0:
                            home_team_val = vals2[0]
                    if home_team_val is not None:
                        cond_for_game['team'] = home_team_val
                except Exception:
                    pass
            tasks.append((int(gid), cond_for_game))
        except Exception:
            continue

    # Execute in parallel
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import os
    max_workers = min(8, os.cpu_count() or 1)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_gid = {
            executor.submit(compute_intervals_for_game, gid, cond, verbose, net_empty_mode): gid
            for gid, cond in tasks
        }
        
        for future in as_completed(future_to_gid):
            gid = future_to_gid[future]
            try:
                res = future.result()
                per_game[gid] = res
            except Exception as e:
                if verbose:
                    logging.exception('demo_for_export: failed for game %s: %s', gid, e)

    # basic aggregate (sum intersection_seconds across games)
    agg = {'intersection_seconds_total': 0.0}
    agg['intersection_seconds_total'] = sum((v.get('intersection_seconds') or 0.0) for v in per_game.values())

    return {'per_game': per_game, 'aggregate': agg}


def intervals_to_json(res: Dict[str, Any]) -> Dict[str, Any]:
    """Convert the result of compute_intervals_for_game to a JSON-serializable dict."""
    if not res:
        return {}
    out = {
        'game_id': int(res.get('game_id')),
        'intervals_per_condition': {},
        'pooled_seconds_per_condition': {},
        'intersection_intervals': [],
        'intersection_seconds': float(res.get('intersection_seconds', 0.0)),
        'total_observed_seconds': float(res.get('total_observed_seconds', 0.0)),
    }
    ipc = res.get('intervals_per_condition', {})
    for k, ivals in ipc.items():
        out['intervals_per_condition'][k] = [[float(s), float(e)] for s, e in (ivals or [])]
    pooled = res.get('pooled_seconds_per_condition', {})
    for k, v in pooled.items():
        out['pooled_seconds_per_condition'][k] = float(v)
    inter = res.get('intersection_intervals', [])
    out['intersection_intervals'] = [[float(s), float(e)] for s, e in (inter or [])]
    return out


def _complement_intervals(intervals: List[Interval], start: float, end: float) -> List[Interval]:
    """Return the complement of `intervals` within [start, end].

    intervals: list of (s,e); may be unsorted/overlapping. The result is a
    sorted list of disjoint intervals covering regions inside [start,end]
    not covered by `intervals`.
    """
    if intervals is None:
        return []
    # Merge input to get disjoint, ordered intervals
    merged = _merge_intervals(intervals)
    out: List[Interval] = []
    cur = float(start)
    for s, e in merged:
        if e <= cur:
            continue
        if s > cur:
            out.append((cur, min(s, end)))
        cur = max(cur, e)
        if cur >= end:
            break
    if cur < end:
        out.append((cur, float(end)))
    return out

# Also provide a short public alias used elsewhere in the file (the code calls `complement(...)`)
def complement(intervals: List[Interval], start: float, end: float) -> List[Interval]:
    return _complement_intervals(intervals, start, end)


def _compute_observed_bounds(combined_pres: List[Interval]) -> Tuple[float, float]:
    """Compute observed min and max times from a list of presence intervals.

    Returns (min_t, max_t). Raises ValueError if bounds cannot be determined.
    """
    if not combined_pres:
        raise ValueError('no presence intervals')
    bounds = []
    for s_e in combined_pres:
        try:
            s, e = s_e
            if s is None or e is None:
                continue
            bounds.append((float(s), float(e)))
        except Exception:
            continue
    if not bounds:
        raise ValueError('no numeric bounds')
    min_t = min(s for s, _ in bounds)
    max_t = max(e for _, e in bounds)
    return min_t, max_t


def _build_is_net_empty_intervals(goalie_presence: Dict[str, List[Interval]],
                                  vals: List[Any],
                                  net_empty_mode: str = 'either',
                                  verbose: bool = False) -> List[Interval]:
    """Build intervals for is_net_empty condition from goalie_presence.

    goalie_presence: {'home': [(s,e),...], 'away': [(s,e), ...]}
    vals: list of requested values (0 or 1)
    net_empty_mode: 'either' or 'both'
    Returns a list of (s,e) intervals (unmerged); caller will merge later.
    """
    home_pres = goalie_presence.get('home', []) or []
    away_pres = goalie_presence.get('away', []) or []
    combined = list(home_pres) + list(away_pres)
    if not combined:
        return []
    try:
        min_t, max_t = _compute_observed_bounds(combined)
    except Exception:
        # fall back to a safe observed span
        min_t, max_t = 0.0, 0.0

    # complements (per-side empty)
    home_empty = complement(home_pres, min_t, max_t)
    away_empty = complement(away_pres, min_t, max_t)

    # union of presences and intersection
    union_pres = _merge_intervals(list(home_pres) + list(away_pres))
    both_empty = complement(union_pres, min_t, max_t)
    both_present = _intersect_multiple([home_pres, away_pres])

    if verbose:
        try:
            def _sample(lst, n=5):
                try:
                    return lst[:n]
                except Exception:
                    return list(lst)[:n]
            print(f"[debug][is_net_empty] home_pres count={len(home_pres)} sample={_sample(home_pres)}")
            print(f"[debug][is_net_empty] away_pres count={len(away_pres)} sample={_sample(away_pres)}")
            print(f"[debug][is_net_empty] both_present count={len(both_present)} sample={_sample(both_present)}")
            print(f"[debug][is_net_empty] both_empty count={len(both_empty)} sample={_sample(both_empty)}")
        except Exception:
            pass

    out_intervals: List[Interval] = []
    for v in vals:
        try:
            vi = int(v)
        except Exception:
            continue
        if vi == 1:
            if net_empty_mode == 'either':
                # either goalie absent -> union of per-side empties
                out_intervals.extend(_union_lists_of_intervals([home_empty, away_empty]))
            else:
                # both mode -> neither goalie present
                out_intervals.extend(both_empty)
        else:
            if net_empty_mode == 'either':
                # prefer both present, fallback to union_pres
                if both_present:
                    out_intervals.extend(both_present)
                else:
                    if verbose:
                        try:
                            print('[debug][is_net_empty] both_present empty; falling back to union of presences')
                        except Exception:
                            pass
                    out_intervals.extend(union_pres)
            else:
                out_intervals.extend(both_present)
    return out_intervals



def add_game_state_relative_column(df: pd.DataFrame, team_val: Any) -> pd.DataFrame:
    """Add a 'game_state_relative_to_team' column to the DataFrame.

    This is used to validate game states from the perspective of a specific team.
    If the row's event owner matches `team_val`, the state is kept as is.
    If the row's event owner is the opponent, the state is flipped (e.g. 5v4 -> 4v5).
    """
    if df is None or df.empty:
        return df
    
    df_out = df.copy()
    if 'game_state' not in df_out.columns:
        df_out['game_state_relative_to_team'] = None
        return df_out

    if team_val is None:
        df_out['game_state_relative_to_team'] = df_out['game_state']
        return df_out

    # Helper to flip state string
    def _flip(s):
        if not isinstance(s, str):
            return s
        if 'v' in s:
            parts = s.split('v')
            if len(parts) == 2:
                return f"{parts[1]}v{parts[0]}"
        return s

    # Determine match for each row
    tstr = str(team_val).strip()
    try:
        tid = int(tstr)
    except:
        tid = None
    tupper = tstr.upper()

    def _get_rel_state(row):
        gs = row.get('game_state')
        if pd.isna(gs):
            return gs
        
        # Check if event owner is the selected team
        is_team = False
        try:
            if tid is not None:
                is_team = (str(row.get('team_id')) == str(tid))
            else:
                # check abbs
                if row.get('home_abb') and str(row.get('home_abb')).upper() == tupper:
                    is_team = (str(row.get('team_id')) == str(row.get('home_id')))
                elif row.get('away_abb') and str(row.get('away_abb')).upper() == tupper:
                    is_team = (str(row.get('team_id')) == str(row.get('away_id')))
        except:
            pass
        
        if is_team:
            return gs
        else:
            return _flip(str(gs))

    df_out['game_state_relative_to_team'] = df_out.apply(_get_rel_state, axis=1)
    return df_out


if __name__ == '__main__':
    import argparse, sys
    parser = argparse.ArgumentParser(description='timing_new demo')
    parser.add_argument('--game_id', '-g', type=int, default=None)
    parser.add_argument('--team', '-t', type=str, default=None)
    parser.add_argument('--verbose', action='store_true', help='print detailed debug output')
    parser.add_argument('--out', '-o', type=str, default=None, help='write JSON debug output to file')
    args = parser.parse_args()
    if args.game_id is None:
        try:
            gid = nhl_api.get_game_id(team='PHI')
            print('Using most recent PHI game ->', gid)
        except Exception:
            print('Failed to find recent PHI game; exit')
            sys.exit(1)
    else:
        gid = args.game_id
    cond = {'game_state': ['5v5'], 'is_net_empty': [0]}
    out = compute_intervals_for_game(gid, cond, verbose=args.verbose, net_empty_mode='either')
    if args.verbose:
        debug_print_game_intervals(out)
    else:
        print('Result keys:', list(out.keys()))
    if args.out:
        try:
            with open(args.out, 'w', encoding='utf-8') as fh:
                json.dump(intervals_to_json(out), fh, indent=2)
            print(f'Wrote JSON debug output to {args.out}')
        except Exception as e:
            print('Failed to write JSON output:', e)
