"""
timing.py

Refactored timing utilities (clean, documented, and backward-compatible).

This module provides a small collection of helpers useful for computing
contiguous time intervals in game event DataFrames and for aggregating
those intervals across games for simple analyses (used by plotting and
xG mapping code elsewhere in the repo).

Design goals:
  - Readability: small focused helpers with clear docstrings.
  - Backwards compatibility: function signatures and return structures
    preserved so callers continue to work unchanged.
  - Minimal third-party dependencies: only pandas and numpy.

Quick example:
  from timing import load_season_df, demo_for_export
  df = load_season_df('20252026')
  # Run demo on all games in the season (no 'team' provided):
  res = demo_for_export(df, condition={'game_state':['5v5']})

Note: demo_for_export will attempt to fetch a raw game feed with
`nhl_api.get_game_feed` for each game id when available; when running
large batches you may want to pass a DataFrame already containing the
full game events to avoid network calls.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

# Public API exported by this module. This keeps imports explicit and
# documents the functions intended for external use.
__all__ = [
    'load_season_df', 'select_team_game', 'intervals_for_condition',
    'intervals_for_conditions', 'add_game_state_relative_column', 'demo_for_export'
]


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


def intervals_for_condition(
    df: pd.DataFrame,
    condition: Dict[str, Any],
    time_col: str = 'total_time_elapsed_seconds',
    verbose: bool = False,
) -> Tuple[List[Tuple[float, float]], float, float]:
    """Compute contiguous time intervals where `condition` holds.

    Implementation notes:
      - Coerces `time_col` to numeric and drops rows without valid time.
      - Sorts events by time so that intervals are contiguous in game-time.
      - Treats a True run in the boolean mask as starting at the time of
        the first row in the run and ending at the time of the next row
        after the run (state persists until the next event).

    Returns a tuple: (intervals, seconds_where_condition_true, total_observed_seconds).
    """
    if df is None or df.empty:
        return [], 0.0, 0.0

    gdf = df.copy()
    if time_col not in gdf.columns:
        if verbose:
            print(f"Warning: time_col {time_col} not in df columns")
        return [], 0.0, 0.0

    gdf[time_col] = pd.to_numeric(gdf[time_col], errors='coerce')
    valid = gdf[time_col].notna()
    if not valid.any():
        return [], 0.0, 0.0

    gdf = gdf.loc[valid].sort_values(by=time_col).reset_index(drop=True)

    if verbose:
        print(f"[debug] intervals_for_condition: original rows={len(df)}, valid_time_rows={valid.sum()}")

    # build boolean mask
    m = pd.Series(True, index=gdf.index)
    for col, val in condition.items():
        if col not in gdf.columns:
            m &= pd.Series(False, index=gdf.index)
            break
        series = gdf[col]
        if isinstance(val, (list, tuple, set)):
            try:
                m &= series.isin(val)
            except Exception:
                m &= pd.Series(False, index=gdf.index)
        else:
            try:
                m &= (series == val)
            except Exception:
                m &= pd.Series(False, index=gdf.index)

    m = m.fillna(False).astype(bool).reset_index(drop=True)
    times = gdf[time_col].astype(float).reset_index(drop=True)

    if verbose:
        print(f"[debug] mask true count={int(m.sum())} / {len(m)} rows")

    arr = np.asarray(m.tolist(), dtype=bool)
    intervals: List[Tuple[float, float]] = []
    cond_seconds = 0.0

    if arr.size == 0:
        if verbose:
            print("[debug] no rows after filtering/time coercion; returning empty intervals")
        return intervals, cond_seconds, 0.0

    darr = np.diff(arr.astype(int))
    starts = np.where(darr == 1)[0] + 1
    ends = np.where(darr == -1)[0]
    if arr[0]:
        starts = np.concatenate(([0], starts))
    if arr[-1]:
        ends = np.concatenate((ends, [arr.size - 1]))

    if verbose:
        print(f"[debug] starts indices: {starts.tolist()}\n[debug] ends indices: {ends.tolist()}")

    if starts.size == ends.size:
        n_times = len(times)
        for s, e in zip(starts.tolist(), ends.tolist()):
            try:
                st = float(times.iloc[s])
            except Exception:
                continue
            en_idx = e + 1 if (e + 1) < n_times else e
            try:
                en = float(times.iloc[en_idx])
            except Exception:
                try:
                    en = float(times.iloc[e])
                except Exception:
                    continue
            if en >= st:
                intervals.append((st, en))
                cond_seconds += (en - st)
                if verbose:
                    print(f"[debug] captured interval: start_idx={s}, end_idx={e}, end_idx_used={en_idx}, start={st:.1f}, end={en:.1f}, dur={(en-st):.1f}s")

    total_observed = float(times.iloc[-1] - times.iloc[0]) if len(times) >= 2 else 0.0
    return intervals, cond_seconds, total_observed


def intervals_for_conditions(
    df: pd.DataFrame,
    conditions: Any,
    time_col: str = 'total_time_elapsed_seconds',
    verbose: bool = False,
) -> Dict[str, Tuple[List[Tuple[float, float]], float, float]]:
    """Wrapper to compute intervals for multiple named conditions.

    Accepts the flexible `conditions` formats used previously.
    """
    if df is None or df.empty:
        return {}

    normalized: Dict[str, Dict[str, Any]] = {}
    if isinstance(conditions, dict) and all(isinstance(v, dict) for v in conditions.values()):
        normalized = conditions
    elif isinstance(conditions, dict):
        normalized = {'cond0': conditions}
    elif isinstance(conditions, (list, tuple)):
        for i, item in enumerate(conditions):
            if isinstance(item, tuple) and len(item) == 2:
                label, cond = item
                normalized[str(label)] = dict(cond)
            elif isinstance(item, dict):
                normalized[f'cond{i}'] = item
    else:
        normalized = {'cond0': dict(conditions)}

    results: Dict[str, Tuple[List[Tuple[float, float]], float, float]] = {}
    for label, cond in normalized.items():
        try:
            intervals, cond_sec, total_sec = intervals_for_condition(df, cond, time_col=time_col, verbose=verbose)
            results[label] = (intervals, cond_sec, total_sec)
        except Exception as e:
            if verbose:
                print(f"[debug] intervals_for_conditions: failed for label={label}: {e}")
            results[label] = ([], 0.0, 0.0)
    return results


def add_game_state_relative_column(df: pd.DataFrame, team: Any) -> pd.DataFrame:
    """Add 'game_state_relative_to_team' which is relative to `team`.

    For each row the value is:
      - the existing `game_state` when the acting team is `team`.
      - the flipped state (e.g. '5v4' -> '4v5') when the acting team is the opponent.

    Penalty rows are treated specially: initially set to None, then when
    possible they inherit a nearby faceoff's state (exact match on time
    or the nearest faceoff within 1 second). This heuristic was used in
    the original code to account for how penalties are logged relative to
    faceoffs.
    """
    if df is None or df.empty:
        return df
    out = df.copy()

    tstr = str(team).strip().upper() if team is not None else ''
    candidate_ids = set()
    if tstr.isdigit():
        candidate_ids.add(int(tstr))
    else:
        if 'home_abb' in out.columns and 'home_id' in out.columns:
            try:
                mask = out['home_abb'].astype(str).str.upper() == tstr
                ids = out.loc[mask, 'home_id'].dropna().unique().tolist()
                candidate_ids.update([int(i) for i in ids if str(i).isdigit() or isinstance(i, int)])
            except Exception:
                pass
        if 'away_abb' in out.columns and 'away_id' in out.columns:
            try:
                mask = out['away_abb'].astype(str).str.upper() == tstr
                ids = out.loc[mask, 'away_id'].dropna().unique().tolist()
                candidate_ids.update([int(i) for i in ids if str(i).isdigit() or isinstance(i, int)])
            except Exception:
                pass

    def _flip_state(s: Any) -> Any:
        try:
            ss = str(s)
            if 'v' in ss:
                a, b = ss.split('v', 1)
                a = a.strip(); b = b.strip()
                if a.isdigit() and b.isdigit():
                    return f"{b}v{a}"
            return s
        except Exception:
            return s

    team_mask = None
    if 'team_id' in out.columns:
        try:
            if candidate_ids:
                team_mask = out['team_id'].astype(str).isin([str(i) for i in candidate_ids])
            else:
                if tstr.isdigit():
                    team_mask = out['team_id'].astype(str) == tstr
                else:
                    team_mask = None
        except Exception:
            team_mask = None

    if team_mask is None:
        try:
            idx = out.index
            left = pd.Series(False, index=idx)
            right = pd.Series(False, index=idx)
            if 'home_abb' in out.columns and 'home_id' in out.columns:
                try:
                    left = (out['home_abb'].astype(str).str.upper() == tstr) & out['team_id'].notna() & (out['team_id'] == out['home_id'])
                except Exception:
                    left = left
            if 'away_abb' in out.columns and 'away_id' in out.columns:
                try:
                    right = (out['away_abb'].astype(str).str.upper() == tstr) & out['team_id'].notna() & (out['team_id'] == out['away_id'])
                except Exception:
                    right = right
            team_mask = left | right
        except Exception:
            team_mask = pd.Series(False, index=out.index)

    try:
        team_mask = team_mask.fillna(False).astype(bool)
    except Exception:
        team_mask = pd.Series(False, index=out.index)

    penalty_mask = pd.Series(False, index=out.index)
    ev_text = None
    try:
        kw = ('event', 'type', 'desc', 'detail', 'penal', 'penalty', 'description')
        ev_cols = [c for c in out.columns if any(k in c.lower() for k in kw)]
        if ev_cols:
            ev_text = out[ev_cols].apply(lambda col: col.astype(str).fillna(''))
            ev_text = ev_text.agg(' '.join, axis=1).str.lower()
            penalty_mask = ev_text.str.contains('penal') | ev_text.str.contains('penalty')
    except Exception:
        penalty_mask = pd.Series(False, index=out.index)

    gs_series = out.get('game_state', pd.Series([None] * len(out)))
    provisional_rel = [None] * len(out)
    for idx, (is_team, gs, is_pen) in enumerate(zip(team_mask.tolist(), gs_series.tolist(), penalty_mask.tolist())):
        if is_pen:
            provisional_rel[idx] = None
        else:
            provisional_rel[idx] = gs if is_team else _flip_state(gs)

    time_col = 'total_time_elapsed_seconds'
    time_series = pd.to_numeric(out.get(time_col, pd.Series([np.nan] * len(out))), errors='coerce')
    if ev_text is None:
        candidate = [c for c in ('event', 'type', 'typeDescKey', 'description') if c in out.columns]
        if candidate:
            ev_text = out[candidate].astype(str).fillna('').agg(' '.join, axis=1).str.lower()
        else:
            ev_text = pd.Series([''] * len(out))

    face_mask = ev_text.str.contains(r'\bface[ -]?off\b', regex=True)

    final_rel = [None] * len(out)
    times_arr = time_series.values
    face_mask_arr = face_mask.fillna(False).values
    for idx in range(len(out)):
        if penalty_mask.iloc[idx]:
            assigned = None
            tval = times_arr[idx]
            if not (tval is None or (isinstance(tval, float) and np.isnan(tval))):
                candidate_idxs = np.where(face_mask_arr & np.isfinite(times_arr))[0]
                if candidate_idxs.size > 0:
                    rel_assigned = None
                    is_exact = np.isclose(times_arr[candidate_idxs], tval, atol=1e-6, rtol=1e-8)
                    exact_idxs = candidate_idxs[is_exact]
                    if exact_idxs.size > 0:
                        rel_assigned = provisional_rel[exact_idxs[0]]
                    else:
                        diffs = np.abs(times_arr[candidate_idxs] - tval)
                        nearest_pos = np.argmin(diffs)
                        if diffs[nearest_pos] <= 1.0:
                            nearest_idx = candidate_idxs[nearest_pos]
                            rel_assigned = provisional_rel[nearest_idx]
                    assigned = rel_assigned
            final_rel[idx] = assigned
        else:
            final_rel[idx] = provisional_rel[idx]

    out['game_state_relative_to_team'] = final_rel
    return out


def demo_for_export(df, condition=None, verbose: bool = False):
    """Aggregate interval information across games for a condition.

    Behavior preserved from previous implementation:
      - If `condition` is a dict and contains 'team', analyze games for that team only.
      - Otherwise analyze every unique game id found in `df`.
      - For each game compute per-side intervals (team/opponent) and an
        intersection across the requested analysis conditions.
      - Aggregate per-game results into per-bucket summaries where the
        buckets are 'team' (selected/local team) and 'other' (opponent(s)).

    The function returns a dict with keys 'per_game' and 'aggregate'. The
    structure is intentionally simple and intended for human inspection
    and downstream plotting.

    Example return structure:
      {
        'per_game': {
          'game_id_1': {
            'selected_team': 'team_abbrev',
            'opponent_team': 'opp_abbrev',
            'sides': {
              'team': {  # local team
                'merged_intervals': { 'game_state': [...], ... },
                'pooled_seconds': { 'game_state': ..., ... },
                'total_observed': <total_seconds>,
                'intersection_intervals': [...],
                'pooled_intersection_seconds': <seconds>,
              },
              'opponent': {  # opponent team
                'merged_intervals': { 'game_state': [...], ... },
                'pooled_seconds': { 'game_state': ..., ... },
                'total_observed': <total_seconds>,
                'intersection_intervals': [...],
                'pooled_intersection_seconds': <seconds>,
              },
            },
            'game_total_observed_seconds': <total_seconds>,
          },
          ...
        },
        'aggregate': {
          'pooled_seconds_per_condition': { 'team': {...}, 'other': {...} },
          'intervals_per_condition': { 'team': {...}, 'other': {...} },
          'intersection_pooled_seconds': { 'team': <seconds>, 'other': <seconds> },
          'intersection_intervals': { 'team': [...], 'other': [...] },
        },
      }

    The 'per_game' section contains detailed info for each game processed,
    while 'aggregate' provides summarized totals and merged interval lists
    for all games analyzed.
    """
    import parse
    import nhl_api
    # Determine the team to analyze: prefer condition['team'] when provided; if
    # no team is provided, we'll analyze all games found in `df` and for each
    # game default the 'team' perspective to the home's team for that game.
    team_param = None
    if isinstance(condition, dict) and 'team' in condition:
        team_param = condition['team']

    # Derive analysis conditions from the passed filter `condition`.
    # If the condition dict contains keys other than 'team', those keys are
    # used as analysis conditions. Otherwise fall back to defaults.
    if isinstance(condition, dict):
        # Pull out non-team keys to use as analysis conditions
        raw_conditions = {k: v for k, v in condition.items() if k != 'team'}
        if not raw_conditions:
            # default analysis conditions
            analysis_conditions = {'game_state': ['5v5'], 'is_net_empty': [0, 1]}
        else:
            # Normalize each value to a list of states. This avoids iterating
            # a string as characters later when we do `for state in cond_def`.
            analysis_conditions = {}
            for k, v in raw_conditions.items():
                if isinstance(v, (list, tuple, set)):
                    analysis_conditions[k] = list(v)
                else:
                    # For scalar values (including strings/ints), wrap in list
                    analysis_conditions[k] = [v]
    else:
        analysis_conditions = {'game_state': ['5v5'], 'is_net_empty': [0, 1]}

    # Determine gids to analyze:
    # - If a team was specified in `condition`, find games for that team.
    # - Otherwise, analyze every unique game_id present in `df`.
    if isinstance(condition, dict) and 'team' in condition:
        # When a team is specified we want to analyze games involving that team.
        # `df` may be either:
        #  - a season-level DataFrame (one row per game), where `select_team_game`
        #    is appropriate, OR
        #  - an events-level DataFrame (many rows per game) where we should
        #    derive the unique game ids by filtering rows that involve the team.
        gids = None
        try:
            # detect events-level DataFrame by presence of typical event cols
            is_events_df = isinstance(df, pd.DataFrame) and 'game_id' in df.columns and any(c in df.columns for c in ('event', 'x', 'y', 'period'))
        except Exception:
            is_events_df = False

        if is_events_df:
            try:
                # Try to find game ids where the team appears as home_abb/away_abb or home_id/away_id
                t = str(team_param).strip().upper()
                mask = pd.Series(False, index=df.index)
                if 'home_abb' in df.columns:
                    try:
                        mask = mask | (df['home_abb'].astype(str).str.upper() == t)
                    except Exception:
                        pass
                if 'away_abb' in df.columns:
                    try:
                        mask = mask | (df['away_abb'].astype(str).str.upper() == t)
                    except Exception:
                        pass
                # numeric id match
                try:
                    tid = int(t)
                except Exception:
                    tid = None
                if tid is not None:
                    if 'home_id' in df.columns:
                        try:
                            mask = mask | (df['home_id'].astype(str) == str(tid))
                        except Exception:
                            pass
                    if 'away_id' in df.columns:
                        try:
                            mask = mask | (df['away_id'].astype(str) == str(tid))
                        except Exception:
                            pass

                if mask.any():
                    try:
                        gids = df.loc[mask, 'game_id'].dropna().unique().tolist()
                    except Exception:
                        gids = []
                else:
                    gids = []
            except Exception:
                gids = None

        # fallback: if not events-level or prior step failed, use select_team_game
        if gids is None:
            gids = select_team_game(df, team_param)

        if not gids:
            # Return a structured empty result to maintain consistent output format
            return {
                'per_game': {},
                'aggregate': {
                    'pooled_seconds_per_condition': {'team': {}, 'other': {}},
                    'intervals_per_condition': {'team': {}, 'other': {}},
                    'intersection_pooled_seconds': {'team': 0.0, 'other': 0.0},
                    'intersection_intervals': {'team': [], 'other': []},
                }
            }
        if not isinstance(gids, (list, tuple, pd.Series)):
            gids = [gids]
    else:
        # No team specified: use all unique game_ids in df (if present). This
        # works for both season-level and events-level DataFrames.
        if isinstance(df, pd.DataFrame) and 'game_id' in df.columns and not df.empty:
            try:
                gids = df['game_id'].dropna().unique().tolist()
            except Exception:
                gids = []
        else:
            gids = []

    # prepare aggregates and helpers
    results_per_game = {}
    aggregate_per_condition: Dict[str, Dict[str, float]] = {'team': {}, 'other': {}}
    aggregate_intervals_per_condition: Dict[str, Dict[str, List[Tuple[float, float]]]] = {'team': {}, 'other': {}}
    aggregate_intersection_total: Dict[str, float] = {'team': 0.0, 'other': 0.0}
    aggregate_intersection_intervals: Dict[str, List[Tuple[float, float]]] = {'team': [], 'other': []}

    def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        merged: List[Tuple[float, float]] = []
        cur_s, cur_e = intervals[0]
        EPS = 1e-9
        for s, e in intervals[1:]:
            if s <= cur_e + EPS:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        return merged

    def _intersect_two(a: List[Tuple[float, float]], b: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        res: List[Tuple[float, float]] = []
        if not a or not b:
            return res
        i = j = 0
        EPS = 1e-9
        while i < len(a) and j < len(b):
            s1, e1 = a[i]
            s2, e2 = b[j]
            start = max(s1, s2)
            end = min(e1, e2)
            if end > start + EPS:
                res.append((start, end))
            if e1 < e2 - EPS:
                i += 1
            elif e2 < e1 - EPS:
                j += 1
            else:
                i += 1
                j += 1
        return res

    # iterate games
    for gid in gids:
        # Ensure we operate on a fresh game-level DataFrame (don't rely on incoming df if it's filtered)
        gdf = None
        feed = None
        # try numeric id first
        try:
            gid_int = int(gid)
        except Exception:
            gid_int = gid
        try:
            feed = nhl_api.get_game_feed(gid_int)
        except Exception:
            try:
                feed = nhl_api.get_game_feed(gid)
            except Exception:
                feed = None

        if feed:
            try:
                ev_df = parse._game(feed)
                if ev_df is None or ev_df.empty:
                    gdf = None
                else:
                    gdf = parse._elaborate(ev_df)
            except Exception:
                gdf = None
        else:
            # fallback to rows from provided df
            try:
                if isinstance(df, pd.DataFrame) and 'game_id' in df.columns:
                    gdf = df[df['game_id'] == gid]
                else:
                    gdf = None
            except Exception:
                gdf = None

        if gdf is None or gdf.empty:
            continue

        # infer basic game metadata
        try:
            home_id = gdf['home_id'].dropna().unique().tolist()[0] if 'home_id' in gdf.columns else None
        except Exception:
            home_id = None
        try:
            away_id = gdf['away_id'].dropna().unique().tolist()[0] if 'away_id' in gdf.columns else None
        except Exception:
            away_id = None
        try:
            home_abb = gdf['home_abb'].dropna().unique().tolist()[0] if 'home_abb' in gdf.columns else None
        except Exception:
            home_abb = None
        try:
            away_abb = gdf['away_abb'].dropna().unique().tolist()[0] if 'away_abb' in gdf.columns else None
        except Exception:
            away_abb = None

        # choose local team for this game
        if team_param is not None:
            local_team = team_param
        else:
            local_team = home_abb if home_abb is not None else (home_id if home_id is not None else (away_abb if away_abb is not None else away_id))

        # normalize for numeric check
        tstr = str(local_team).strip() if local_team is not None else ''
        try:
            tid = int(tstr)
        except Exception:
            tid = None

        # compute opponent keys
        # determine if local_team corresponds to home or away
        selected_is_home = False
        selected_is_away = False
        if tid is not None:
            if home_id is not None and str(home_id) == str(tid):
                selected_is_home = True
            if away_id is not None and str(away_id) == str(tid):
                selected_is_away = True
        else:
            tupper = str(local_team).upper() if local_team is not None else ''
            if home_abb is not None and str(home_abb).upper() == tupper:
                selected_is_home = True
            if away_abb is not None and str(away_abb).upper() == tupper:
                selected_is_away = True

        if selected_is_home:
            opp_id = away_id
            opp_abb = away_abb
        elif selected_is_away:
            opp_id = home_id
            opp_abb = home_abb
        else:
            # fallback
            opp_id = away_id if (home_abb is not None and str(home_abb).upper() == str(local_team).upper()) else home_id
            opp_abb = away_abb if (home_abb is not None and str(home_abb).upper() == str(local_team).upper()) else home_abb

        team_key = str(local_team)
        opp_key = str(opp_abb or opp_id or 'opponent')

        per_game_info = {}
        for side_label, side_team in (('team', local_team), ('opponent', opp_key)):
            side_team_val = side_team
            if side_label == 'opponent':
                side_team_val = opp_id if opp_id is not None else (opp_abb if opp_abb is not None else opp_key)

            # annotate relative game_state
            df_side = add_game_state_relative_column(gdf.copy(), side_team_val)

            merged_per_condition_local: Dict[str, List[Tuple[float, float]]] = {}
            pooled_seconds_per_condition_local: Dict[str, float] = {}

            times = pd.to_numeric(df_side.get('total_time_elapsed_seconds', pd.Series(dtype=float)), errors='coerce').dropna()
            total_observed = float(times.max() - times.min()) if len(times) >= 2 else 0.0

            for cond_label, cond_def in analysis_conditions.items():
                all_intervals: List[Tuple[float, float]] = []
                for state in cond_def:
                    if cond_label == 'game_state':
                        cond_dict = {'game_state_relative_to_team': state}
                    else:
                        cond_dict = {cond_label: state}
                    try:
                        intervals, _, _ = intervals_for_condition(df_side, cond_dict, time_col='total_time_elapsed_seconds', verbose=verbose)
                        for it in intervals:
                            try:
                                all_intervals.append((float(it[0]), float(it[1])))
                            except Exception:
                                continue
                    except Exception as e:
                        if verbose:
                            print(f"[debug] demo_for_export: intervals_for_condition failed for {side_label} - {cond_label} state={state}: {e}")
                        continue

                merged_intervals = _merge_intervals(all_intervals)
                pooled = sum((e - s) for s, e in merged_intervals) if merged_intervals else 0.0
                merged_per_condition_local[str(cond_label)] = merged_intervals
                pooled_seconds_per_condition_local[str(cond_label)] = pooled

            per_game_info[side_label] = {
                'merged_intervals': merged_per_condition_local,
                'pooled_seconds': pooled_seconds_per_condition_local,
                'total_observed': total_observed,
            }

        # compute intersections per side
        for side_label, side_info in per_game_info.items():
            cond_interval_lists = list(side_info['merged_intervals'].values())
            if cond_interval_lists:
                inter = cond_interval_lists[0]
                for lst in cond_interval_lists[1:]:
                    inter = _intersect_two(inter, lst)
            else:
                inter = []
            pooled_intersection = sum((e - s) for s, e in inter) if inter else 0.0
            side_info['intersection_intervals'] = inter
            side_info['pooled_intersection_seconds'] = pooled_intersection

            agg_bucket = 'team' if side_label == 'team' else 'other'
            for cond_label, pooled_seconds in side_info['pooled_seconds'].items():
                aggregate_per_condition[agg_bucket].setdefault(cond_label, 0.0)
                aggregate_per_condition[agg_bucket][cond_label] += pooled_seconds
                existing = aggregate_intervals_per_condition[agg_bucket].setdefault(cond_label, [])
                aggregate_intervals_per_condition[agg_bucket][cond_label] = _merge_intervals(existing + side_info['merged_intervals'].get(cond_label, []))

            aggregate_intersection_total[agg_bucket] += pooled_intersection
            aggregate_intersection_intervals[agg_bucket] = _merge_intervals(aggregate_intersection_intervals[agg_bucket] + inter)

        results_per_game[gid] = {
            'selected_team': team_key,
            'opponent_team': opp_key,
            'sides': per_game_info,
            'game_total_observed_seconds': total_observed,
        }

    # recompute aggregate totals from per-game to avoid drift
    recomputed_intersection_total: Dict[str, float] = {'team': 0.0, 'other': 0.0}
    for gid, info in results_per_game.items():
        sides = info.get('sides', {})
        tsec = float(sides.get('team', {}).get('pooled_intersection_seconds', 0.0) or 0.0)
        osec = float(sides.get('opponent', {}).get('pooled_intersection_seconds', 0.0) or 0.0)
        recomputed_intersection_total['team'] += tsec
        recomputed_intersection_total['other'] += osec

    aggregate_intersection_total = recomputed_intersection_total

    final_results = {
        'per_game': results_per_game,
        'aggregate': {
            'pooled_seconds_per_condition': aggregate_per_condition,
            'intervals_per_condition': aggregate_intervals_per_condition,
            'intersection_pooled_seconds': aggregate_intersection_total,
            'intersection_intervals': aggregate_intersection_intervals,
        },
    }

    return final_results
