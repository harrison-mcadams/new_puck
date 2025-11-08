"""
timing_brainstorming.py

Small, readable utilities to compute time-intervals in a game's event
DataFrame where simple conditions hold (for debugging game_state and
is_net_empty logic).

Usage (demo): run the module directly; it will attempt to load
`data/20252026/20252026_df.csv` (or a few reasonable fallbacks), select
team 'PHI' and compute intervals for several game_state values.

The core function is `intervals_for_condition(df, condition, time_col)`.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np


def load_season_df(season: str = '20252026', data_dir: str = 'data') -> pd.DataFrame:
    """Try a few likely locations for a season-level CSV and return a DataFrame.

    Looks for (in order):
      - {data_dir}/{season}/{season}_df.csv
      - {data_dir}/{season}/{season}.csv
      - {data_dir}/{season}_df.csv
      - {data_dir}/{season}.csv

    If none exist, returns an empty DataFrame.
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
    print("No season CSV found among candidates; returning empty DataFrame", flush=True)
    return pd.DataFrame()


def select_team_game(df: pd.DataFrame, team: str) -> Optional[Any]:
    """Select a reasonable single game id for `team` from `df`.

    Strategy: match by home_abb/away_abb (case-insensitive) or home_id/away_id.
    Return the most-recent game id by numeric value if possible, otherwise
    the last seen game_id in the DataFrame filtered for that team.
    If no matches, return None.
    """
    if df is None or df.empty:
        return None
    t = str(team).strip().upper()
    # Build candidate boolean mask defensively
    home_abb = df.get('home_abb')
    away_abb = df.get('away_abb')
    home_id = df.get('home_id')
    away_id = df.get('away_id')

    mask = pd.Series(False, index=df.index)
    if home_abb is not None:
        try:
            mask = mask | (home_abb.astype(str).str.upper() == t)
        except Exception:
            pass
    if away_abb is not None:
        try:
            mask = mask | (away_abb.astype(str).str.upper() == t)
        except Exception:
            pass
    # also allow numeric id string match
    try:
        tid = int(t)
    except Exception:
        tid = None
    if tid is not None:
        if home_id is not None:
            try:
                mask = mask | (home_id.astype(str) == str(tid))
            except Exception:
                pass
        if away_id is not None:
            try:
                mask = mask | (away_id.astype(str) == str(tid))
            except Exception:
                pass

    if not mask.any():
        return None

    # unique game ids where team appears
    if 'game_id' in df.columns:
        try:
            games = df.loc[mask, 'game_id'].dropna().unique().tolist()
        except Exception:
            games = []
    else:
        games = []

    if not games:
        # fallback: return the last game_id seen in filtered rows (if any)
        try:
            last = df.loc[mask].tail(1)
            if 'game_id' in last.columns:
                return last['game_id'].iloc[0]
        except Exception:
            return None
    # choose the numerically largest game id when possible (proxy for most recent)
    try:
        numeric_games = sorted([int(g) for g in games])
        return numeric_games
    except Exception:
        # fallback to the last in appearance order
        return games


def intervals_for_condition(
    df: pd.DataFrame,
    condition: Dict[str, Any],
    time_col: str = 'total_time_elapsed_seconds',
    verbose: bool = False,
) -> Tuple[List[Tuple[float, float]], float, float]:
    """Compute contiguous time intervals where `condition` holds in `df`.

    - `condition` is a dict like {'game_state': '5v5'} or {'is_net_empty': 1}.
    - Returns (intervals, condition_total_seconds, total_observed_seconds).

    The function is intentionally simple and robust:
      - Coerces time_col to numeric and drops rows without time.
      - Sorts by time and finds contiguous True blocks in the boolean mask.
    """
    if df is None or df.empty:
        return [], 0.0, 0.0

    # Defensive copy and ensure time column exists
    gdf = df.copy()
    if time_col not in gdf.columns:
        print(f"Warning: time_col {time_col} not in df columns")
        return [], 0.0, 0.0

    # Coerce time and drop rows without valid time
    gdf[time_col] = pd.to_numeric(gdf[time_col], errors='coerce')
    valid = gdf[time_col].notna()
    if not valid.any():
        return [], 0.0, 0.0

    # Filter, sort, and reset index so we can build a mask aligned to gdf
    gdf = gdf.loc[valid].sort_values(by=time_col).reset_index(drop=True)

    if verbose:
        print(f"[debug] intervals_for_condition: original rows={len(df)}, valid_time_rows={valid.sum()}")
        try:
            print(f"[debug] time range after filter: {gdf[time_col].min()} -> {gdf[time_col].max()}")
        except Exception:
            pass

    # Build boolean mask for the condition on the filtered/reindexed DataFrame
    m = pd.Series(True, index=gdf.index)
    for col, val in condition.items():
        if col not in gdf.columns:
            # if column missing, mask is all False
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

    # Ensure boolean Series without NaNs and aligned with integer index
    m = m.fillna(False).astype(bool).reset_index(drop=True)
    times = gdf[time_col].astype(float).reset_index(drop=True)

    if verbose:
        try:
            print(f"[debug] mask true count={int(m.sum())} / {len(m)} rows")
        except Exception:
            pass

    arr = np.asarray(m.tolist(), dtype=bool)
    intervals: List[Tuple[float, float]] = []
    cond_seconds = 0.0

    if arr.size == 0:
        if verbose:
            print("[debug] no rows after filtering/time coercion; returning empty intervals")
        return intervals, cond_seconds, 0.0

    # find starts/ends
    darr = np.diff(arr.astype(int))
    starts = np.where(darr == 1)[0] + 1
    ends = np.where(darr == -1)[0]
    if arr[0]:
        starts = np.concatenate(([0], starts))
    if arr[-1]:
        ends = np.concatenate((ends, [arr.size - 1]))

    # pair starts/ends and report details if requested
    if verbose:
        print(f"[debug] starts indices: {starts.tolist()}\n[debug] ends indices: {ends.tolist()}")
    if starts.size == ends.size:
        n_times = len(times)
        for s, e in zip(starts.tolist(), ends.tolist()):
            try:
                st = float(times.iloc[s])
            except Exception:
                continue
            # end should be the time of the next event after index e (state persists until next event)
            en_idx = e + 1 if (e + 1) < n_times else e
            try:
                en = float(times.iloc[en_idx])
            except Exception:
                # fallback to last known time at e
                try:
                    en = float(times.iloc[e])
                except Exception:
                    continue
            # only accept intervals with non-negative duration
            if en >= st:
                intervals.append((st, en))
                dur = en - st
                cond_seconds += dur
                if verbose:
                    print(f"[debug] captured interval: start_idx={s}, end_idx={e}, end_idx_used={en_idx}, start={st:.1f}, end={en:.1f}, dur={dur:.1f}s")

    total_observed = float(times.iloc[-1] - times.iloc[0]) if len(times) >= 2 else 0.0
    return intervals, cond_seconds, total_observed


def intervals_for_conditions(
    df: pd.DataFrame,
    conditions: Any,
    time_col: str = 'total_time_elapsed_seconds',
    verbose: bool = False,
) -> Dict[str, Tuple[List[Tuple[float, float]], float, float]]:
    """Compute intervals and totals for multiple named conditions.

    `conditions` can be provided in several forms:
      - a dict mapping label -> condition_dict (recommended)
      - a list/tuple of (label, condition_dict)
      - a single condition_dict (will be labeled 'cond0')

    Returns a dict: label -> (intervals, cond_seconds, total_observed_seconds).
    This is a thin wrapper around `intervals_for_condition` to make batch
    processing easier and keep the API simple for callers that want many
    conditions at once.
    """
    if df is None or df.empty:
        return {}

    # Normalize conditions into an ordered mapping of label -> cond_dict
    normalized: Dict[str, Dict[str, Any]] = {}
    if isinstance(conditions, dict) and all(isinstance(v, dict) for v in conditions.values()):
        normalized = conditions
    elif isinstance(conditions, dict):
        # single unnamed condition dict passed
        normalized = {'cond0': conditions}
    elif isinstance(conditions, (list, tuple)):
        for i, item in enumerate(conditions):
            if isinstance(item, tuple) and len(item) == 2:
                label, cond = item
                normalized[str(label)] = dict(cond)
            elif isinstance(item, dict):
                normalized[f'cond{i}'] = item
            else:
                # skip unknown entries
                continue
    else:
        # last resort: wrap into single condition
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
    """Return a copy of df with a new column 'game_state_relative_to_team'.

    The logic:
      - Determine numeric team_id(s) corresponding to the provided `team`.
        `team` may be a numeric id (int or digit string) or a team abbrev (e.g. 'PHI').
      - For each row, if the event's 'team_id' matches the team_id(s),
        then game_state_relative_to_team == game_state.
        Otherwise, flip the game_state (swap sides: '5v4' -> '4v5').

    This function is defensive to missing columns and returns the input df
    (shallow copy) with the new column added.
    """
    if df is None or df.empty:
        return df
    out = df.copy()

    # Normalize team input
    tstr = str(team).strip().upper() if team is not None else ''
    candidate_ids = set()
    # If numeric, use directly
    if tstr.isdigit():
        candidate_ids.add(int(tstr))
    else:
        # try to infer id(s) by matching home_abb/away_abb
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

    # Helper to flip 'XvY' -> 'YvX'
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

    # Construct boolean mask for events where team is the actor
    team_mask = None
    if 'team_id' in out.columns:
        try:
            if candidate_ids:
                team_mask = out['team_id'].astype(str).isin([str(i) for i in candidate_ids])
            else:
                # As fallback, consider matching by comparing 'team_id' to any numeric form of tstr
                if tstr.isdigit():
                    team_mask = out['team_id'].astype(str) == tstr
                else:
                    # try matching by team abbreviation against home/away mapping
                    # create a mapping of abb -> id from the first few rows
                    team_mask = None
        except Exception:
            team_mask = None
    # If team_mask still None, attempt to build it via row-level check using home_abb/away_abb and home_id/away_id
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

    # ensure boolean
    try:
        team_mask = team_mask.fillna(False).astype(bool)
    except Exception:
        team_mask = pd.Series(False, index=out.index)

    # Detect penalty rows to censor them when building relative state. Look for typical
    # penalty indicators in common columns (e.g., 'event', 'event_type', 'typeDescKey', 'type').
    penalty_mask = pd.Series(False, index=out.index)
    ev_text = None
    try:
        # choose candidate columns heuristically: any column name containing these tokens
        kw = ('event', 'type', 'desc', 'detail', 'penal', 'penalty', 'description')
        ev_cols = [c for c in out.columns if any(k in c.lower() for k in kw)]
        if ev_cols:
            # stringify and concatenate the candidate columns per-row
            # guard against non-scalar entries by converting to string
            ev_text = out[ev_cols].apply(lambda col: col.astype(str).fillna(''))
            ev_text = ev_text.agg(' '.join, axis=1).str.lower()
            # detect common penalty indicators
            penalty_mask = ev_text.str.contains('penal') | ev_text.str.contains('penalty')
    except Exception:
        penalty_mask = pd.Series(False, index=out.index)

    # First pass: compute provisional relative state for non-penalty rows
    gs_series = out.get('game_state', pd.Series([None] * len(out)))
    provisional_rel = [None] * len(out)
    for idx, (is_team, gs, is_pen) in enumerate(zip(team_mask.tolist(), gs_series.tolist(), penalty_mask.tolist())):
        if is_pen:
            provisional_rel[idx] = None
        else:
            provisional_rel[idx] = gs if is_team else _flip_state(gs)

    # Build vectors for time and make sure ev_text exists
    time_col = 'total_time_elapsed_seconds'
    time_series = pd.to_numeric(out.get(time_col, pd.Series([np.nan] * len(out))), errors='coerce')
    if ev_text is None:
        candidate = [c for c in ('event', 'type', 'typeDescKey', 'description') if c in out.columns]
        if candidate:
            ev_text = out[candidate].astype(str).fillna('').agg(' '.join, axis=1).str.lower()
        else:
            ev_text = pd.Series([''] * len(out))

    face_mask = ev_text.str.contains(r'\bface[ -]?off\b', regex=True)

    # Second pass: for penalty rows, find faceoff rows at same time (or nearest within 1s)
    final_rel = [None] * len(out)
    times_arr = time_series.values
    face_mask_arr = face_mask.fillna(False).values
    for idx in range(len(out)):
        if penalty_mask.iloc[idx]:
            assigned = None
            tval = times_arr[idx]
            # ensure tval is a finite number
            if not (tval is None or (isinstance(tval, float) and np.isnan(tval))):
                # candidate indices where faceoff exists and time is finite
                candidate_idxs = np.where(face_mask_arr & np.isfinite(times_arr))[0]
                if candidate_idxs.size > 0:
                    # exact-ish match using isclose
                    rel_assigned = None
                    is_exact = np.isclose(times_arr[candidate_idxs], tval, atol=1e-6, rtol=1e-8)
                    exact_idxs = candidate_idxs[is_exact]
                    if exact_idxs.size > 0:
                        rel_assigned = provisional_rel[exact_idxs[0]]
                    else:
                        # pick nearest within 1.0 second
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


def demo_for_team_game(season: str = '20252026', team: str = 'PHI', data_dir: str = 'data', verbose: bool = False):
    """Demo runner: load season df, pick a game for `team`, compute intervals.

    Prints results for a handful of canonical game_state values and for is_net_empty.
    """
    df = load_season_df(season=season, data_dir=data_dir)
    if df.empty:
        print('No data loaded; aborting demo', flush=True)
        return

    gid = select_team_game(df, team)
    gid = gid[0]
    if gid is None:
        print(f'No games found for team {team}', flush=True)
        return

    print(f"Selected game id {gid} for team {team}", flush=True)
    gdf = df[df['game_id'] == gid]
    if gdf.empty:
        print('Selected game has no rows; aborting', flush=True)
        return

    # Add relative game state column for the demo
    gdf = add_game_state_relative_column(gdf, team)

    # Build a set of named conditions and compute them in batch using our helper
    conditions = {
        '5v5': {'game_state_relative_to_team': '5v5'},
        'is_net_empty_1': {'is_net_empty': 1},
        'is_net_empty_0': {'is_net_empty': 0},
    }
    results = intervals_for_conditions(gdf, conditions, time_col='total_time_elapsed_seconds', verbose=verbose)

    # Print game-state intervals (relative to team)
    print('\nGame-state intervals (relative to team):', flush=True)
    intervals, cond_sec, total_sec = results.get('5v5', ([], 0.0, 0.0))
    pct = (cond_sec / total_sec * 100.0) if total_sec > 0 else 0.0
    print(f"  5v5: {intervals}, {cond_sec:.1f}s / {total_sec:.1f}s ({pct:.1f}%)", flush=True)

    # is_net_empty intervals
    print('\nis_net_empty intervals (relative to team):', flush=True)
    for key in ('is_net_empty_1', 'is_net_empty_0'):
        intervals, cond_sec, total_sec = results.get(key, ([], 0.0, 0.0))
        empty = 1 if key.endswith('_1') else 0
        pct = (cond_sec / total_sec * 100.0) if total_sec > 0 else 0.0
        print(f"  is_net_empty={empty}: {intervals}, {cond_sec:.1f}s / {total_sec:.1f}s ({pct:.1f}%)", flush=True)

    # Call demo_for_export with explicit keyword for `conditions` so that
    # `verbose` is not accidentally bound to the wrong parameter.
    result = demo_for_export(df, None, conditions=None, verbose=verbose)
    print('done')

def demo_for_export(df, condition=None, verbose: bool = True):
    import parse
    import nhl_api
    # Determine the team to analyze: prefer condition['team'] when provided; default to 'PHI'
    team_param = 'PHI'
    if isinstance(condition, dict) and 'team' in condition:
        team_param = condition['team']

    # Derive analysis conditions from the passed filter `condition`.
    # If the condition dict contains keys other than 'team', those keys are
    # used as analysis conditions. Otherwise fall back to defaults.
    if isinstance(condition, dict):
        analysis_conditions = {k: v for k, v in condition.items() if k != 'team'}
        if not analysis_conditions:
            analysis_conditions = {'game_state': ['5v5'], 'is_net_empty': [0, 1]}
    else:
        analysis_conditions = {'game_state': ['5v5'], 'is_net_empty': [0, 1]}

    gids = select_team_game(df, team_param)

    # Ensure gids is iterable
    if gids is None:
        return {}
    if not isinstance(gids, (list, tuple, pd.Series)):
        gids = [gids]

    results_per_game = {}
    # Aggregates across all games: fixed labels 'team' (selected team) and 'other' (pooled opponents)
    aggregate_per_condition: Dict[str, Dict[str, float]] = {'team': {}, 'other': {}}  # {'team'|'other': {cond: seconds}}
    aggregate_intervals_per_condition: Dict[str, Dict[str, List[Tuple[float, float]]]] = {'team': {}, 'other': {}}  # {'team'|'other': {cond: intervals}}
    aggregate_intersection_total: Dict[str, float] = {'team': 0.0, 'other': 0.0}  # {'team'|'other': total pooled intersection seconds}
    aggregate_intersection_intervals: Dict[str, List[Tuple[float, float]]] = {'team': [], 'other': []}  # {'team'|'other': unioned intersection intervals}

    # helper to merge intervals (union)
    def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        merged: List[Tuple[float, float]] = []
        if not intervals:
            return merged
        intervals = sorted(intervals, key=lambda x: x[0])
        cur_start, cur_end = intervals[0]
        EPS = 1e-9
        for s, e in intervals[1:]:
            if s <= cur_end + EPS:
                cur_end = max(cur_end, e)
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = s, e
        merged.append((cur_start, cur_end))
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

    for gid in gids:
        # We must not rely on the incoming `df` (it may be pre-filtered).
        # Instead fetch the raw game feed and re-parse it using `nhl_api` + `parse`.
        gdf = None
        feed = None
        # Try to call nhl_api.get_game_feed with an int when possible
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
                    # _elaborate returns the parsed/feature-enriched DataFrame
                    gdf = parse._elaborate(ev_df)
            except Exception:
                gdf = None
        else:
            # If we couldn't fetch the feed, fall back to the input df selection
            try:
                if isinstance(df, pd.DataFrame) and 'game_id' in df.columns:
                    gdf = df[df['game_id'] == gid]
                else:
                    gdf = None
            except Exception:
                gdf = None

        if gdf is None or gdf.empty:
            # nothing to do for this game
            continue

        # determine home/away ids/abbs for opponent inference
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

        # normalize team_param and identify opponent identifier (prefer id, else abb)
        tstr = str(team_param).strip()
        try:
            tid = int(tstr)
        except Exception:
            tid = None
        # determine which side is the selected team in this game
        selected_is_home = False
        selected_is_away = False
        if tid is not None:
            if home_id is not None and str(home_id) == str(tid):
                selected_is_home = True
            if away_id is not None and str(away_id) == str(tid):
                selected_is_away = True
        else:
            tupper = tstr.upper()
            if home_abb is not None and str(home_abb).upper() == tupper:
                selected_is_home = True
            if away_abb is not None and str(away_abb).upper() == tupper:
                selected_is_away = True

        # identify opponent val (id preferred)
        if selected_is_home:
            opp_id = away_id
            opp_abb = away_abb
        elif selected_is_away:
            opp_id = home_id
            opp_abb = home_abb
        else:
            # fallback: pick the other team listed as opponent of PHI
            opp_id = away_id if home_abb == str(team_param).upper() else home_id
            opp_abb = away_abb if home_abb == str(team_param).upper() else home_abb

        team_key = str(team_param)
        opp_key = str(opp_abb or opp_id or 'opponent')

        # compute for both selected team and opponent
        per_game_info = {}
        for side_label, side_team in (('team', team_param), ('opponent', opp_key)):
            # build a team-specific relative df
            # when computing for opponent, pass opponent id/abb where possible
            side_team_val = side_team
            if side_label == 'opponent':
                # prefer numeric id if available
                side_team_val = opp_id if opp_id is not None else (opp_abb if opp_abb is not None else opp_key)

            # annotate df with relative game_state for this side
            df_side = add_game_state_relative_column(gdf.copy(), side_team_val)

            merged_per_condition_local: Dict[str, List[Tuple[float, float]]] = {}
            pooled_seconds_per_condition_local: Dict[str, float] = {}

            times = pd.to_numeric(df_side.get('total_time_elapsed_seconds', pd.Series(dtype=float)), errors='coerce').dropna()
            total_observed = float(times.max() - times.min()) if len(times) >= 2 else 0.0

            for cond_label, cond_def in analysis_conditions.items():
                # cond_def is a list of states (e.g., ['5v5'] or [0,1]).
                all_intervals: List[Tuple[float, float]] = []
                for state in cond_def:
                    # Build the proper condition dict expected by intervals_for_condition
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

                # Merge intervals across all states for this condition and compute pooled seconds
                merged_intervals = _merge_intervals(all_intervals)
                pooled = sum((e - s) for s, e in merged_intervals) if merged_intervals else 0.0
                merged_per_condition_local[str(cond_label)] = merged_intervals
                pooled_seconds_per_condition_local[str(cond_label)] = pooled

            # Store per-game results
            per_game_info[side_label] = {
                'merged_intervals': merged_per_condition_local,
                'pooled_seconds': pooled_seconds_per_condition_local,
                'total_observed': total_observed,
            }

        # For each side, compute the intersection across merged intervals for all conditions
        for side_label, side_info in per_game_info.items():
            cond_interval_lists = list(side_info['merged_intervals'].values())
            # start with first condition intervals, then intersect across the rest
            if cond_interval_lists:
                inter = cond_interval_lists[0]
                for lst in cond_interval_lists[1:]:
                    inter = _intersect_two(inter, lst)
            else:
                inter = []
            pooled_intersection = sum((e - s) for s, e in inter) if inter else 0.0
            # store intersection results in the per-game side info
            side_info['intersection_intervals'] = inter
            side_info['pooled_intersection_seconds'] = pooled_intersection

            # map side_label -> aggregate bucket: 'team' or 'other'
            agg_bucket = 'team' if side_label == 'team' else 'other'
            # accumulate per-condition pooled seconds and union intervals into global aggregates keyed by agg_bucket
            for cond_label, pooled_seconds in side_info['pooled_seconds'].items():
                aggregate_per_condition[agg_bucket].setdefault(cond_label, 0.0)
                aggregate_per_condition[agg_bucket][cond_label] += pooled_seconds
                # union intervals for this condition across games
                existing = aggregate_intervals_per_condition[agg_bucket].setdefault(cond_label, [])
                aggregate_intervals_per_condition[agg_bucket][cond_label] = _merge_intervals(existing + side_info['merged_intervals'].get(cond_label, []))

            # accumulate intersection totals across games (per bucket) and union intersection intervals
            aggregate_intersection_total[agg_bucket] += pooled_intersection
            aggregate_intersection_intervals[agg_bucket] = _merge_intervals(aggregate_intersection_intervals[agg_bucket] + inter)

        # Save this game's per-game info keyed by game id
        results_per_game[gid] = {
            'selected_team': team_key,
            'opponent_team': opp_key,
            'sides': per_game_info,
            'game_total_observed_seconds': total_observed,
        }

    # Recompute aggregate intersection totals from per-game results, but bucket into 'team'/'other'
    recomputed_intersection_total: Dict[str, float] = {'team': 0.0, 'other': 0.0}
    for gid, info in results_per_game.items():
        sides = info.get('sides', {})
        tsec = float(sides.get('team', {}).get('pooled_intersection_seconds', 0.0) or 0.0)
        osec = float(sides.get('opponent', {}).get('pooled_intersection_seconds', 0.0) or 0.0)
        recomputed_intersection_total['team'] += tsec
        recomputed_intersection_total['other'] += osec

    # Use recomputed totals for the final output to avoid accumulation drift
    aggregate_intersection_total = recomputed_intersection_total

    # Prepare the final results structure
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
