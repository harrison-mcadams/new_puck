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
        return numeric_games[-1]
    except Exception:
        # fallback to the last in appearance order
        return games[-1]


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

    # Conditions to inspect
    states = ['5v4']
    print('\nGame-state intervals (relative to team):', flush=True)
    for s in states:
        intervals, cond_sec, total_sec = intervals_for_condition(gdf, {'game_state_relative_to_team': s}, verbose=verbose)
        pct = (cond_sec / total_sec * 100.0) if total_sec > 0 else 0.0
        print(f"  {s}: {intervals}, {cond_sec:.1f}s / {total_sec:.1f}s ({pct:.1f}%)", flush=True)

    # is_net_empty condition: special handling as it's not a simple state value
    print('\nis_net_empty intervals (relative to team):', flush=True)
    for empty in [0, 1]:
        intervals, cond_sec, total_sec = intervals_for_condition(gdf, {'is_net_empty': empty}, verbose=verbose)
        pct = (cond_sec / total_sec * 100.0) if total_sec > 0 else 0.0
        print(f"  is_net_empty={empty}: {intervals}, {cond_sec:.1f}s / {total_sec:.1f}s ({pct:.1f}%)", flush=True)


if __name__ == "__main__":
    import sys
    import os
    import traceback

    verbose = False
    if len(sys.argv) > 1:
        if sys.argv[1] in ('-v', '--verbose'):
            verbose = True
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Usage: python timing_brainstorming.py [-v|--verbose]")
            sys.exit(1)

    # Ensure data directory exists
    if not Path("data").exists():
        print("Data directory 'data' not found; please run from the project root")
        sys.exit(1)

    # Run demo for the default team/game
    try:
        demo_for_team_game(verbose=verbose)
    except Exception as e:
        print(f"Error in demo: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
