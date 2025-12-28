
import sys
import os
import pandas as pd
import logging
import json
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puck import parse

# =============================================================================
# FIXED HELPER FUNCTIONS (LOCAL)
# =============================================================================

def _period_time_to_seconds(time_str):
    if time_str is None:
        return None
    try:
        s = str(time_str).strip()
        if ':' in s:
            mm, ss = s.split(':')
            return int(mm) * 60 + int(ss)
        else:
            return int(float(s))
    except Exception:
        return None

def rink_goal_xs_local():
    return -89.0, 89.0

def calculate_distance_and_angle_local(x, y, goal_x, goal_y=0.0):
    import math
    distance = math.hypot(x - goal_x, y - goal_y)
    vx = x - goal_x
    vy = y - goal_y
    if goal_x < 0:
        rx, ry = 0.0, 1.0
    else:
        rx, ry = 0.0, -1.0
    cross = rx * vy - ry * vx
    dot = rx * vx + ry * vy
    angle_rad_ccw = math.atan2(cross, dot)
    angle_deg = (-math.degrees(angle_rad_ccw)) % 360.0
    return distance, angle_deg

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
    return None

def infer_home_defending_side_from_play(p: dict, game_feed: Optional[dict] = None, events_df=None) -> Optional[str]:
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
    if isinstance(game_feed, dict):
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

    # 3) Fallback heuristic
    try:
        if events_df is not None and hasattr(events_df, 'empty') and not events_df.empty:
            home_id = p.get('home_id') or p.get('homeId') or None
            if home_id is None and 'home_id' in events_df.columns:
                vals = events_df['home_id'].dropna().unique()
                if len(vals) > 0:
                    home_id = vals[0]

            # require numeric x/y and the rink helper
            left_x, right_x = rink_goal_xs_local()

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

def _elaborate(game_feed: pd.DataFrame) -> pd.DataFrame:
    """Take a DataFrame produced by `_game` and derive ML-friendly features.
    
    MONKEY PATCHED VERSION: No imports from .rink
    """

    elaborated_game_feed: List[Dict[str, Any]] = []

    # canonical goal positions from local helper
    left_goal_x, right_goal_x = rink_goal_xs_local()

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
        return pd.DataFrame()

    # Build a temporary DataFrame of the raw rows
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

                if 'event' in period_slice.columns:
                    shots_slice = period_slice[period_slice['event'].isin(shot_types)]
                else:
                    shots_slice = period_slice

                rep = period_slice.iloc[0].to_dict()

                try_df = shots_slice if (isinstance(shots_slice, (list, tuple)) or hasattr(shots_slice, 'shape')) and len(shots_slice) >= 2 else (
                    period_slice if (isinstance(period_slice, (list, tuple)) or hasattr(period_slice, 'shape')) and len(period_slice) >= 2 else None
                )
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

            try:
                rec['period'] = int(rec.get('period')) if rec.get('period') is not None else None
            except Exception:
                rec['period'] = None

            if rec.get('home_team_defending_side') is None:
                inferred_side = period_side_map.get(rec.get('period'))
                if inferred_side is not None:
                    rec['home_team_defending_side'] = inferred_side

            # TIME PARSING LOGIC (CRITICAL FIX)
            period_time_str = rec.get('periodTime') or rec.get('period_time')
            period_time_type = rec.get('periodTimeType') or rec.get('periodTimeType')

            period_elapsed = None
            if period_time_str is not None:
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

            total_elapsed = None
            if rec.get('period') is not None and rec.get('period') >= 1 and period_elapsed is not None:
                p = int(rec['period'])
                if p <= 3:
                    total_elapsed = (p - 1) * 1200 + period_elapsed
                else:
                    total_elapsed = 3 * 1200 + (p - 4) * 300 + period_elapsed

            rec['total_time_elapsed_seconds'] = int(total_elapsed) if total_elapsed is not None else None

            try:
                x = float(rec.get('x'))
                y = float(rec.get('y'))
            except Exception:
                x = None
                y = None
            rec['x'] = x
            rec['y'] = y

            if x is not None and y is not None:
                if rec.get('team_id') == rec.get('home_id'):
                    if rec.get('home_team_defending_side') == 'left':
                        goal_x = right_goal_x
                    elif rec.get('home_team_defending_side') == 'right':
                        goal_x = left_goal_x
                    else:
                        goal_x = right_goal_x
                elif rec.get('team_id') == rec.get('away_id'):
                    if rec.get('home_team_defending_side') == 'left':
                        goal_x = left_goal_x
                    elif rec.get('home_team_defending_side') == 'right':
                        goal_x = right_goal_x
                    else:
                        goal_x = left_goal_x
                else:
                    goal_x = right_goal_x

                distance, angle_deg = calculate_distance_and_angle_local(x, y, goal_x, 0.0)
                rec['distance'] = distance
                rec['angle_deg'] = angle_deg

            else:
                rec['dist_center'] = None
                rec['angle_deg'] = None

            rec.pop('periodTimeType', None)

            elaborated_game_feed.append(rec)

        except Exception as e:
            logging.warning('Error elaborating event: %s', e)
            continue

    try:
        return pd.DataFrame.from_records(elaborated_game_feed)
    except Exception as e:
        logging.warning('Failed to build elaborated DataFrame: %s', e)
        return pd.DataFrame()


# =============================================================================
# MONKEY PATCH AND RUN
# =============================================================================

if __name__ == "__main__":
    print("Monkey-patching puck.parse with local fixes...")
    parse._period_time_to_seconds = _period_time_to_seconds  # MISSING HELPER INJECTION
    parse._elaborate = _elaborate
    parse.infer_home_defending_side_from_play = infer_home_defending_side_from_play
    parse._normalize_side = _normalize_side
    
    print("Starting re-parse for 2025-2026...")
    try:
        # Force re-processing by calling _scrape with process_elaborated=True and save_elaborated=True
        result = parse._scrape(
            season="20252026",
            out_dir='data', 
            use_cache=True,  # Use existing raw JSONs
            verbose=True,
            max_workers=4,   # Parallel processing
            process_elaborated=True,
            save_elaborated=True,
            save_raw=False,
            save_json=False,
            save_csv=True,
            return_elaborated_df=True # FORCE execution past cache check
        )
        print("Completed 20252026.")
    except Exception as e:
        print(f"Error parsing 2025: {e}")
