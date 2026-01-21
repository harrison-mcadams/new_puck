from typing import Dict, Any, List, Optional
import pandas as pd
from collections import defaultdict

def classify_player_roles(df_shifts: pd.DataFrame, game_length_seconds: Optional[float] = None) -> Dict[str, Any]:
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
    try:
        df['player_id_str'] = df['player_id'].astype(str)
        df['team_id_str'] = df['team_id'].astype(str)
    except Exception:
        # Fallback if columns missing
        if 'player_id' not in df.columns:
            return {'roles': out_roles, 'by_player': stats}
        df['player_id_str'] = df['player_id'].astype(str)
        if 'team_id' in df.columns:
            df['team_id_str'] = df['team_id'].astype(str)
        else:
             df['team_id_str'] = 'UNK'

    # gather per-player stats and explicit positions
    explicit_pos_map: Dict[str, str] = {}
    
    # Defensive grouping
    if 'player_id_str' not in df.columns:
         return {'roles': out_roles, 'by_player': stats}

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
            t_series = df.loc[df['player_id_str'] == pid, 'team_id_str']
            if not t_series.empty:
                t = t_series.iloc[0]
            else:
                t = 'UNK'
        except Exception:
            t = 'UNK'
        per_team_players[t].append(pid)

    # First, assign explicit positions
    for pid, pos in explicit_pos_map.items():
        try:
            s_pos = pos.upper()
            if s_pos.startswith('G') or 'GOAL' in s_pos:
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
