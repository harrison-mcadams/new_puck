# Revised plotting routine
# The main input is an events DataFrame. Provide flexibility in what events
# are plotted and what plot symbols/colors are used for each event type.

from typing import List, Dict, Optional, Tuple
import os
import matplotlib
# Only force the non-interactive 'Agg' backend when explicitly requested
# (e.g. in headless CI or via `export FORCE_AGG=1`). Otherwise leave the
# backend selection to matplotlib (which enables interactive backends).
if os.environ.get('FORCE_AGG', '0') == '1':
    try:
        matplotlib.use('Agg')
    except Exception:
        pass
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
from rink import draw_rink, rink_goal_xs, rink_half_height_at_x
import matplotlib.colors as mcolors


def _events(events: pd.DataFrame, events_to_plot: Optional[List[str]] = None) -> pd.DataFrame:
    """Return a filtered view of `events` containing only the requested event types.

    Parameters
    - events: DataFrame with at least columns ['event','x','y']
    - events_to_plot: list of event type names (case-insensitive) or None to keep all

    Returns a DataFrame (possibly empty) containing only rows whose 'event'
    value is in events_to_plot.
    """
    if events_to_plot is None:
        return events.copy()
    # Normalize to set of lowercase names for comparison
    wanted = {e.strip().lower() for e in events_to_plot}

    # Filter
    mask = events['event'].astype(str).str.strip().str.lower().isin(wanted)
    return events.loc[mask].copy()

def adjust_xy_for_homeaway(df, split_mode: str = 'home_away', team_for_heatmap: Optional[object] = None):
    # orient all sides such as home shots are directed to the left,
    # away shots are directed to the right. also need to flip y accordingly

    # If another routine already produced adjusted coordinates, don't recompute.
    # Check that both columns exist and at least one valid value is present.
    try:
        if df is not None and 'x_a' in df.columns and 'y_a' in df.columns:
            if df['x_a'].notna().any() and df['y_a'].notna().any():
                return df.copy()
    except Exception:
        # fall through and recompute if anything unexpected happens
        pass

    # The approach:
    # - For each row determine which goal the shooter was attacking (attacked_goal_x)
    #   using the same convention as parse._elaborate (home_team_defending_side)
    # - Define desired goal: home -> left_goal_x, away -> right_goal_x
    # - If attacked_goal_x != desired_goal_x, rotate the point 180deg by
    #   setting x_a = -x, y_a = -y. Otherwise keep x_a = x, y_a = y.
    # This yields coordinates standardized so home events face left and away events face right.

    if df is None or df.shape[0] == 0:
        return df

    left_goal_x, right_goal_x = rink_goal_xs()

    # Ensure columns exist to avoid KeyError
    cols = df.columns
    team_col = 'team_id' if 'team_id' in cols else None
    home_col = 'home_id' if 'home_id' in cols else None
    away_col = 'away_id' if 'away_id' in cols else None
    defend_col = 'home_team_defending_side' if 'home_team_defending_side' in cols else None

    x_vals = df['x'].astype(float) if 'x' in cols else pd.Series([float('nan')] * len(df), index=df.index)
    y_vals = df['y'].astype(float) if 'y' in cols else pd.Series([float('nan')] * len(df), index=df.index)

    attacked_goal = []
    desired_goal = []

    for idx, row in df.iterrows():
        t_id = row.get('team_id') if team_col else None
        h_id = row.get('home_id') if home_col else None
        a_id = row.get('away_id') if away_col else None
        home_def_side = row.get('home_team_defending_side') if defend_col else None

        # determine attacked goal using same logic as parse
        if t_id is not None and h_id is not None and a_id is not None:
            if t_id == h_id:
                # shooter is home -> attacking goal is opposite what home defends
                if home_def_side == 'left':
                    ag = right_goal_x
                elif home_def_side == 'right':
                    ag = left_goal_x
                else:
                    ag = right_goal_x
            elif t_id == a_id:
                # shooter is away -> attacking goal is side not defended by home
                if home_def_side == 'left':
                    ag = left_goal_x
                elif home_def_side == 'right':
                    ag = right_goal_x
                else:
                    ag = left_goal_x
            else:
                # unknown team -> fallback to right
                ag = right_goal_x
        else:
            # insufficient info -> assume attacking right (fallback)
            ag = right_goal_x

        # desired goal depending on split_mode
        if split_mode == 'team_not_team' and team_for_heatmap is not None:
            # determine whether this shooter belongs to the target team
            is_team = False
            try:
                # numeric id match
                if str(team_for_heatmap).strip().isdigit():
                    is_team = (str(t_id) == str(int(team_for_heatmap)))
                else:
                    # try matching abbreviations
                    if 'home_abb' in df.columns and str(row.get('home_abb')).upper() == str(team_for_heatmap).upper():
                        is_team = (t_id is not None and str(t_id) == str(row.get('home_id')))
                    if not is_team and 'away_abb' in df.columns and str(row.get('away_abb')).upper() == str(team_for_heatmap).upper():
                        is_team = (t_id is not None and str(t_id) == str(row.get('away_id')))
            except Exception:
                is_team = False
            if is_team:
                dg = left_goal_x
            else:
                dg = right_goal_x
        else:
            # legacy home/away: desired goal depending on whether shooter is home or away
            if t_id is not None and h_id is not None:
                if str(t_id) == str(h_id):
                    dg = left_goal_x
                elif str(t_id) == str(a_id):
                    dg = right_goal_x
                else:
                    dg = right_goal_x
            else:
                dg = right_goal_x

        attacked_goal.append(ag)
        desired_goal.append(dg)

    attacked_goal = pd.Series(attacked_goal, index=df.index)
    desired_goal = pd.Series(desired_goal, index=df.index)

    # compute adjusted coordinates
    x_a = []
    y_a = []
    for xi, yi, ag, dg in zip(x_vals, y_vals, attacked_goal, desired_goal):
        try:
            if pd.isna(xi) or pd.isna(yi):
                x_a.append(float('nan'))
                y_a.append(float('nan'))
            else:
                if ag == dg:
                    x_a.append(xi)
                    y_a.append(yi)
                else:
                    # rotate 180 degrees: (x,y) -> (-x, -y)
                    x_a.append(-float(xi))
                    y_a.append(-float(yi))
        except Exception:
            x_a.append(float('nan'))
            y_a.append(float('nan'))

    df = df.copy()
    df['x_a'] = pd.Series(x_a, index=df.index)
    df['y_a'] = pd.Series(y_a, index=df.index)

    return df

def plot_events(
    events: pd.DataFrame,
    events_to_plot: Optional[List[str]] = None,
    event_styles: Optional[Dict[str, Dict]] = None,
    ax: Optional[plt.Axes] = None,
    rink: bool = True,
    out_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 4.5),
    title: Optional[str] = None,
    return_heatmaps: bool = False,
    # heatmap mode: 'home_away' keeps legacy behavior, 'team_not_team' groups
    # xG into 'team' vs 'not_team' based on `team_for_heatmap`.
    heatmap_split_mode: str = 'home_away',
    team_for_heatmap: Optional[object] = None,
    # Optional summary statistics (dict) produced by analyze.xgs_map; used to
    # display aggregated xG per 60 values on the top text block.
    summary_stats: Optional[Dict[str, float]] = None,
    conditions: Optional[Dict] = None,
    plot_kwargs: Optional[Dict] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot event locations onto a rink schematic.

    Parameters
    - events: DataFrame with at least columns ['x','y','event','team_id','home_id','away_id']
    - events_to_plot: list of event types (case-insensitive). If None, plot all rows.
    - event_styles: mapping event_type -> style dict. The style dict may include:
        - 'marker' (matplotlib marker, default 'o' for shots, 'x' for goals)
        - 'size' (marker size)
        - 'home_color' (color for home team attempts)
        - 'away_color' (color for away team attempts)
    - ax: optional matplotlib Axes to plot onto; if None a new figure is created
    - rink: whether to draw the rink background (uses rink.draw_rink)
    - out_path: if provided, save the figure to this path
    - figsize: size for new figure
    - title: optional plot title
   - return_heatmaps: if True return (fig, ax, {{'home': heat_home, 'away': heat_away}})

    Returns (fig, ax).
    """
    # initialize heatmap outputs for backward compatibility
    heat_home = None
    heat_away = None
    heat_team = None
    heat_not_team = None

    if plot_kwargs is None:
        plot_kwargs = {}

    if out_path is None:
        out_path = plot_kwargs.get('out_path')
    if team_for_heatmap is None:
        team_for_heatmap = plot_kwargs.get('team_for_heatmap')
    if summary_stats is None:
        summary_stats = plot_kwargs.get('summary_stats')

    # Basic validation
    if not isinstance(events, pd.DataFrame):
        raise TypeError('events must be a pandas DataFrame')

    # Default event styles
    default_styles = {
        'shot-on-goal': {'marker': 'o', 'size': 20, 'home_color': 'black',
                  'away_color': 'orange'},
        'missed-shot': {'marker': 'o', 'size': 15, 'home_color': 'black',
                         'away_color': 'orange'},
        'blocked-shot': {'marker': 'o', 'size': 15, 'home_color': 'black',
                         'away_color': 'orange'},
        'goal': {'marker': 'x', 'size': 60, 'home_color': 'black', 'away_color': 'orange'},
    }
    # Merge user-provided event_styles into defaults (case-insensitive keys)
    merged_styles = {}
    if event_styles:
        # user keys might be like 'SHOT' or 'Shot'
        for k, v in event_styles.items():
            merged_styles[k.strip().lower()] = v.copy()
    # fill with defaults when available
    for k, v in default_styles.items():
        if k not in merged_styles:
            merged_styles[k] = v
        else:
            # fill missing fields from default
            for fk, fv in v.items():
                merged_styles[k].setdefault(fk, fv)

    # filter events
    # Normalize requested event types and detect if caller wants an xG heatmap
    requested = [e.strip().lower() for e in events_to_plot] if events_to_plot is not None else None
    wants_xgs = (requested is not None and 'xgs' in requested)

    # (no debugger-based skip — plot all requested features including xgs)

    # Build a filtered list for _events that excludes 'xgs' (it's not an event name)
    if requested is not None:
        filtered_events = [e for e in events_to_plot if e.strip().lower() != 'xgs']
    else:
        filtered_events = None

    # filter events for normal scatter plotting (exclude 'xgs' token)
    df = _events(events, filtered_events)
    # Only compute adjusted coordinates if they are not already present or are all NaN.
    try:
        need_adjust = ('x_a' not in df.columns) or ('y_a' not in df.columns) or df['x_a'].isna().all() or df['y_a'].isna().all()
    except Exception:
        need_adjust = True
    if need_adjust:
        df = adjust_xy_for_homeaway(df, split_mode=heatmap_split_mode, team_for_heatmap=team_for_heatmap)

    # make sure we have numeric x,y
    if 'x' not in df.columns or 'y' not in df.columns:
        raise KeyError("events DataFrame must contain numeric 'x' and 'y' columns")

    # Prepare plotting axes. If caller supplied an `ax`, use it. Otherwise create
    # a single figure/axes and reserve a small top margin for the text block.
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        # Reserve a little area at the top for the summary text. Increasing
        # `top` pulls the rink up; decreasing it leaves more room for text.
        top = 0.82
        fig.subplots_adjust(top=top)
        _text_ax = None
    else:
        fig = ax.figure
        _text_ax = None

    if rink:
        try:
            draw_rink(ax=ax)
        except Exception:
            # If draw_rink fails for any reason, fall back to simple limits
            ax.set_xlim(-100, 100)
            ax.set_ylim(-50, 50)
            ax.set_aspect('equal', adjustable='box')
            ax.axis('off')

    # If no events to plot, only return early when the caller did not request
    # an xG heatmap. If 'xgs' was requested we still want to draw the heatmap
    # (it uses the original `events` DataFrame), so do not return in that case.
    if df.shape[0] == 0 and not wants_xgs:
        if title:
            ax.set_title(title)
        return fig, ax

    # Determine group membership depending on split mode
    if heatmap_split_mode == 'orient_all_left':
        # All shots are oriented to the left, so all events are in one group
        is_group1 = pd.Series([True] * len(df), index=df.index)
        group1_name = 'all'
        group2_name = None
    elif heatmap_split_mode == 'team_not_team' and team_for_heatmap is not None:
        # compute boolean series: True if event belongs to the requested team
        def _is_team_row(r):
            try:
                tid = None
                if str(team_for_heatmap).strip().isdigit():
                    tid = str(int(team_for_heatmap))
                tupper = None if tid is not None else str(team_for_heatmap).upper()
                team_id = r.get('team_id')
                if tid is not None and team_id is not None:
                    return str(team_id) == tid
                if tupper is not None:
                    home_abb = r.get('home_abb')
                    away_abb = r.get('away_abb')
                    if home_abb is not None and str(home_abb).upper() == tupper:
                        return team_id is not None and str(team_id) == str(r.get('home_id'))
                    if away_abb is not None and str(away_abb).upper() == tupper:
                        return team_id is not None and str(team_id) == str(r.get('away_id'))
                return False
            except Exception:
                return False

        is_group1 = df.apply(_is_team_row, axis=1)
        group1_name = str(team_for_heatmap)
        group2_name = 'Opponents'
    else:
        # legacy home/away grouping: True if team_id == home_id
        if all(c in df.columns for c in ('team_id', 'home_id')):
            is_group1 = df['team_id'] == df['home_id']
        else:
            is_group1 = pd.Series([False] * len(df), index=df.index)
        group1_name = None
        group2_name = None

    # For legend handles
    handles = []
    labels = []
    # ensure ev_type is defined even if grouping yields no iterations
    ev_type = None

    # Group by event type (case-insensitive)
    df = df.copy()
    df['_event_lc'] = df['event'].astype(str).str.strip().str.lower()
    for ev_type, group in df.groupby('_event_lc'):
        style = merged_styles.get(ev_type, {'marker': 'o', 'size': 30, 'home_color': 'black', 'away_color': 'gray'})
        m = style.get('marker', 'o')
        s = style.get('size', 30)
        home_c = style.get('home_color', 'black')
        away_c = style.get('away_color', 'gray')

        # plot home and away separately for consistent coloring
        grp_home = group[is_group1.loc[group.index]]
        grp_away = group[~is_group1.loc[group.index]]

        # choose adjusted coords if present
        xcol = 'x_a' if 'x_a' in group.columns else 'x'
        ycol = 'y_a' if 'y_a' in group.columns else 'y'

        grp1 = group[is_group1.loc[group.index]]
        grp2 = group[~is_group1.loc[group.index]]

        # choose colors depending on split mode
        if heatmap_split_mode == 'team_not_team':
            # event_styles may include team_color/not_team_color overrides
            evx = merged_styles.get(ev_type, {})
            c1 = evx.get('team_color', evx.get('home_color', 'black'))
            c2 = evx.get('not_team_color', evx.get('away_color', 'orange'))
        else:
            c1 = home_c
            c2 = away_c

        if not grp1.empty:
            h = ax.scatter(grp1[xcol], grp1[ycol], c=c1, marker=m, s=s, edgecolors='none', zorder=5, alpha=0.7)
            if ev_type not in labels:
                handles.append(h)
                labels.append(ev_type)
        if not grp2.empty:
            h2 = ax.scatter(grp2[xcol], grp2[ycol], c=c2, marker=m, s=s, edgecolors='none', zorder=5, alpha=0.7)
            if ev_type not in labels:
                handles.append(h2)
                labels.append(ev_type)

    # tidy up
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    # Title and legend intentionally suppressed here; summary text above the rink
    # is used instead (suptitle + stacked summary lines). Do not set an axes title
    # and do not show the legend to keep the image clean.

    # --- SUMMARY TEXT BLOCK ---
    # compute representative team ids/abbrevs (for legacy home/away display)
    home_id = None
    away_id = None
    if 'home_id' in df.columns:
        vals = df['home_id'].dropna().unique()
        if len(vals) > 0:
            home_id = vals[0]
    if 'away_id' in df.columns:
        vals = df['away_id'].dropna().unique()
        if len(vals) > 0:
            away_id = vals[0]

    # If the filtered plotting dataframe `df` is empty (for example when the
    # caller only requested 'xgs' heatmap), fall back to the original
    # `events` DataFrame for summary metadata (team abbreviations, ids).
    summary_df = df if (df is not None and not df.empty) else events

    # Safely extract home/away abbreviations
    if 'home_abb' in summary_df.columns and not summary_df['home_abb'].dropna().empty:
        home_name = str(summary_df['home_abb'].dropna().unique()[0])
    else:
        home_name = ''
    if 'away_abb' in summary_df.columns and not summary_df['away_abb'].dropna().empty:
        away_name = str(summary_df['away_abb'].dropna().unique()[0])
    else:
        away_name = ''

    # compute goals (count events with 'goal')
    ev_lc = events['event'].astype(str).str.strip().str.lower() if 'event' in events.columns else pd.Series([], dtype=object)
    is_goal = ev_lc == 'goal'
    # Count goals using the original events DataFrame (more complete than filtered df)
    if heatmap_split_mode == 'team_not_team' and team_for_heatmap is not None:
        # compute group1 (team) vs group2 (not team) goal counts
        def _is_team_row_events(r):
            try:
                if str(team_for_heatmap).strip().isdigit():
                    return str(r.get('team_id')) == str(int(team_for_heatmap))
                tupper = str(team_for_heatmap).upper()
                home_abb = r.get('home_abb')
                away_abb = r.get('away_abb')
                if home_abb is not None and str(home_abb).upper() == tupper:
                    return str(r.get('team_id')) == str(r.get('home_id'))
                if away_abb is not None and str(away_abb).upper() == tupper:
                    return str(r.get('team_id')) == str(r.get('away_id'))
                return False
            except Exception:
                return False

        mask_team = events.apply(_is_team_row_events, axis=1) if 'team_id' in events.columns or 'home_abb' in events.columns else pd.Series([False] * len(events), index=events.index)
        home_goals = int(((mask_team) & is_goal).sum())
        away_goals = int(((~mask_team) & is_goal).sum())
    else:
        if 'team_id' in events.columns and home_id is not None:
            home_goals = int(((events['team_id'].astype(str) == str(home_id)) & is_goal).sum())
            away_goals = int(((events['team_id'].astype(str) != str(home_id)) & is_goal).sum())
        else:
            home_goals = int(is_goal.sum())
            away_goals = 0

    # shot attempts totals (shot-on-goal, missed-shot, blocked-shot)
    # Use the normalized lowercase event column computed earlier on the full events df.
    shot_attempt_types = {'shot-on-goal', 'missed-shot', 'blocked-shot'}
    ev_lc_full = events['event'].astype(str).str.strip().str.lower() if 'event' in events.columns else pd.Series([], dtype=object)
    attempt_mask = ev_lc_full.isin(shot_attempt_types)
    attempts_df = events.loc[attempt_mask]

    if heatmap_split_mode == 'team_not_team' and team_for_heatmap is not None:
        # team vs not_team attempts
        def _is_team_row_events(r):
            try:
                if str(team_for_heatmap).strip().isdigit():
                    return str(r.get('team_id')) == str(int(team_for_heatmap))
                tupper = str(team_for_heatmap).upper()
                home_abb = r.get('home_abb')
                away_abb = r.get('away_abb')
                if home_abb is not None and str(home_abb).upper() == tupper:
                    return str(r.get('team_id')) == str(r.get('home_id'))
                if away_abb is not None and str(away_abb).upper() == tupper:
                    return str(r.get('team_id')) == str(r.get('away_id'))
                return False
            except Exception:
                return False

        mask_attempts_team = attempts_df.apply(_is_team_row_events, axis=1) if not attempts_df.empty else pd.Series([], dtype=bool)
        home_attempts = int(mask_attempts_team.sum())
        away_attempts = int((~mask_attempts_team).sum() if not attempts_df.empty else 0)
    else:
        if 'team_id' in events.columns and home_id is not None:
            home_attempts = int((attempts_df['team_id'].astype(str) == str(home_id)).sum())
            away_attempts = int((attempts_df['team_id'].astype(str) != str(home_id)).sum())
        else:
            xs_series = attempts_df['x_a'] if 'x_a' in attempts_df.columns else (attempts_df['x'] if 'x' in attempts_df.columns else pd.Series([]))
            home_attempts = int((xs_series < 0).sum())
            away_attempts = int((xs_series >= 0).sum())

    total_attempts = home_attempts + away_attempts
    if total_attempts > 0:
        home_shot_pct = 100.0 * home_attempts / total_attempts
        away_shot_pct = 100.0 * away_attempts / total_attempts
    else:
        home_shot_pct = away_shot_pct = 0.0

    # Calculate xG totals with correct grouping
    home_xg = 0.0
    away_xg = 0.0
    have_xg = False

    if summary_stats and 'team_xgs' in summary_stats:
        try:
            home_xg = float(summary_stats['team_xgs'])
            away_xg = float(summary_stats['other_xgs'])
            have_xg = True
        except Exception:
            pass

    if not have_xg and 'xgs' in df.columns:
        try:
            have_xg = True
            # If team_not_team, we need to sum based on team membership
            if heatmap_split_mode == 'team_not_team' and team_for_heatmap is not None:
                team_xg_sum = 0.0
                other_xg_sum = 0.0
                
                def _is_team_row_local(r):
                    try:
                        if str(team_for_heatmap).strip().isdigit():
                            return str(r.get('team_id')) == str(int(team_for_heatmap))
                        tupper = str(team_for_heatmap).upper()
                        home_abb = r.get('home_abb')
                        away_abb = r.get('away_abb')
                        if home_abb is not None and str(home_abb).upper() == tupper:
                            return str(r.get('team_id')) == str(r.get('home_id'))
                        if away_abb is not None and str(away_abb).upper() == tupper:
                            return str(r.get('team_id')) == str(r.get('away_id'))
                        return False
                    except Exception:
                        return False

                for _, r in df.iterrows():
                    val = r.get('xgs')
                    try:
                        xv = float(val)
                    except Exception:
                        continue
                    if _is_team_row_local(r):
                        team_xg_sum += xv
                    else:
                        other_xg_sum += xv
                
                home_xg = team_xg_sum
                away_xg = other_xg_sum
            else:
                # Legacy home/away
                for _, r in df.iterrows():
                    val = r.get('xgs')
                    try:
                        xv = float(val)
                    except Exception:
                        continue
                    if 'team_id' in df.columns and home_id is not None and str(r.get('team_id')) == str(home_id):
                        home_xg += xv
                    else:
                        away_xg += xv
        except Exception:
            have_xg = False

    # Determine if this is a season summary for a specific team
    n_games = int(summary_stats.get('n_games', 0)) if summary_stats else 0
    is_season_summary = (heatmap_split_mode == 'team_not_team' and team_for_heatmap is not None and n_games > 1)
    
    # Prepare arguments for add_summary_text
    text_stats = summary_stats.copy() if summary_stats else {}
    
    text_stats['home_goals'] = home_goals
    text_stats['away_goals'] = away_goals
    text_stats['home_attempts'] = home_attempts
    text_stats['away_attempts'] = away_attempts
    text_stats['home_shot_pct'] = home_shot_pct
    text_stats['away_shot_pct'] = away_shot_pct
    
    if have_xg:
        text_stats['home_xg'] = home_xg
        text_stats['away_xg'] = away_xg
        text_stats['have_xg'] = True
    else:
        text_stats['have_xg'] = False
        
    if is_season_summary:
        main_title = f"Season Summary for {team_for_heatmap}"
    else:
        # Ensure we have valid names
        h_name = home_name if home_name else "Home"
        a_name = away_name if away_name else "Away"
        main_title = f"{h_name} vs {a_name}"
        

        
    # Adjust layout to make room for summary text
    plt.subplots_adjust(top=0.75, bottom=0.05)
        
    add_summary_text(
        ax=ax,
        stats=text_stats,
        main_title=main_title,
        is_season_summary=is_season_summary,
        team_name=str(team_for_heatmap) if team_for_heatmap else None,
        filter_str=plot_kwargs.get('filter_str')
    )

    # No need to set individual y variables anymore since we loop

    # NOTE: do not save here — we need to overlay heatmap (if requested)
    # before writing the final image. Saving occurs after the heatmap block.

    # If an xG heatmap was requested, compute it from the original `events` DataFrame
    # (not the event-filtered `df`) so we include all shot attempts that have xgs.
    if wants_xgs:
        # default heatmap params
        hm_sigma = 6.0  # feet
        hm_res = 1.0    # grid resolution in feet
        hm_cmap = 'viridis'
        hm_alpha = 0.6
        # allow overrides via event_styles['xgs']
        if event_styles and isinstance(event_styles, dict):
            evx = event_styles.get('xgs') or event_styles.get('xg') or {}
            try:
                hm_sigma = float(evx.get('sigma', hm_sigma))
            except Exception:
                pass
            try:
                hm_res = float(evx.get('res', hm_res))
            except Exception:
                pass
            hm_cmap = evx.get('cmap', hm_cmap)
            hm_alpha = evx.get('alpha', hm_alpha)

        # select rows with numeric xgs and locations
        events_with_xg = events.copy()
        if 'xgs' in events_with_xg.columns:
            events_with_xg = events_with_xg[pd.to_numeric(events_with_xg['xgs'], errors='coerce').notna()].copy()
        else:
            events_with_xg = events_with_xg[[]]

        heat_team = None
        heat_not_team = None
        heat_home = None
        heat_away = None
        heat_all_arr = None

        if not events_with_xg.empty:
            # ensure adjusted coords exist; if either 'x_a' or 'y_a' is missing
            # (for example they were not created earlier), compute them now.
            if ('x_a' not in events_with_xg.columns) or ('y_a' not in events_with_xg.columns):
                events_with_xg = adjust_xy_for_homeaway(events_with_xg, split_mode=heatmap_split_mode, team_for_heatmap=team_for_heatmap)

            xcol = 'x_a' if 'x_a' in events_with_xg.columns else 'x'
            ycol = 'y_a' if 'y_a' in events_with_xg.columns else 'y'

            # helper to convert heat -> rgba image (reused)
            def heat_to_rgba(h, color, alpha_scale=hm_alpha):
                try:
                    maxv = float(np.nanmax(h)) if np.nanmax(h) > 0 else 0.0
                except Exception:
                    maxv = 0.0
                if maxv <= 0:
                    return None
                norm = np.clip(h / maxv, 0.0, 1.0)
                rgba = np.zeros((h.shape[0], h.shape[1], 4), dtype=float)
                r, g, b, _ = mcolors.to_rgba(color)
                rgba[..., 0] = r
                rgba[..., 1] = g
                rgba[..., 2] = b
                rgba[..., 3] = norm * alpha_scale
                return rgba

            # Use the canonical heatmap computation in analyze.compute_xg_heatmap_from_df
            try:
                import analyze
                if heatmap_split_mode == 'team_not_team' and team_for_heatmap is not None:
                    # team_not_team mode: compute two separate maps (team and not_team)
                    gx, gy, heat_team, team_xg, team_seconds = analyze.compute_xg_heatmap_from_df(
                        events_with_xg, grid_res=hm_res, sigma=hm_sigma, x_col=xcol, y_col=ycol, amp_col='xgs', normalize_per60=False, selected_team=team_for_heatmap, selected_role='team')
                    _, _, heat_not_team, not_team_xg, not_team_seconds = analyze.compute_xg_heatmap_from_df(
                        events_with_xg, grid_res=hm_res, sigma=hm_sigma, x_col=xcol, y_col=ycol, amp_col='xgs', normalize_per60=False, selected_team=team_for_heatmap, selected_role='other')
                    heat_team_arr = heat_team
                    heat_not_team_arr = heat_not_team
                elif heatmap_split_mode == 'orient_all_left':
                    # All shots oriented left, single heatmap
                    gx, gy, heat_all, all_xg, all_seconds = analyze.compute_xg_heatmap_from_df(
                        events_with_xg, grid_res=hm_res, sigma=hm_sigma, x_col=xcol, y_col=ycol, amp_col='xgs', normalize_per60=False)
                    # Overlay as a single color (e.g., black)
                    all_color = 'black'

                    heat_home = None
                    heat_away = None
                    heat_team = None
                    heat_not_team = None
                    heat_all_arr = heat_all
                else:
                    # legacy home/away: detect canonical home_id and use that as selected_team
                    home_id_val = None
                    if 'home_id' in events_with_xg.columns:
                        vals = events_with_xg['home_id'].dropna().unique()
                        if len(vals) > 0:
                            home_id_val = vals[0]
                    elif 'home_id' in events.columns:
                        vals = events['home_id'].dropna().unique()
                        if len(vals) > 0:
                            home_id_val = vals[0]
                    gx, gy, heat_home_arr, home_xg, home_seconds = analyze.compute_xg_heatmap_from_df(
                        events_with_xg, grid_res=hm_res, sigma=hm_sigma, x_col=xcol, y_col=ycol, amp_col='xgs', normalize_per60=False, selected_team=home_id_val, selected_role='team')
                    _, _, heat_away_arr, away_xg, away_seconds = analyze.compute_xg_heatmap_from_df(
                        events_with_xg, grid_res=hm_res, sigma=hm_sigma, x_col=xcol, y_col=ycol, amp_col='xgs', normalize_per60=False, selected_team=home_id_val, selected_role='other')

                    # alias names to match previous expectations
                    heat_team_arr = heat_home_arr
                    heat_not_team_arr = heat_away_arr

                # compute extent from gx, gy
                extent = (gx[0] - hm_res / 2.0, gx[-1] + hm_res / 2.0, gy[0] - hm_res / 2.0, gy[-1] + hm_res / 2.0)

                # pick colors depending on split mode (allow overrides via event_styles)
                if heatmap_split_mode == 'team_not_team':
                    team_color = 'black'
                    not_team_color = 'orange'
                    if event_styles and isinstance(event_styles, dict):
                        evx = event_styles.get('xgs') or {}
                        team_color = evx.get('team_color', team_color)
                        not_team_color = evx.get('not_team_color', not_team_color)
                else:
                    home_color = 'black'
                    away_color = 'orange'
                    if event_styles and isinstance(event_styles, dict):
                        evx = event_styles.get('xgs') or {}
                        home_color = evx.get('home_color', home_color)
                        away_color = evx.get('away_color', away_color)



                # render overlays
                if heatmap_split_mode == 'team_not_team':
                    rgba_team = heat_to_rgba(heat_team_arr, team_color) if heat_team_arr is not None else None
                    rgba_not_team = heat_to_rgba(heat_not_team_arr, not_team_color) if heat_not_team_arr is not None else None
                    if rgba_team is not None:
                        ax.imshow(rgba_team, extent=extent, origin='lower', zorder=1)
                    if rgba_not_team is not None:
                        ax.imshow(rgba_not_team, extent=extent, origin='lower', zorder=2)
                    # assign back to heat_team/heat_not_team for return
                    heat_team = heat_team_arr
                    heat_not_team = heat_not_team_arr
                elif heatmap_split_mode == 'orient_all_left':
                    rgba_all = heat_to_rgba(heat_all_arr, 'black') if heat_all_arr is not None else None
                    if rgba_all is not None:
                        ax.imshow(rgba_all, extent=extent, origin='lower', zorder=1)
                    heat_team = None
                    heat_not_team = None
                    heat_all_arr = heat_all_arr
                else:
                    rgba_home = heat_to_rgba(heat_team_arr, home_color) if heat_team_arr is not None else None
                    rgba_away = heat_to_rgba(heat_not_team_arr, away_color) if heat_not_team_arr is not None else None
                    if rgba_home is not None:
                        ax.imshow(rgba_home, extent=extent, origin='lower', zorder=1)
                    if rgba_away is not None:
                        ax.imshow(rgba_away, extent=extent, origin='lower', zorder=2)
                    heat_home = heat_team_arr
                    heat_away = heat_not_team_arr
            except Exception as e:
                # If analyze.compute_xg_heatmap_from_df or plotting overlay fails, fall back to no overlay
                try:
                    print('plot_events: failed to compute or render heatmap via analyze.compute_xg_heatmap_from_df:', e)
                except Exception:
                    pass

    # final save (after heatmap overlay)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        try:
            print(f"Saved plot to {out_path}")
        except Exception:
            pass

    # Return consistent shapes:
    # - if return_heatmaps is True -> (fig, ax, {'team': heat_team, 'not_team': heat_not_team})
    # - if orient_all_left -> (fig, ax, {'all': heat_all_arr})
    # - else -> (fig, ax)
    if return_heatmaps:
        if heatmap_split_mode == 'team_not_team':
            return fig, ax, {'team': heat_team, 'not_team': heat_not_team}
        elif heatmap_split_mode == 'orient_all_left':
            return fig, ax, {'all': heat_all_arr}
        else:
            return fig, ax, {'home': heat_home, 'away': heat_away}

    return fig, ax

def _game(gameID, conditions=None, plot_kwargs=None):
    """Simple helper to create a shot/goal plot for a single game.

    Parameters
    - gameID: game identifier (string) passed to `fit_xgs.analyze_game`
    - conditions: optional dict to control behavior. Supported keys:
        - 'events': list of event types to plot (case-insensitive). Example:
            ['shot-on-goal','goal','xgs']
        - 'out_path': path to save the resulting image
        - 'figsize': tuple for figure size
        - 'title': optional title string (note plot_events prefers summary text)
        - 'return_heatmaps': bool whether to return heatmap arrays
        - 'heatmap_split_mode': 'home_away' (default) or 'team_not_team'
        - 'team_for_heatmap': team id or abbreviation when using 'team_not_team'
        - 'event_styles': optional style overrides passed to plot_events
        - 'summary_stats': optional dict of summary stats to display above rink

    Returns
    - If return_heatmaps is False: (fig, ax)
    - If return_heatmaps is True: (fig, ax, heatmap_dict)

    The function is intentionally small: it obtains the game's events via
    `fit_xgs.analyze_game`, applies minimal sanity checks, and calls
    `plot_events` with straightforward mappings from `conditions`.
    """
    import fit_xgs  # local import so this module can be imported without heavy deps
    import parse as _parse

    if conditions is None:
        conditions = {}

    # Obtain the game DataFrame. Support two input modes for convenience:
    # 1) gameID is a game identifier (string/int) -> call fit_xgs.analyze_game(gameID)
    # 2) gameID is itself a DataFrame of events for one or more games -> use it directly
    df = None
    inferred_game_id = None
    if isinstance(gameID, pd.DataFrame):
        df = gameID.copy()
        # try to infer a representative game id (if games concatenated, may be multiple)
        try:
            gids = df.get('game_id')
            if gids is not None:
                unique_gids = gids.dropna().unique().tolist()
                if unique_gids:
                    inferred_game_id = str(unique_gids[0])
        except Exception:
            inferred_game_id = None
    else:
        try:
            df = fit_xgs.analyze_game(gameID)
            inferred_game_id = str(gameID)
        except Exception as e:
            try:
                print(f"_game: failed to analyze game {gameID}: {e}")
            except Exception:
                pass
            df = pd.DataFrame([
                {'event': 'shot-on-goal', 'x': -60, 'y': 10, 'team_id': 1, 'home_id': 1, 'away_id': 2, 'home_team_defending_side': 'right'},
                {'event': 'shot-on-goal', 'x': -40, 'y': -5, 'team_id': 1, 'home_id': 1, 'away_id': 2, 'home_team_defending_side': 'right'},
                {'event': 'goal', 'x': 20, 'y': 5, 'team_id': 2, 'home_id': 1, 'away_id': 2, 'home_team_defending_side': 'right'},
            ])

    # Interpret `conditions` similarly to analyze._apply_condition:
    # - `conditions` is ONLY used to filter the game DataFrame (via parse.build_mask).
    # - Plotting-specific arguments (events, heatmap_split_mode, team_for_heatmap, etc.)
    #   must be supplied via `plot_kwargs`.

    # Filter conditions (like game_state, is_net_empty) are applied to the raw events df
    cond_work = conditions.copy() if isinstance(conditions, dict) else conditions
    team_val = None
    if isinstance(cond_work, dict) and 'team' in cond_work:
        team_val = cond_work.pop('team', None)

    # Normalize keys to match df columns where possible (tolerant mapping)
    if isinstance(cond_work, dict):
        def _norm(k: str) -> str:
            return ''.join(ch.lower() for ch in str(k) if ch.isalnum())
        col_map = { _norm(c): c for c in df.columns }
        corrected = {}
        for k, v in cond_work.items():
            nk = _norm(k)
            if nk in col_map:
                corrected[col_map[nk]] = v
            else:
                corrected[k] = v
        cond_work = corrected

    # If cond_work is a dict with filter keys, apply parse.build_mask to filter df
    df_filtered = df
    if isinstance(cond_work, dict) and cond_work:
        try:
            mask = _parse.build_mask(df, cond_work)
            if mask is not None:
                # align mask to df and apply
                mask = mask.reindex(df.index).fillna(False).astype(bool)
                df_filtered = df.loc[mask].copy()
        except Exception as e:
            try:
                print(f"_game: failed to apply conditions filter: {e}")
            except Exception:
                pass

    # Plot arguments must come from `plot_kwargs`. This keeps filtering separate
    # from display behavior. Provide sensible defaults when not supplied.
    if plot_kwargs is None:
        plot_kwargs = {}

    events_to_plot = plot_kwargs.get('events', plot_kwargs.get('events_to_plot', ['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot', 'xgs']))
    pe_kwargs = {
         'events_to_plot': events_to_plot,
         'event_styles': plot_kwargs.get('event_styles'),
         'out_path': plot_kwargs.get('out_path'),
         'figsize': plot_kwargs.get('figsize', (8, 4.5)),
         'title': plot_kwargs.get('title'),
         'return_heatmaps': plot_kwargs.get('return_heatmaps', False),
         'heatmap_split_mode': plot_kwargs.get('heatmap_split_mode', 'home_away'),
         'team_for_heatmap': plot_kwargs.get('team_for_heatmap'),
         # We'll populate 'summary_stats' below with timing/demo info when available.
         'summary_stats': plot_kwargs.get('summary_stats'),
     }

    # --- TIMING: call timing.compute_game_timing to get interval/timing info for this filtered game
    timing_info = None
    try:
        import timing as _timing
        # Always call compute_game_timing on the events-level dataframe we have
        # available (prefer the filtered dataframe). This avoids any season
        # CSV loading here and keeps _game lightweight and deterministic.
        df_for_timing = df_filtered if isinstance(df_filtered, pd.DataFrame) and not df_filtered.empty else (df if isinstance(df, pd.DataFrame) else None)
        demo_res = _timing.compute_game_timing(df_for_timing, condition=conditions, verbose=False)
        timing_info = demo_res
        # Try to extract per-game info for this game id
        per_game_info = None
        if isinstance(demo_res, dict):
            pg = demo_res.get('per_game', {})
            if pg:
                per_game_info = pg.get(gameID) or pg.get(str(gameID))
                try:
                    if per_game_info is None:
                        per_game_info = pg.get(int(gameID))
                except Exception:
                    pass

        # If demo_for_export didn't include the requested game, compute a
        # compact per-game timing summary locally so callers still receive
        # useful timing_info (this avoids silent empty results).
        if not per_game_info:
            try:
                # Build analysis_conditions same as demo_for_export
                if isinstance(conditions, dict):
                    raw_conditions = {k: v for k, v in conditions.items() if k != 'team'}
                    if not raw_conditions:
                        analysis_conditions = {'game_state': ['5v5'], 'is_net_empty': [0, 1]}
                    else:
                        analysis_conditions = {k: (list(v) if isinstance(v, (list, tuple, set)) else [v]) for k, v in raw_conditions.items()}
                else:
                    analysis_conditions = {'game_state': ['5v5'], 'is_net_empty': [0, 1]}

                # infer local team and opponent from df_filtered; if that is
                # empty fall back to the full game DataFrame `df` obtained earlier
                gdf = df_filtered.copy() if isinstance(df_filtered, pd.DataFrame) else None
                if gdf is None or gdf.empty:
                    # fallback to full game df
                    try:
                        gdf = df.copy() if isinstance(df, pd.DataFrame) else None
                    except Exception:
                        gdf = None
                if gdf is None or gdf.empty:
                    per_game_info = None
                else:
                    # derive home/away ids/abbs
                    home_id = gdf['home_id'].dropna().unique().tolist()[0] if 'home_id' in gdf.columns and not gdf['home_id'].dropna().empty else None
                    away_id = gdf['away_id'].dropna().unique().tolist()[0] if 'away_id' in gdf.columns and not gdf['away_id'].dropna().empty else None
                    home_abb = gdf['home_abb'].dropna().unique().tolist()[0] if 'home_abb' in gdf.columns and not gdf['home_abb'].dropna().empty else None
                    away_abb = gdf['away_abb'].dropna().unique().tolist()[0] if 'away_abb' in gdf.columns and not gdf['away_abb'].dropna().empty else None

                    # chosen local_team: use provided conditions team if present, else home_abb or home_id
                    team_param = conditions.get('team') if isinstance(conditions, dict) and 'team' in conditions else None
                    local_team = team_param if team_param is not None else (home_abb or home_id or away_abb or away_id)
                    # opponent key
                    if str(local_team) == str(home_abb) or (home_id is not None and str(local_team) == str(home_id)):
                        opp = away_abb or away_id
                    else:
                        opp = home_abb or home_id

                    per_game_info = {}
                    # Helper: compute merged intervals and pooled seconds per condition for a given side
                    def _compute_side_info(side_team_val):
                        side_df = _timing.add_game_state_relative_column(gdf.copy(), side_team_val)
                        merged_per_condition_local = {}
                        pooled_seconds_per_condition_local = {}
                        times = pd.to_numeric(side_df.get('total_time_elapsed_seconds', pd.Series(dtype=float)), errors='coerce').dropna()
                        total_observed = float(times.max() - times.min()) if len(times) >= 2 else 0.0
                        for cond_label, cond_def in analysis_conditions.items():
                            all_intervals = []
                            for state in cond_def:
                                if cond_label == 'game_state':
                                    cond_dict = {'game_state_relative_to_team': state}
                                else:
                                    cond_dict = {cond_label: state}
                                intervals, _, _ = _timing.intervals_for_condition(side_df, cond_dict, time_col='total_time_elapsed_seconds', verbose=False)
                                for it in intervals:
                                    try:
                                        all_intervals.append((float(it[0]), float(it[1])))
                                    except Exception:
                                        continue
                            # merge intervals
                            if all_intervals:
                                all_intervals = sorted(all_intervals, key=lambda x: x[0])
                                merged = []
                                cur_s, cur_e = all_intervals[0]
                                for s, e in all_intervals[1:]:
                                    if s <= cur_e:
                                        cur_e = max(cur_e, e)
                                    else:
                                        merged.append((cur_s, cur_e))
                                        cur_s, cur_e = s, e
                                merged.append((cur_s, cur_e))
                            else:
                                merged = []
                            pooled = sum((e - s) for s, e in merged) if merged else 0.0
                            merged_per_condition_local[str(cond_label)] = merged
                            pooled_seconds_per_condition_local[str(cond_label)] = pooled
                        return merged_per_condition_local, pooled_seconds_per_condition_local, total_observed

                    team_merged, team_pooled, total_obs = _compute_side_info(local_team)
                    opp_merged, opp_pooled, _ = _compute_side_info(opp)

                    # compute intersections across conditions for each side
                    def _intersect_lists(lists):
                        if not lists:
                            return []
                        inter = lists[0]
                        def _intersect_two(a,b):
                            res=[]
                            i=j=0
                            while i<len(a) and j<len(b):
                                s1,e1=a[i]; s2,e2=b[j]
                                start=max(s1,s2); end=min(e1,e2)
                                if end>start:
                                    res.append((start,end))
                                if e1<e2:
                                    i+=1
                                elif e2<e1:
                                    j+=1
                                else:
                                    i+=1; j+=1
                            return res
                        for lst in lists[1:]:
                            inter = _intersect_two(inter, lst)
                        return inter

                    team_inter = _intersect_lists(list(team_merged.values()))
                    opp_inter = _intersect_lists(list(opp_merged.values()))
                    team_inter_pooled = sum(e-s for s,e in team_inter) if team_inter else 0.0
                    opp_inter_pooled = sum(e-s for s,e in opp_inter) if opp_inter else 0.0

                    per_game_info = {
                        'selected_team': str(local_team),
                        'opponent_team': str(opp),
                        'sides': {
                            'team': {
                                'merged_intervals': team_merged,
                                'pooled_seconds': team_pooled,
                                'total_observed': total_obs,
                                'intersection_intervals': team_inter,
                                'pooled_intersection_seconds': team_inter_pooled,
                            },
                            'opponent': {
                                'merged_intervals': opp_merged,
                                'pooled_seconds': opp_pooled,
                                'total_observed': total_obs,
                                'intersection_intervals': opp_inter,
                                'pooled_intersection_seconds': opp_inter_pooled,
                            }
                        },
                        'game_total_observed_seconds': total_obs,
                    }

                # build small aggregate mirrors
                demo_res = {'per_game': {str(gameID): per_game_info}, 'aggregate': {
                    'pooled_seconds_per_condition': {'team': team_pooled, 'other': opp_pooled},
                    'intervals_per_condition': {'team': team_merged, 'other': opp_merged},
                    'intersection_pooled_seconds': {'team': team_inter_pooled, 'other': opp_inter_pooled},
                    'intersection_intervals': {'team': team_inter, 'other': opp_inter},
                }}
                timing_info = demo_res
                per_game_info = per_game_info
            except Exception:
                per_game_info = None
                timing_info = demo_res

        # Attach concise timing summary into plot summary_stats
        if per_game_info is not None:
            # include the per_game info under a clear key
            pe_kwargs['summary_stats'] = {'timing_per_game': per_game_info}
        else:
            # fallback: put the full demo result
            pe_kwargs['summary_stats'] = {'timing': demo_res}
    except Exception:
        # if timing call fails, leave summary_stats as provided (or None)
        timing_info = None

    # NOTE: we intentionally DO NOT derive plotting args from `conditions`.
    result = plot_events(df_filtered, **pe_kwargs)

    # Optionally return timing info along with the plot when requested
    if plot_kwargs.get('return_timing', False):
        # result may be (fig, ax) or (fig, ax, heatmaps)
        if isinstance(result, tuple) and len(result) == 2:
            return result + (timing_info,)
        elif isinstance(result, tuple) and len(result) == 3:
            # (fig, ax, heatmaps) -> append timing
            return (result[0], result[1], result[2], timing_info)
        else:
            return result, timing_info

    return result


# small demo helper (not run automatically) showing expected usage
def _example_usage():
    """Example usage of plot_events (for developer reference).

    df = pd.read_csv('static/some_game.csv')  # must contain x,y,event,team_id,home_id
    fig, ax = plot_events(df, events_to_plot=['SHOT','GOAL'], out_path='static/example_shots.png', title='Home vs Away Shots')
    """
    pass


if __name__ == '__main__':
    # Save an example plot for a sample game. If the live NHL feed is
    # unavailable, fall back to a tiny synthetic example so the script still
    # produces an image for inspection.
    import nhl_api
    import parse
    import fit_xgs

    game_id = '2025020196'
    out_file = f'static/{game_id}.png'

    try:
        # This is just a demo, so we don't need full functionality
        print("Running demo plot...")
        # In a real run, we would fetch data:
        # feed = nhl_api.get_game_feed(game_id)
        # events = parse._game(feed)
        # plot_events(events, out_path=out_file)
    except Exception as e:
        print('Error:', e)



def add_summary_text(ax, stats: dict, main_title: str, is_season_summary: bool, team_name: Optional[str] = None, full_team_name: Optional[str] = None, filter_str: Optional[str] = None):
    """
    Add summary text to the plot.
    
    stats: dictionary containing stats like 'home_goals', 'away_goals', 'home_xg', etc.
    """
    fig = ax.figure
    
    # Extract stats
    home_goals = stats.get('home_goals', 0)
    away_goals = stats.get('away_goals', 0)
    home_attempts = stats.get('home_attempts', 0)
    away_attempts = stats.get('away_attempts', 0)
    home_shot_pct = stats.get('home_shot_pct', 0.0)
    away_shot_pct = stats.get('away_shot_pct', 0.0)
    
    home_xg = stats.get('home_xg', 0.0)
    away_xg = stats.get('away_xg', 0.0)
    have_xg = stats.get('have_xg', False)
    
    # Title Logic
    if is_season_summary:
        # Use full team name if available, otherwise team_name (abbr)
        display_name = full_team_name if full_team_name else (team_name if team_name else "Team")
        final_title = display_name
    else:
        # If main_title is provided, use it. Otherwise default to "Home vs Away"
        final_title = main_title if main_title and main_title.strip() != "vs" else "Home vs Away"

    # Layout Configuration
    bbox = ax.get_position()
    axes_top = bbox.y1
    start_y = axes_top + 0.01
    gap = 0.04  # Slightly reduced gap
    

    
    # Column X-coordinates (relative to figure width, 0-1)
    cols_x = [0.32, 0.42, 0.5, 0.58, 0.68]
    cols_align = ['right', 'center', 'center', 'center', 'left']
    
    # Data Rows (Bottom to Top)
    rows = []
    
    # 1. Shots (SA)
    rows.append([
        f"{home_attempts}", 
        f"({home_shot_pct:.1f}%)", 
        "SA", 
        f"({away_shot_pct:.1f}%)", 
        f"{away_attempts}"
    ])
    
    # 2. xG/60
    # Hide xG/60 if game is ongoing
    game_ongoing = stats.get('game_ongoing', False)
    if not game_ongoing:
        t60 = float(stats.get('team_xg_per60', 0.0) or 0.0)
        o60 = float(stats.get('other_xg_per60', 0.0) or 0.0)
        rel_off = stats.get('rel_off_pct')
        rel_def = stats.get('rel_def_pct')
        
        xg60_row = [f"{t60:.3f}", "", "xG/60", "", f"{o60:.3f}"]
        if rel_off is not None:
            xg60_row[1] = f"({float(rel_off):+.1f}%)"
        if rel_def is not None:
            xg60_row[3] = f"({float(rel_def):+.1f}%)"
        rows.append(xg60_row)
    
    # 3. xG
    if have_xg and (home_xg > 0 or away_xg > 0):
        total_xg = home_xg + away_xg
        hx_pct = 100.0 * home_xg / total_xg if total_xg > 0 else 0.0
        ax_pct = 100.0 * away_xg / total_xg if total_xg > 0 else 0.0
        
        rows.append([
            f"{home_xg:.2f}",
            f"({hx_pct:.1f}%)",
            "xG",
            f"({ax_pct:.1f}%)",
            f"{away_xg:.2f}"
        ])
    else:
        rows.append(["-", "", "xG", "", "-"])
        
    # 4. Goals
    total_goals = home_goals + away_goals
    hg_pct = 100.0 * home_goals / total_goals if total_goals > 0 else 0.0
    ag_pct = 100.0 * away_goals / total_goals if total_goals > 0 else 0.0
    
    # Only show goal distribution for season summary
    g_dist_team = f"({hg_pct:.1f}%)" if (is_season_summary and total_goals > 0) else ""
    g_dist_other = f"({ag_pct:.1f}%)" if (is_season_summary and total_goals > 0) else ""
    
    rows.append([
        f"{home_goals}",
        g_dist_team,
        "Goals",
        g_dist_other,
        f"{away_goals}"
    ])
    
    # Plot Rows
    current_y = start_y
    font_size = 9
    font_weight = 'normal'
    
    for row in rows:
        for i, text in enumerate(row):
            if text:
                fig.text(cols_x[i], current_y, text, 
                         fontsize=font_size, fontweight=font_weight, 
                         ha=cols_align[i], color='black')
        current_y += gap
        
    # Header Section (Filter, Score, Title)
    # Start slightly above the table
    header_y = current_y + 0.005 # Small extra buffer
    
    # Filter Line
    if filter_str:
        fig.text(0.5, header_y, filter_str, fontsize=9, fontweight='bold', ha='center', color='black')
        header_y += gap
        
    # Score Line
    if is_season_summary:
        n_games = int(stats.get('n_games', 0))
        team_seconds = float(stats.get('team_seconds', 0.0))
        minutes = int(round(team_seconds / 60.0))
        if n_games > 0:
            score_line = f"{n_games} Games - {minutes} Minutes"
            fig.text(0.5, header_y, score_line, fontsize=10, fontweight='bold', ha='center', color='black')
            header_y += gap
    else:
        score_line = f"{home_goals} - {away_goals}"
        # Add TOI if available (useful for player/state filters)
        team_seconds = float(stats.get('team_seconds', 0.0))
        if team_seconds > 0:
            m = int(team_seconds // 60)
            s = int(team_seconds % 60)
            score_line += f" | TOI: {m:02d}:{s:02d}"
        
        fig.text(0.5, header_y, score_line, fontsize=10, fontweight='bold', ha='center', color='black')
        header_y += gap

    # Game Ongoing Line (if applicable, push title up further)
    # Check both game_ongoing flag AND time_remaining presence
    # Game Ongoing Line (if applicable, push title up further)
    # Check both game_ongoing flag AND time_remaining presence
    time_rem = stats.get('time_remaining')
    # Fix: Only show if game is actually ongoing. Previously showed if time_rem was present (e.g. '00:00 P3')
    if stats.get('game_ongoing'):
        display_time = time_rem if (time_rem and str(time_rem).lower() != 'none') else 'Ongoing'
        game_ongoing_line = f"Game Ongoing: {display_time}"
        fig.text(0.5, header_y, game_ongoing_line, fontsize=10, fontweight='bold', ha='center', color='red')
        header_y += gap

    # Title

    fig.text(0.5, header_y, final_title, fontsize=12, fontweight='bold', ha='center', color='black')
