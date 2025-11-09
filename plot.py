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

    # Basic validation
    if not isinstance(events, pd.DataFrame):
        raise TypeError('events must be a pandas DataFrame')

    # Default event styles
    default_styles = {
        'shot-on-goal': {'marker': 'o', 'size': 40, 'home_color': 'black',
                  'away_color': 'orange'},
        'missed-shot': {'marker': 'o', 'size': 40, 'home_color': 'black',
                         'away_color': 'orange'},
        'blocked-shot': {'marker': 'o', 'size': 40, 'home_color': 'black',
                         'away_color': 'orange'},
        'goal': {'marker': 'x', 'size': 80, 'home_color': 'black', 'away_color': 'orange'},
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
    if heatmap_split_mode == 'team_not_team' and team_for_heatmap is not None:
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
            h = ax.scatter(grp1[xcol], grp1[ycol], c=c1, marker=m, s=s, edgecolors='none', zorder=5)
            if ev_type not in labels:
                handles.append(h)
                labels.append(ev_type)
        if not grp2.empty:
            h2 = ax.scatter(grp2[xcol], grp2[ycol], c=c2, marker=m, s=s, edgecolors='none', zorder=5)
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

    # xG totals if available (column 'xgs')
    home_xg = 0.0
    away_xg = 0.0
    have_xg = False
    if 'xgs' in df.columns:
        try:
            have_xg = True
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

    # Format text lines: main title, score, xG per60 line (from summary_stats),
    # full xG line, shots line
    main_title = f"{home_name} vs {away_name}"
    score_line = f"{home_goals} - {away_goals}"
    # xG per 60 from summary_stats (if provided)
    xg_per60_line = None
    if isinstance(summary_stats, dict):
        try:
            t60 = float(summary_stats.get('team_xg_per60', 0.0) or 0.0)
            o60 = float(summary_stats.get('other_xg_per60', 0.0) or 0.0)
            xg_per60_line = f"{t60:.3f} xG/60 - xG/60 - {o60:.3f} xG/60"
        except Exception:
            xg_per60_line = None

    if have_xg and (home_xg or away_xg):
        total_xg = home_xg + away_xg
        if total_xg > 0:
            hx_pct = 100.0 * home_xg / total_xg
            ax_pct = 100.0 * away_xg / total_xg
        else:
            hx_pct = ax_pct = 0.0
        xg_line = f"{home_xg:.2f} ({hx_pct:.1f}%) - xG - {away_xg:.2f} ({ax_pct:.1f}%)"
    else:
        xg_line = "xG: N/A"
    shots_line = f"{home_attempts} ({home_shot_pct:.1f}%) - SA - {away_attempts} ({away_shot_pct:.1f}%)"

    # Derive positions from the axes bounding box so the shots line sits just
    # above the rink. Use a larger inter-line gap to avoid overlap.
    bbox = ax.get_position()
    axes_top = bbox.y1

    # place shots line slightly above axes top
    shots_y = axes_top + 0.006
    # inter-line gap (increase by ~5% relative to earlier small gaps)
    gap = 0.035
    xg_y = shots_y + gap
    # If we have an xg_per60_line, allocate another gap for it
    if xg_per60_line is not None:
         xg_per60_y = xg_y + gap
         score_y = xg_per60_y + gap
    else:
         xg_per60_y = None
         score_y = xg_y + gap
    main_y = score_y + gap

    # clamp to figure
    shots_y = max(0.0, min(0.995, shots_y))
    xg_y = max(0.0, min(0.995, xg_y))
    if xg_per60_y is not None:
         xg_per60_y = max(0.0, min(0.995, xg_per60_y))
    score_y = max(0.0, min(0.995, score_y))
    main_y = max(0.0, min(0.995, main_y))

    # Slightly reduce main/score fonts to help avoid overlap at small figure sizes
    fig.text(0.5, main_y, main_title, fontsize=11, fontweight='bold', ha='center')
    fig.text(0.5, score_y, score_line, fontsize=10, fontweight='bold', ha='center')
    # render the xG per60 line (if available) between score and xG line
    if xg_per60_line is not None:
         fig.text(0.5, xg_per60_y, xg_per60_line, fontsize=9, fontweight='bold', ha='center')
    fig.text(0.5, xg_y, xg_line, fontsize=9, fontweight='normal', ha='center')
    fig.text(0.5, shots_y, shots_line, fontsize=9, fontweight='normal', ha='center')

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

        if not events_with_xg.empty:
            # ensure adjusted coords exist; if either 'x_a' or 'y_a' is missing
            # (for example they were not created earlier), compute them now.
            if ('x_a' not in events_with_xg.columns) or ('y_a' not in events_with_xg.columns):
                # Respect the caller's split mode when computing adjusted coords
                events_with_xg = adjust_xy_for_homeaway(events_with_xg, split_mode=heatmap_split_mode, team_for_heatmap=team_for_heatmap)
            try:
                print('events_with_xg rows after filter:', events_with_xg.shape[0])
            except Exception:
                pass
            xcol = 'x_a' if 'x_a' in events_with_xg.columns else 'x'
            ycol = 'y_a' if 'y_a' in events_with_xg.columns else 'y'
            xs = pd.to_numeric(events_with_xg[xcol], errors='coerce')
            ys = pd.to_numeric(events_with_xg[ycol], errors='coerce')
            amps = pd.to_numeric(events_with_xg['xgs'], errors='coerce')
            mask = (~xs.isna()) & (~ys.isna()) & (~amps.isna())
            xs = xs[mask].values
            ys = ys[mask].values
            amps = amps[mask].values

            if len(xs) > 0:
                # get axes bounds and build grid
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                # build grid with hm_res
                gx = np.arange(xmin, xmax + hm_res, hm_res)
                gy = np.arange(ymin, ymax + hm_res, hm_res)
                try:
                    print(f'grid sizes gx={len(gx)}, gy={len(gy)}, xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}')
                except Exception:
                    pass
                XX, YY = np.meshgrid(gx, gy)
                # build heat arrays depending on split mode
                if heatmap_split_mode == 'team_not_team':
                    heat_team = np.zeros_like(XX, dtype=float)
                    heat_not_team = np.zeros_like(XX, dtype=float)
                else:
                    # default legacy mode: home vs away
                    heat_home = np.zeros_like(XX, dtype=float)
                    heat_away = np.zeros_like(XX, dtype=float)

                # precompute team_for_heatmap parsing
                try:
                    tid = int(team_for_heatmap) if team_for_heatmap is not None and str(team_for_heatmap).strip().isdigit() else None
                except Exception:
                    tid = None
                tupper = str(team_for_heatmap).upper() if team_for_heatmap is not None and not isinstance(team_for_heatmap, (int, float)) else None

                # canonical home id preference for legacy home/away logic
                home_id_val = None
                if 'home_id' in events_with_xg.columns:
                    vals = events_with_xg['home_id'].dropna().unique()
                    if len(vals) > 0:
                        home_id_val = vals[0]
                elif 'home_id' in events.columns:
                    vals = events['home_id'].dropna().unique()
                    if len(vals) > 0:
                        home_id_val = vals[0]

                # Gaussian kernel precomputations
                two_sigma2 = 2.0 * (hm_sigma ** 2)
                # normalization factor so discrete grid sums approximate integral = ai
                norm_factor = (hm_res ** 2) / (2.0 * np.pi * (hm_sigma ** 2))

                for xi, yi, ai, row in zip(xs, ys, amps, events_with_xg.loc[mask].itertuples(index=False)):
                    dx = XX - xi
                    dy = YY - yi
                    kern = ai * norm_factor * np.exp(-(dx * dx + dy * dy) / two_sigma2)
                    try:
                        # row may have team_id/home_id attributes; decide bucket based on mode
                        team_id = getattr(row, 'team_id', None)
                        h_id = getattr(row, 'home_id', None)

                        if heatmap_split_mode == 'team_not_team' and team_for_heatmap is not None:
                            # determine whether this event belongs to the requested team
                            is_team_event = False
                            try:
                                if tid is not None and team_id is not None:
                                    is_team_event = str(team_id) == str(tid)
                                elif tupper is not None:
                                    # check whether home/away abbrevs match and team_id aligns
                                    home_abb = getattr(row, 'home_abb', None)
                                    away_abb = getattr(row, 'away_abb', None)
                                    if home_abb is not None and str(home_abb).upper() == tupper:
                                        is_team_event = (team_id is not None and str(team_id) == str(getattr(row, 'home_id', None)))
                                    elif away_abb is not None and str(away_abb).upper() == tupper:
                                        is_team_event = (team_id is not None and str(team_id) == str(getattr(row, 'away_id', None)))
                            except Exception:
                                is_team_event = False

                            if is_team_event:
                                heat_team += kern
                            else:
                                heat_not_team += kern
                        else:
                            # legacy home/away behavior
                            is_home_event = None
                            if team_id is not None and h_id is not None:
                                is_home_event = str(team_id) == str(h_id)
                            elif home_id_val is not None and team_id is not None:
                                is_home_event = str(team_id) == str(home_id_val)
                            else:
                                # fallback: use x-coordinate sign (adjusted coords)
                                is_home_event = (xi < 0)
                            if is_home_event:
                                heat_home += kern
                            else:
                                heat_away += kern
                    except Exception:
                        # if anything goes wrong, accumulate into combined heat
                        if heatmap_split_mode == 'team_not_team':
                            heat_team += kern
                        else:
                            heat_home += kern

                # mask out points outside the rink boundary
                rink_mask = np.vectorize(rink_half_height_at_x)(XX) >= np.abs(YY)
                if heatmap_split_mode == 'team_not_team':
                    heat_team *= rink_mask
                    heat_not_team *= rink_mask
                else:
                    heat_home *= rink_mask
                    heat_away *= rink_mask

                # create RGBA images where rgb is team color and alpha is normalized heat
                extent = (gx[0] - hm_res / 2.0, gx[-1] + hm_res / 2.0, gy[0] - hm_res / 2.0, gy[-1] + hm_res / 2.0)

                # pick colors depending on split mode (allow overrides)
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

                # helper to convert heat -> rgba image
                def heat_to_rgba(h, color, alpha_scale=hm_alpha):
                    maxv = float(np.nanmax(h)) if np.nanmax(h) > 0 else 0.0
                    if maxv <= 0:
                        return None
                    norm = np.clip(h / maxv, 0.0, 1.0)
                    rgba = np.zeros((h.shape[0], h.shape[1], 4), dtype=float)
                    r, g, b, _ = mcolors.to_rgba(color)
                    rgba[..., 0] = r
                    rgba[..., 1] = g
                    rgba[..., 2] = b
                    # alpha channel is opacity gradient: normalized heat * alpha_scale
                    rgba[..., 3] = norm * alpha_scale
                    return rgba

                if heatmap_split_mode == 'team_not_team':
                    rgba_team = heat_to_rgba(heat_team, team_color) if heat_team is not None else None
                    rgba_not_team = heat_to_rgba(heat_not_team, not_team_color) if heat_not_team is not None else None
                    # overlay team then not_team (not_team on top)
                    if rgba_team is not None:
                        ax.imshow(rgba_team, extent=extent, origin='lower', zorder=1)
                    if rgba_not_team is not None:
                        ax.imshow(rgba_not_team, extent=extent, origin='lower', zorder=2)
                else:
                    rgba_home = heat_to_rgba(heat_home, home_color) if heat_home is not None else None
                    rgba_away = heat_to_rgba(heat_away, away_color) if heat_away is not None else None
                    # overlay home then away (so away appears on top if overlapping)
                    if rgba_home is not None:
                        ax.imshow(rgba_home, extent=extent, origin='lower', zorder=1)
                    if rgba_away is not None:
                        ax.imshow(rgba_away, extent=extent, origin='lower', zorder=2)

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
    # - else -> (fig, ax)
    if return_heatmaps:
        if heatmap_split_mode == 'team_not_team':
            return fig, ax, {'team': heat_team, 'not_team': heat_not_team}
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

    # Obtain the game DataFrame (analyze_game may raise; handle gracefully)
    try:
        df = fit_xgs.analyze_game(gameID)
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

    # --- TIMING: call timing.demo_for_export to get interval/timing info for this filtered game
    timing_info = None
    try:
        import timing as _timing
        # Prefer to call demo_for_export with a season-level dataframe so the
        # function can locate all games and compute aggregates correctly. Try
        # to infer the season from available data; if that fails, fall back
        # to calling demo_for_export on the single-game df.
        season_df = None
        # 1) if df_filtered contains a 'season' or 'season_id' column, try that
        try:
            if isinstance(df_filtered, pd.DataFrame) and 'season' in df_filtered.columns:
                val = df_filtered['season'].dropna().unique()
                if len(val) > 0:
                    season_str = str(val[0])
                    season_df = _timing.load_season_df(season_str)
        except Exception:
            season_df = None

        # 2) infer from gameID (common format: YYYY...); build season like '20252026'
        if season_df is None:
            try:
                gid = str(gameID)
                if len(gid) >= 4 and gid[:4].isdigit():
                    start = int(gid[:4])
                    season_guess = f"{start}{start+1}"
                    season_df = _timing.load_season_df(season_guess)
            except Exception:
                season_df = None

        # 3) fallback: try the default season loader without guess (lets it search)
        if season_df is None:
            try:
                season_df = _timing.load_season_df()
            except Exception:
                season_df = None

        # finally invoke demo_for_export using season_df when available
        if season_df is not None and not season_df.empty:
            demo_res = _timing.demo_for_export(season_df, condition=conditions, verbose=False)
        else:
            demo_res = _timing.demo_for_export(df_filtered, condition=conditions, verbose=False)
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

        df = fit_xgs.analyze_game(game_id)
        # concatenate xG results back to game DataFrame for further analysis if desired



        print(f'Parsed {len(df)} events from game {game_id}')
    except Exception as e:
        print('Failed to fetch/parse game feed — using synthetic demo data. Error:', e)
        # minimal synthetic dataset with home_id=1, away_id=2
        df = pd.DataFrame([
            {'event': 'SHOT', 'x': -60, 'y': 10, 'team_id': 1, 'home_id': 1, 'away_id': 2, 'home_team_defending_side': 'right'},
            {'event': 'SHOT', 'x': -40, 'y': -5, 'team_id': 1, 'home_id': 1, 'away_id': 2, 'home_team_defending_side': 'right'},
            {'event': 'GOAL', 'x': 20, 'y': 5, 'team_id': 2, 'home_id': 1, 'away_id': 2, 'home_team_defending_side': 'right'},
            {'event': 'SHOT', 'x': 30, 'y': -15, 'team_id': 2, 'home_id': 1, 'away_id': 2, 'home_team_defending_side': 'right'},
        ])

    # Ensure static dir exists
    Path('static').mkdir(parents=True, exist_ok=True)

    print('Generating plot to', out_file)
    plot_events(df, events_to_plot=['shot-on-goal', 'goal',
                                              'blocked-shot', 'missed-shot',
                                              'xGs'], out_path=out_file)
    print('Saved example plot to', out_file)
