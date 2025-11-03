# Revised plotting routine
# The main input is an events DataFrame. Provide flexibility in what events
# are plotted and what plot symbols/colors are used for each event type.

from typing import List, Dict, Optional, Tuple
import matplotlib
matplotlib.use('Agg')
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

def adjust_xy_for_homeaway(df):
    # orient all sides such as home shots are directed to the left,
    # away shots are directed to the right. also need to flip y accordingly

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

        # desired goal depending on whether shooter is home or away
        if t_id is not None and h_id is not None:
            if t_id == h_id:
                dg = left_goal_x
            elif t_id == a_id:
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

    Returns (fig, ax).
    """
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
    try:
        print(f"plot_events called: wants_xgs={wants_xgs}, requested={requested}, events_rows={len(events)}")
    except Exception:
        pass

    # Build a filtered list for _events that excludes 'xgs' (it's not an event name)
    if requested is not None:
        filtered_events = [e for e in events_to_plot if e.strip().lower() != 'xgs']
    else:
        filtered_events = None

    # filter events for normal scatter plotting (exclude 'xgs' token)
    df = _events(events, filtered_events)
    # compute adjusted coordinates so home events face left and away events face right
    df = adjust_xy_for_homeaway(df)

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

    # Attempt to determine home/away assignment per row
    # We'll treat a row as 'home' when team_id == home_id; otherwise 'away'
    is_home = None
    if all(c in df.columns for c in ('team_id', 'home_id')):
        is_home = df['team_id'] == df['home_id']
    else:
        # if missing, treat all as away to ensure something is plotted
        is_home = pd.Series([False] * len(df), index=df.index)

    # For legend handles
    handles = []
    labels = []

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
        grp_home = group[is_home.loc[group.index]]
        grp_away = group[~is_home.loc[group.index]]

        # choose adjusted coords if present
        xcol = 'x_a' if 'x_a' in group.columns else 'x'
        ycol = 'y_a' if 'y_a' in group.columns else 'y'

        if not grp_home.empty:
            h = ax.scatter(grp_home[xcol], grp_home[ycol], c=home_c, marker=m, s=s, edgecolors='none', zorder=5)
            if ev_type not in labels:
                handles.append(h)
                labels.append(ev_type)
        if not grp_away.empty:
            h2 = ax.scatter(grp_away[xcol], grp_away[ycol], c=away_c, marker=m, s=s, edgecolors='none', zorder=5)
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
    # compute home/away ids (prefer explicit columns from filtered df)
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



    home_name = df['home_abb'].iloc[0]
    away_name = df['away_abb'].iloc[0]

    # compute goals (count events with 'goal')
    ev_lc = df['event'].astype(str).str.strip().str.lower() if 'event' in df.columns else pd.Series([], dtype=object)
    is_goal = ev_lc == 'goal'
    home_goals = int(((df['team_id'].astype(str) == str(home_id)) & is_goal).sum()) if 'team_id' in df.columns and home_id is not None else int(is_goal.sum())
    away_goals = int(((df['team_id'].astype(str) != str(home_id)) & is_goal).sum()) if 'team_id' in df.columns and home_id is not None else 0

    # shot attempts totals (shot-on-goal, missed-shot, blocked-shot)
    # Use the normalized lowercase event column computed earlier (`ev_lc`).
    shot_attempt_types = {'shot-on-goal', 'missed-shot', 'blocked-shot'}
    ev_lc = df['event'].astype(str).str.strip().str.lower() if 'event' in df.columns else pd.Series([], dtype=object)
    attempt_mask = ev_lc.isin(shot_attempt_types)
    attempts_df = df.loc[attempt_mask]

    if 'team_id' in df.columns and home_id is not None:
        home_attempts = int((attempts_df['team_id'].astype(str) == str(home_id)).sum())
        away_attempts = int((attempts_df['team_id'].astype(str) != str(home_id)).sum())
    else:
        xs_series = attempts_df['x_a'] if 'x_a' in attempts_df.columns else attempts_df['x'] if 'x' in attempts_df.columns else pd.Series([])
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

    # Format text lines: main title, score, xG line, shots line
    main_title = f"{home_name} vs {away_name}"
    score_line = f"{home_goals} - {away_goals}"
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
    score_y = xg_y + gap
    main_y = score_y + gap

    # clamp to figure
    shots_y = max(0.0, min(0.995, shots_y))
    xg_y = max(0.0, min(0.995, xg_y))
    score_y = max(0.0, min(0.995, score_y))
    main_y = max(0.0, min(0.995, main_y))

    # Slightly reduce main/score fonts to help avoid overlap at small figure sizes
    fig.text(0.5, main_y, main_title, fontsize=11, fontweight='bold', ha='center')
    fig.text(0.5, score_y, score_line, fontsize=10, fontweight='bold', ha='center')
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
            # ensure adjusted coords exist
            events_with_xg = adjust_xy_for_homeaway(events_with_xg)
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
                # build two heat arrays: home and away, so we can color them
                heat_home = np.zeros_like(XX, dtype=float)
                heat_away = np.zeros_like(XX, dtype=float)
                # determine which events are home/away using home_id if available
                # try to get home_id from events_with_xg, else try df
                home_id_val = None
                if 'home_id' in events_with_xg.columns:
                    vals = events_with_xg['home_id'].dropna().unique()
                    if len(vals) > 0:
                        home_id_val = vals[0]
                elif 'home_id' in df.columns:
                    vals = df['home_id'].dropna().unique()
                    if len(vals) > 0:
                        home_id_val = vals[0]

                two_sigma2 = 2.0 * (hm_sigma ** 2)
                # iterate over events and add to appropriate heat map
                for xi, yi, ai, row in zip(xs, ys, amps, events_with_xg.loc[mask].itertuples(index=False)):
                    dx = XX - xi
                    dy = YY - yi
                    kern = ai * np.exp(-(dx * dx + dy * dy) / two_sigma2)
                    try:
                        # row may have team_id/home_id attributes; compare if present
                        team_id = getattr(row, 'team_id', None)
                        h_id = getattr(row, 'home_id', None)
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
                        heat_home += kern

                # mask out points outside the rink boundary
                rink_mask = np.vectorize(rink_half_height_at_x)(XX) >= np.abs(YY)
                heat_home *= rink_mask
                heat_away *= rink_mask

                # create RGBA images where rgb is team color and alpha is normalized heat
                extent = (gx[0] - hm_res / 2.0, gx[-1] + hm_res / 2.0, gy[0] - hm_res / 2.0, gy[-1] + hm_res / 2.0)

                # pick colors for home/away (allow overrides)
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

    return fig, ax


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
    out_file = f'static/example_game_{game_id}.png'

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
    fig, ax = plot_events(df, events_to_plot=['shot-on-goal', 'goal',
                                              'blocked-shot', 'missed-shot',
                                              'xGs'],
                          out_path=out_file, title=f'Game {game_id} — shots (home left)')
    print('Saved example plot to', out_file)
