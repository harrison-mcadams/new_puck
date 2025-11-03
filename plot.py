# Revised plotting routine
# The main input is an events DataFrame. Provide flexibility in what events
# are plotted and what plot symbols/colors are used for each event type.

from typing import List, Dict, Optional, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from rink import draw_rink, rink_goal_xs


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
    # Some feeds use 'event' or 'type'; prefer 'event'
    if 'event' not in events.columns:
        # try to be helpful: if there's a 'type' column use it
        if 'type' in events.columns:
            ev_col = 'type'
        else:
            raise KeyError("events DataFrame must contain an 'event' or 'type' column")
    else:
        ev_col = 'event'
    # Filter
    mask = events[ev_col].astype(str).str.strip().str.lower().isin(wanted)
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
        'shot': {'marker': 'o', 'size': 40, 'home_color': 'black', 'away_color': 'orange'},
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
    df = _events(events, events_to_plot)
    # compute adjusted coordinates so home events face left and away events face right
    df = adjust_xy_for_homeaway(df)

    # make sure we have numeric x,y
    if 'x' not in df.columns or 'y' not in df.columns:
        raise KeyError("events DataFrame must contain numeric 'x' and 'y' columns")

    # Prepare plotting axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if rink:
        try:
            draw_rink(ax=ax)
        except Exception:
            # If draw_rink fails for any reason, fall back to simple limits
            ax.set_xlim(-100, 100)
            ax.set_ylim(-50, 50)
            ax.set_aspect('equal', adjustable='box')
            ax.axis('off')

    # If no events to plot, return empty rink
    if df.shape[0] == 0:
        if title:
            ax.set_title(title)
        if out_path:
            fig.savefig(out_path, dpi=150)
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

    # Title and legend
    if title:
        ax.set_title(title)
    if handles:
        # show a compact legend; convert labels to Title case for readability
        ax.legend(handles, [lab.title() for lab in labels], loc='upper right', fontsize='small')

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)

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

    game_id = '2025020196'
    out_file = f'static/example_game_{game_id}.png'

    try:
        print(f'Attempting to fetch game feed for {game_id}...')
        feed = nhl_api.get_game_feed(game_id)
        events = parse._game(feed)
        events = parse._elaborate(events)
        df = pd.DataFrame.from_records(events)
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
    fig, ax = plot_events(df, events_to_plot=['SHOT', 'GOAL'], out_path=out_file, title=f'Game {game_id} — shots (home left)')
    print('Saved example plot to', out_file)
