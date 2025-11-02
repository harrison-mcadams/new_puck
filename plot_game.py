"""Shot-plot generation utilities.

This module exposes `plot_shots(gameID, output_file, ...)` which fetches a game's
play-by-play feed (via `nhl_api.get_game_feed`), parses SHOT/GOAL events using
`parse_events.parse_shot_and_goal_events`, and renders shot locations onto the
schematic rink provided by `rink.draw_rink`.

Coordinate / convention notes:
- The NHL rink coordinate system used by many feeds places the center at (0,0),
  x-axis running from left (negative) to right (positive), and y-axis across
  the rink width. The plotting function enforces the following visual rule:
    - Home team attempts are displayed toward the left goal (negative x).
    - Away team attempts are displayed toward the right goal (positive x).
  To guarantee this visual convention the code flips shot x-values based on the
  detected `home_id` from the feed.

Colors / markers used:
- Home shots: black circles
- Away shots: orange circles
- Goals: plotted as 'x' markers (black for home, orange for away)
"""

import os
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nhl_api import get_game_ID, get_game_feed
from parse import game
from rink import draw_rink

OUTPUT_DIR = "static"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "shot_plot.png")


def _extract_home_away_from_feed(feed):
    """Try to extract home and away team IDs from the feed using common paths."""
    # try common shapes
    home_id = None
    away_id = None
    # api-web play-by-play often includes top-level 'game' or 'gameData' structures
    gd = feed.get('gameData') or feed.get('game') or feed.get('gameInfo') or feed.get('gameData', {})
    if isinstance(gd, dict):
        teams = gd.get('teams') or gd.get('team') or {}
        if isinstance(teams, dict):
            home = teams.get('home') or teams.get('Home') or teams.get('HOME')
            away = teams.get('away') or teams.get('Away') or teams.get('AWAY')
            if isinstance(home, dict):
                home_id = home.get('id') or home.get('teamId')
            if isinstance(away, dict):
                away_id = away.get('id') or away.get('teamId')
    # fallback: some feeds include 'home'/'away' at top-level
    if not home_id:
        home = feed.get('home') or feed.get('homeTeam') or {}
        if isinstance(home, dict):
            home_id = home.get('id') or home.get('teamId')
    if not away_id:
        away = feed.get('away') or feed.get('awayTeam') or {}
        if isinstance(away, dict):
            away_id = away.get('id') or away.get('teamId')
    # final fallback: try feed['teams']
    teams = feed.get('teams') or {}
    if not home_id and isinstance(teams, dict) and 'home' in teams:
        try:
            home_id = teams['home'].get('id')
        except Exception:
            pass
    if not away_id and isinstance(teams, dict) and 'away' in teams:
        try:
            away_id = teams['away'].get('id')
        except Exception:
            pass
    return home_id, away_id


def plot_shots(gameID: int, output_file: str = OUTPUT_FILE, mirror: bool = True,
               show_goals: bool = True, rotate180: bool = False) -> str:
    """Create and save a shot-location plot for the given gameID.

    Behavior:
      - Parse SHOT/GOAL events via parse_shot_and_goal_events(feed).
      - If home team id is found in the feed, force home events to x = -abs(x) and
        away events to x = +abs(x) so home attempts appear toward the left goal.
      - Then apply optional rotate180 (visual rotation by reversing axes limits)
        and optional horizontal mirror (negate x coordinates).

    Returns the path to the saved image.
    """
    os.makedirs(os.path.dirname(output_file) or OUTPUT_DIR, exist_ok=True)
    print(f"Fetching game feed for gameID={gameID}...")
    feed = get_game_feed(gameID)

    events = parse.game(feed)
    if not events:
        print("parse_shot_and_goal_events: no events parsed; nothing to plot")

    home_id, away_id = _extract_home_away_from_feed(feed)
    # If feed doesn't include home_id, choose the most frequent teamID among events as a best-effort home
    if home_id is None:
        team_counts = {}
        for e in events:
            t = e.get('teamID')
            if t is None:
                continue
            team_counts[str(t)] = team_counts.get(str(t), 0) + 1
        if team_counts:
            # pick the teamID with maximum events
            most_common_team = max(team_counts.items(), key=lambda kv: kv[1])[0]
            home_id = most_common_team
            print(f"No explicit home_id found in feed; using most-common team {home_id} as home")
    print(f"Detected home_id={home_id}, away_id={away_id}")

    # prepare plotted points
    xs = []
    ys = []
    event_types = []
    teams_list = []

    # initialize per-class lists so they exist even if no events are plotted
    home_shot_xs = []
    home_shot_ys = []
    away_shot_xs = []
    away_shot_ys = []
    home_goal_xs = []
    home_goal_ys = []
    away_goal_xs = []
    away_goal_ys = []

    for e in events:
        raw_x = e.get('x')
        raw_y = e.get('y')
        team = e.get('teamID')
        ev_type = e.get('event')

        # default coordinates
        x = raw_x
        y = raw_y

        # If we know home_id, force home shots to negative x and away to positive x.
        if home_id is not None:
            try:
                home_match = str(team) == str(home_id)
            except Exception:
                home_match = False
            if home_match:
                x = -abs(raw_x)
            else:
                x = abs(raw_x)

        # debug: print mapping for first few events
        if len(xs) < 5:
            print(f"event team={team} raw_x={raw_x} -> plotted_x={x}")

        xs.append(x)
        ys.append(y)
        teams_list.append(team)
        event_types.append(ev_type.upper() if ev_type else '')

    # draw
    fig, ax = plt.subplots(figsize=(8, 4.2))
    draw_rink(ax, mirror=False, show_goals=show_goals)  # draw_rink mirror handled via x flipping above

    if xs:
        # Separate points into shots vs goals and home vs away so we can color/mark them.
        home_shot_xs = []
        home_shot_ys = []
        away_shot_xs = []
        away_shot_ys = []
        home_goal_xs = []
        home_goal_ys = []
        away_goal_xs = []
        away_goal_ys = []

        for i, x in enumerate(xs):
            y = ys[i]
            team = teams_list[i]
            et = event_types[i]
            is_home = (home_id is not None and str(team) == str(home_id))
            # classify
            if et and et.upper().startswith('GOAL'):
                if is_home:
                    home_goal_xs.append(x); home_goal_ys.append(y)
                else:
                    away_goal_xs.append(x); away_goal_ys.append(y)
            else:
                # treat as shot
                if is_home:
                    home_shot_xs.append(x); home_shot_ys.append(y)
                else:
                    away_shot_xs.append(x); away_shot_ys.append(y)

        # plot: home shots black, away shots orange; goals as 'x' markers
        if home_shot_xs:
            ax.scatter(home_shot_xs, home_shot_ys, c='black', s=40, marker='o', edgecolor='k', alpha=0.9)
        if away_shot_xs:
            ax.scatter(away_shot_xs, away_shot_ys, c='orange', s=40, marker='o', edgecolor='k', alpha=0.9)
        if home_goal_xs:
            ax.scatter(home_goal_xs, home_goal_ys, c='black', s=100, marker='x', linewidths=2)
        if away_goal_xs:
            ax.scatter(away_goal_xs, away_goal_ys, c='orange', s=100, marker='x', linewidths=2)
    else:
        print('No coordinates to plot after parsing.')

    # --- compute goal tallies for titles ---
    # ensure variables exist even if xs was empty
    home_goals = len(home_goal_xs) if 'home_goal_xs' in locals() else 0
    away_goals = len(away_goal_xs) if 'away_goal_xs' in locals() else 0

    goals_line = f"{home_goals}  -  {away_goals}"

    # --- Title and subtitle (home vs away and shot totals/percentages) ---
    # Determine display names for home/away
    def _team_display_name(side: str):
        """Robustly determine a display name for a team side ('home' or 'away').

        Tries several common feed locations and falls back to event data.
        """
        # 1) gameData -> teams -> side
        try:
            gd = feed.get('gameData', {}) if isinstance(feed, dict) else {}
            teams = gd.get('teams', {}) if isinstance(gd, dict) else {}
            t = teams.get(side) if isinstance(teams, dict) else None
            if isinstance(t, dict):
                return t.get('triCode') or t.get('abbrev') or t.get('name') or str(t.get('id'))
        except Exception:
            pass

        # 2) top-level keys like 'homeTeam' or 'awayTeam'
        try:
            st = feed.get(f"{side}Team") if isinstance(feed, dict) else None
            if isinstance(st, dict):
                return st.get('abbrev') or st.get('triCode') or st.get('name') or str(st.get('id'))
        except Exception:
            pass

        # 3) try generic 'teams' or other structures
        try:
            teams = feed.get('teams') if isinstance(feed, dict) else None
            if isinstance(teams, dict):
                t = teams.get(side)
                if isinstance(t, dict):
                    return t.get('triCode') or t.get('abbrev') or t.get('name') or str(t.get('id'))
        except Exception:
            pass

        # 4) fallback: use first matching teamAbbrev from parsed events
        for ev in events:
            ta = ev.get('teamAbbrev')
            if ta:
                return ta
            tid = ev.get('teamID')
            if tid is not None:
                return str(tid)

        # ultimate fallback
        return side.upper()

    home_name = _team_display_name('home')
    away_name = _team_display_name('away')

    # shot totals
    total_shots = len(xs)
    home_shots = 0
    away_shots = 0
    if home_id is not None:
        for i, t in enumerate(teams_list):
            if str(t) == str(home_id):
                home_shots += 1
            else:
                away_shots += 1
    else:
        # if no home_id available, attempt to count by comparing plotted x signs
        for x in xs:
            if x < 0:
                home_shots += 1
            else:
                away_shots += 1

    # percentages
    if (home_shots + away_shots) > 0:
        home_pct = 100.0 * home_shots / (home_shots + away_shots)
        away_pct = 100.0 * away_shots / (home_shots + away_shots)
    else:
        home_pct = away_pct = 0.0

    main_title = f"{home_name}  vs  {away_name}"
    subtitle = (f"{home_shots} ({home_pct:.1f}%)  -  {away_shots} "
                f"({away_pct:.1f}%)")

    # Small nudge down so the title block sits a smidge closer to the rink.
    # Main title stays bold; subtitle is unbolded (normal) and slightly dimmer.
    fig.suptitle(main_title, fontsize=13, fontweight='bold', ha='center',
                 y=0.877)
    # goals subtitle (bold) placed between main title and shot stats
    fig.text(0.5, 0.809, goals_line, ha='center', fontsize=11,
             fontweight='bold')
    # subtitle unbolded and slightly dim (placed below the goals line)
    fig.text(0.5, 0.78, subtitle, ha='center', fontsize=9,
             fontweight='normal', color='black', alpha=0.95)
    # Adjust subplot top to give the title block room but keep it close to the plot
    fig.subplots_adjust(top=0.80)


    # debug: report axis limits after rotation
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    print(f"Axis limits after rotation (if any): xlim={xlim}, ylim={ylim}")

    # Final enforcement: ensure home team is displayed on the left. If after transforms
    # the mean x for home events is positive (i.e., home appears on the right), flip x-axis.
    if home_id is not None and xs and teams_list:
        # compute mean x for events that belong to home
        home_xs = [xs[i] for i, t in enumerate(teams_list) if str(t) == str(home_id)]
        if home_xs:
            mean_home = sum(home_xs) / len(home_xs)
            # rotation reverses left/right visually, so multiplier = -1 if rotated
            display_mean = mean_home * (-1 if rotate180 else 1)
            if display_mean > 0:
                x0, x1 = ax.get_xlim()
                ax.set_xlim(x1, x0)
                print(f"Home mean after transforms was {display_mean}; flipped x-axis to make home left.")
            else:
                print(f"Home mean after transforms is {display_mean}; home is already left.")

    # Save
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to {output_file}")
    return output_file


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot shots for a given NHL game ID (Flyers by default)")
    parser.add_argument("--game", "-g", type=int, default=None, help="gameID to plot (integer). If omitted, the most recent Flyers game is used.")
    parser.add_argument("--out", "-o", type=str, default=OUTPUT_FILE, help="output image path")
    parser.add_argument("--mirror", action="store_true", help="mirror the rink horizontally")
    parser.add_argument("--no-goals", dest="show_goals", action="store_false", help="hide schematic goal creases/posts")
    parser.add_argument("--rotate180", action="store_true", help="rotate plot 180 degrees")
    args = parser.parse_args(argv)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if args.game is None:
        print("Finding most recent Flyers game...")
        gameID = get_gameID(method='most_recent')
    else:
        gameID = args.game

    plot_shots(gameID, output_file=args.out, mirror=args.mirror, show_goals=args.show_goals, rotate180=args.rotate180)


if __name__ == "__main__":
    main()
