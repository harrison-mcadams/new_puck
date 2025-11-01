import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
#from nhl_api import get_most_recent_flyers_game_pk, get_game_feed
from nhl_api import get_gameID, get_game_feed
from parse_events import parse_shot_and_goal_events
from rink import draw_rink

OUTPUT_DIR = "static"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "shot_plot.png")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Fetching most recent Flyers game...")
    gameID = get_gameID(method='most_recent')
    feed = get_game_feed(gameID)

    events = parse_shot_and_goal_events(feed)
    if not events:
        print("No shot/goal events with coordinates found in the game feed.")
        return
    shots_x = [e["x"] for e in events]
    shots_y = [e["y"] for e in events]
    colors = ["red" if e["event"] == "GOAL" else "blue" for e in events]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    draw_rink(ax)
    ax.scatter(shots_x, shots_y, c=colors, s=40, edgecolor="k", alpha=0.8)
    ax.set_title("Played shots/goals (most recent Flyers game)")
    fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

