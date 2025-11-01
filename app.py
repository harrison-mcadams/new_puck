from flask import Flask, render_template, url_for
import os

app = Flask(__name__, static_folder="static", template_folder="templates")


@app.route("/")
def index():
    img_path = "shot_plot.png"
    full = os.path.join(app.static_folder, img_path)
    exists = os.path.exists(full)
    return render_template("index.html", img=img_path, exists=exists)


if __name__ == "__main__":
    app.run(debug=True)
import requests
from datetime import date, timedelta

TEAM_ID = 4  # Philadelphia Flyers


def get_most_recent_flyers_game_pk(days_back=30):
    """Return the gamePk for the most recent Flyers game within `days_back`.

    Falls back to a larger window if none found.
    """
    end = date.today()
    start = end - timedelta(days=days_back)
    url = (
        "https://statsapi.web.nhl.com/api/v1/schedule"
        f"?teamId={TEAM_ID}&startDate={start.isoformat()}&endDate={end.isoformat()}"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    dates = data.get("dates", [])
    if not dates and days_back < 365:
        return get_most_recent_flyers_game_pk(days_back=365)
    if not dates:
        raise RuntimeError("No recent Flyers games found via NHL API.")
    # take last date's last game (most recent)
    last_date = dates[-1]
    games = last_date.get("games", [])
    if not games:
        raise RuntimeError("No games found in schedule response.")
    return games[-1]["gamePk"]


def get_game_feed(game_pk):
    url = f"https://statsapi.web.nhl.com/api/v1/game/{game_pk}/feed/live"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()

