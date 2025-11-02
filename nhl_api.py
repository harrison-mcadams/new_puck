"""Small NHL API helpers used by the demo.

This module provides a lightweight helper to locate a recent game ID for a
team (currently defaulting to PHI) from the api-web schedule endpoint, and a
simple helper to fetch the play-by-play feed for a given game ID.

The helpers are intentionally small and defensive: they try a couple of common
timestamp formats and return the most-recent past game (or the next future
one if no past games are available).
"""

import requests
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List


def get_season(team: str = 'PHI', season: str = '20252026') -> List[Dict[str, Any]]:
    """Return the season's games list for the given team.

    Parameters
    - team: optional team abbreviation (e.g. 'PHI', 'NYR') or 'all'. Defaults to 'PHI'.
    - season: optional season string (e.g. '20252026'). Defaults to '20252026'.

    This function fetches the api-web schedule for the team and returns the
    top-level 'games' list from the response (or an empty list).
    """
    # Handle the special 'all' request which pages weekly schedule endpoints
    if team == 'all':
        # Pare the season string to a start year (e.g. '20252026' -> '2025')
        season_start_year = season[:4] if season and len(season) >= 4 else '2025'
        try:
            start_year = int(season_start_year)
        except Exception:
            start_year = 2025

        # season typically starts in October of the start year
        week_start_dt = datetime(start_year, 10, 1)
        # stop once we've passed the following calendar year (safe upper bound)
        stop_year = start_year + 1

        games: List[Dict[str, Any]] = []

        # Page week-by-week until we've covered the season year
        while week_start_dt.year <= stop_year:
            date_str = week_start_dt.strftime('%Y-%m-%d')
            url = f'https://api-web.nhle.com/v1/schedule/{date_str}'

            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                # stop on any network or parsing error — we've gathered what we can
                break

            # The schedule endpoint can return either a 'gameWeek' structure or
            # a 'dates' list; handle both shapes defensively.
            if isinstance(data, dict):
                # handle 'gameWeek' shape (older / weekly API)
                gw = data.get('gameWeek')
                if isinstance(gw, list):
                    for day in gw:
                        if not isinstance(day, dict):
                            continue
                        for g in day.get('games', []) or []:
                            if isinstance(g, dict):
                                games.append(g)

                # handle 'dates' shape (the more common schedule API)
                dates = data.get('dates')
                if isinstance(dates, list):
                    for d in dates:
                        if not isinstance(d, dict):
                            continue
                        for g in d.get('games', []) or []:
                            if isinstance(g, dict):
                                games.append(g)

            # advance by one week
            week_start_dt = week_start_dt + timedelta(days=7)

        return games

    TEAM_ABB = (team or 'PHI').upper()

    # The api-web endpoint for a club's schedule for a season
    url = (f"https://api-web.nhle.com/v1/club-schedule-season/"
           f"{TEAM_ABB}/{season}")

    resp = requests.get(url, timeout=10)
    resp.raise_for_status()

    # be defensive about the JSON shape
    data = resp.json()
    if not isinstance(data, dict):
        return []

    games_raw = data.get('games')
    if not isinstance(games_raw, list):
        return []

    # ensure entries are dicts
    games: List[Dict[str, Any]] = [g for g in games_raw if isinstance(g, dict)]

    return games


def get_game_ID(method: str = 'most_recent', team: str = 'PHI') -> int:
    """Return a game ID for the team's schedule found in the api-web response.

    Parameters
    - method: 'most_recent' (default) returns the most recent game on or before
      now; if none found, return the next future game.
    - team: optional team abbreviation (e.g. 'PHI', 'NYR'). Defaults to 'PHI'.

    This function expects the api-web response to contain a top-level 'games'
    list where each game has an 'id' and a start time such as 'startTimeUTC'.

    The function is defensive about timestamp shapes: it first expects an
    ISO-like UTC format (YYYY-MM-DDTHH:MM:SSZ) and will fall back to
    datetime.fromisoformat for other forms that include offsets.
    """
    TEAM_ABB = (team or 'PHI').upper()

    games = get_season(team=TEAM_ABB) or []  # ensure we have a list

    now_utc = datetime.now(timezone.utc)

    past_games: List[Dict[str, Any]] = []
    future_games: List[Dict[str, Any]] = []

    for game in games:
        # skip unexpected entries
        if not isinstance(game, dict):
            continue

        # expected keys: 'id', 'startTimeUTC' e.g. '2025-09-21T23:00:00Z'
        game_id_raw = game.get('id') or game.get('gamePk') or game.get('gameID')
        start_ts = game.get('startTimeUTC') or game.get('gameDate') or game.get('startTime')
        if game_id_raw is None or start_ts is None:
            continue

        # normalize id to int when possible
        try:
            game_id = int(game_id_raw)
        except Exception:
            # fallback: try string -> strip and cast
            try:
                game_id = int(str(game_id_raw).strip())
            except Exception:
                continue

        try:
            # prefer the simple UTC 'Z' format
            start_dt = datetime.strptime(start_ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except Exception:
            # try parsing with fromisoformat as fallback (may include offset)
            try:
                start_dt = datetime.fromisoformat(start_ts)
                if start_dt.tzinfo is None:
                    start_dt = start_dt.replace(tzinfo=timezone.utc)
            except Exception:
                # skip entries that we cannot parse
                continue
        if start_dt <= now_utc:
            past_games.append({'id': game_id, 'start': start_dt})
        else:
            future_games.append({'id': game_id, 'start': start_dt})

    # prefer the most recent past game
    if past_games:
        # sort by start and return last (most recent)
        past_games.sort(key=lambda g: g['start'])
        return int(past_games[-1]['id'])

    # otherwise return the soonest future game
    if future_games:
        future_games.sort(key=lambda g: g['start'])
        return int(future_games[0]['id'])

    raise RuntimeError("No games found in api-web schedule response.")


def get_game_feed(game_ID: int) -> Dict[str, Any]:
    """Fetch and return the play-by-play feed JSON for the requested game.

    The returned value is the parsed JSON response from the NHL `api-web`
    play-by-play endpoint. Callers should be defensive about the structure as
    the NHL has historically varied response shapes across endpoints.
    """
    url = f'https://api-web.nhle.com/v1/gamecenter/{game_ID}/play-by-play'
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    # keep original main behavior for quick command-line inspection
    # show the most-recent PHI game by default
    #games = get_season(team='all')

    game_ID = get_game_ID(method='most_recent', team='PHI')
    feed = get_game_feed(game_ID)
    print('GameID:', game_ID)
    # print a small summary
    if isinstance(feed, dict):
        print('Feed keys:', list(feed.keys())[:10])
