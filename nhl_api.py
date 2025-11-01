import requests
from datetime import datetime, timezone
from typing import Any, Dict, List


def get_gameID(method: str = 'most_recent') -> int:
    """Return a game ID for the team schedule found in the api-web response.

    method:
      - 'most_recent' (default): return the most recent game on or before now;
        if none found, return the next future game.

    This function expects the api-web response to contain a top-level 'games'
    list where each game has an 'id' and 'startTimeUTC'/'startTimeLocal' fields.
    """
    TEAM_ABB = 'PHI'

    url = (f"https://api-web.nhle.com/v1/club-schedule-season/"
           f"{TEAM_ABB}/20252026")
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    games = data.get('games') or []

    now_utc = datetime.now(timezone.utc)

    past_games: List[Dict[str, Any]] = []
    future_games: List[Dict[str, Any]] = []

    for game in games:
        # expected keys: 'id', 'startTimeUTC' e.g. '2025-09-21T23:00:00Z'
        game_id = game.get('id')
        start_ts = game.get('startTimeUTC') or game.get('gameDate') or game.get('startTime')
        if not game_id or not start_ts:
            continue
        try:
            start_dt = datetime.strptime(start_ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except Exception:
            # try parsing with fromisoformat as fallback (may include offset)
            try:
                start_dt = datetime.fromisoformat(start_ts)
                if start_dt.tzinfo is None:
                    start_dt = start_dt.replace(tzinfo=timezone.utc)
            except Exception:
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


def get_game_feed(gameID: int) -> Dict[str, Any]:
    url = f'https://api-web.nhle.com/v1/gamecenter/{gameID}/play-by-play'
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    # keep original main behavior
    gameID = get_gameID(method='most_recent')
    feed = get_game_feed(gameID)
    print('GameID:', gameID)
    # print a small summary
    if isinstance(feed, dict):
        print('Feed keys:', list(feed.keys())[:10])
