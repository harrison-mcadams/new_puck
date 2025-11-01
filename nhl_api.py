import requests
from datetime import date, timedelta, datetime, timezone
from typing import Any, Dict, List, Optional

def get_gameID(method):
    ## The default method is to search for the most recent Flyers game.
    TEAM_ABB = 'PHI'

    today = date.today().strftime("%Y-%m-%d")

    url = (f"https://api-web.nhle.com/v1/club-schedule-season/"
           f"{TEAM_ABB}/20252026")
    response = requests.get(url)
    response = response.json()
    games = response['games']

    # From these games, identify the gameID that's closest to today but not
    # after today.
    gameIDs = []
    for game in games:

        # Set target time. Default target time will be now
        now_time = datetime.now()

        game_date = game['gameDate']
        game_start_time_UTC = datetime.strptime(game['startTimeUTC'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

        # Compare game_start_time to now
        now_UTC = datetime.now(timezone.utc)

        # compare game_start_time_UTC to now_UTC

        if method == 'most_recent':
            if game_start_time_UTC <= now_UTC:
                gameID = game['id']
            else:
                continue

    return gameID

def get_game_feed(gameID: int) -> Dict[str, Any]:
    url = 'https://api-web.nhle.com/v1/gamecenter/' + str(gameID) + '/play-by-play'
    response = requests.get(url)
    game_feed = response.json()
    return game_feed

if __name__ == "__main__":


    # keep original main behavior
    gameID = get_gameID(method='most_recent')
    get_game_feed(gameID)

    print('GameID:'+str(gameID))
