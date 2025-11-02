"""Parse various NHL game-feed shapes and extract shot/goal events.

This module exposes `_game` which normalizes api-web play objects into a
consistent list of event dictionaries. It also provides `_season` which
fetches a season's games and returns a concatenated pandas.DataFrame
suitable for ML.
"""

import math
import logging
from typing import List, Dict, Any, Optional

import pandas as pd

import nhl_api


logging.basicConfig(level=logging.INFO)


def _game(game_feed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract shot and goal events with coordinates from an api-web game feed.

    This parser intentionally supports the api-web/play-by-play shape where a
    top-level `plays` (or `playByPlay.play`) list contains events with
    `details` and `coordinates`.

    Returns a list of event dicts with keys:
      - event: 'SHOT' or 'GOAL'
      - x, y: float coordinates
      - period, periodTime
      - playerID, teamID, teamAbbrev
      - homeTeamDefendingSide, gameID
    """
    events: List[Dict[str, Any]] = []

    plays = None
    if isinstance(game_feed, dict):
        plays = game_feed.get('plays') or game_feed.get('playByPlay', {}).get('plays')

    if not isinstance(plays, list):
        # nothing we can parse
        logging.debug('_game(): no plays list found in feed; keys=%s', list(game_feed.keys()) if isinstance(game_feed, dict) else type(game_feed))
        return events

    for p in plays:
        if not isinstance(p, dict):
            continue

        # api-web often uses a textual descriptor key such as 'typeDescKey'
        ev_type = p.get('typeDescKey') or (p.get('type') or {}).get('description') or p.get('typeCode')
        if not isinstance(ev_type, str):
            continue
        r = ev_type.strip().lower()
        if r in ('shot-on-goal', 'shot_on_goal', 'shot', 'shot on goal'):
            ev_norm = 'SHOT'
        elif r == 'goal':
            ev_norm = 'GOAL'
        else:
            continue

        details = p.get('details') or p.get('detail') or {}
        coords = p.get('coordinates') or {}

        x = None
        y = None
        if isinstance(details, dict):
            x = details.get('xCoord') if x is None else x
            y = details.get('yCoord') if y is None else y
        if x is None:
            x = coords.get('x')
        if y is None:
            y = coords.get('y')
        if x is None or y is None:
            continue

        # period info may be at different keys
        period = None
        if isinstance(p.get('periodDescriptor'), dict):
            period = p.get('periodDescriptor', {}).get('number')
        else:
            period = p.get('period') or p.get('periodNumber')
        period_time = p.get('timeRemaining') or p.get('timeInPeriod') or None

        player_id = None
        if isinstance(details, dict):
            player_id = details.get('shootingPlayerId') or details.get('playerId')

        team_id = details.get('eventOwnerTeamId') if isinstance(details, dict) else None
        team_abbrev = None
        team_obj = p.get('team') or {}
        if isinstance(team_obj, dict):
            team_abbrev = team_obj.get('triCode') or team_obj.get('abbrev') or team_obj.get('name')
            team_id = team_id or team_obj.get('id')

        home_side = p.get('homeTeamDefendingSide') or p.get('home_team_defending_side')

        try:
            events.append({
                'event': ev_norm,
                'x': float(x),
                'y': float(y),
                'period': period,
                'periodTime': period_time,
                'playerID': player_id,
                'teamID': team_id,
                'teamAbbrev': team_abbrev,
                'homeTeamDefendingSide': home_side,
                'gameID': game_feed.get('id') or game_feed.get('gamePk')
            })
        except Exception:
            # skip rows that fail numeric conversion
            continue

    return events


def _period_time_to_seconds(t: Optional[str]) -> Optional[int]:
    """Convert a period time like '12:34' to seconds remaining or elapsed.

    If the input is None or not parseable, return None.
    """
    if t is None:
        return None
    if isinstance(t, (int, float)):
        return int(t)
    try:
        parts = str(t).split(':')
        if len(parts) == 2:
            mm, ss = parts
            return int(mm) * 60 + int(ss)
        return int(float(t))
    except Exception:
        return None


def _season(season: str = '20252026', team: str = 'all', out_path: Optional[str] = None) -> pd.DataFrame:
    """Fetch a full season (or a team's season) and return a concatenated
    pandas.DataFrame of shot and goal events suitable for ML.

    Parameters
    - season: season string like '20252026' (default).
    - team: pass a team abbreviation to limit to one club, or 'all' (default).
    - out_path: optional folder path to save the resulting CSV as `<out_path>/<season>.csv`.

    Returns a pandas.DataFrame with derived numeric columns (is_goal,
    periodTime_seconds, dist_center, angle_deg) in addition to the raw fields.
    """
    games = nhl_api.get_season(team=team, season=season)

    records: List[Dict[str, Any]] = []

    for gm in games:
        # game dictionaries from different endpoints may use different id keys
        game_ID = gm.get('id') or gm.get('gamePk') or gm.get('gameID')
        if game_ID is None:
            continue

        try:
            game_feed = nhl_api.get_game_feed(game_ID)
        except Exception as e:
            logging.warning('Failed to fetch feed for game %s: %s', game_ID, e)
            continue

        # Use the canonical parser function `_game` to extract events
        try:
            events = _game(game_feed)
        except Exception as e:
            logging.warning('Parser error for game %s: %s', game_ID, e)
            continue

        records.append(game_feed)

    if records:
        df = pd.DataFrame.from_records(records)
    else:
        df = pd.DataFrame()

    if out_path and not df.empty:
        try:
            df.to_csv(out_path + '/' + season + '.csv', index=False)
            logging.info('Saved season data to %s', out_path + '/' + season + '.csv')
        except Exception as e:
            logging.warning('Failed to save CSV %s: %s', out_path, e)

    return df

def _elaborate(game_feed: Dict[str, Any]) -> List[Dict[str, Any]]:

    elaborated_game_feed = []

    for ev in game_feed:
        rec = dict(ev)  # shallow copy
        #rec['season'] = season
        # Normalize and derive helpful ML features
        rec['is_goal'] = 1 if rec.get('event') == 'GOAL' else 0
        # ensure numeric period when possible
        try:
            rec['period'] = int(rec['period']) if rec.get(
                'period') is not None else None
        except Exception:
            rec['period'] = None
        rec['periodTime_seconds'] = _period_time_to_seconds(
            rec.get('periodTime'))

        # x/y to numeric
        try:
            x = float(rec.get('x'))
            y = float(rec.get('y'))
        except Exception:
            x = None
            y = None
        rec['x'] = x
        rec['y'] = y

        if x is not None and y is not None:
            rec['dist_center'] = math.hypot(x, y)
            rec['angle_deg'] = math.degrees(math.atan2(y, x))
        else:
            rec['dist_center'] = None
            rec['angle_deg'] = None

        elaborated_game_feed.append(rec)


    return elaborated_game_feed

if __name__ == '__main__':
    df = _season(out_path='/Users/harrisonmcadams/PycharmProjects/new_puck/static')
    print('Season dataframe shape:', df.shape)
    if not df.empty:
        print(df.head())
    else:
        print('No events found for season')
