"""Parse various NHL game-feed shapes and extract shot/goal events.

This module exposes a parser function `game` which normalizes different
play-by-play shapes into a consistent list of event dictionaries. It also
provides `season` which fetches the season's games and returns a single
pandas.DataFrame concatenating all shot/goal events for model-ready use.
"""

import math
import logging
from typing import List, Dict, Any, Optional

import pandas as pd

import nhl_api


logging.basicConfig(level=logging.INFO)


def _game(game_feed):
    """Extract shot and goal events with coordinates from various game-feed shapes.

    Supports:
      - statsapi live feed: game_feed['liveData']['plays']['allPlays']
      - api-web play-by-play: top-level game_feed['plays'] list where each play
        has 'details' and 'coordinates'

    Returns a list of dicts with keys: event, x, y, period, periodTime, playerID, teamID, teamAbbrev, homeTeamDefendingSide
    Only events returned are 'SHOT' (for shot-on-goal) and 'GOAL'.
    """
    events = []

    # 1) Try statsapi live feed shape: liveData -> plays -> allPlays
    live_plays = game_feed.get('liveData', {}).get('plays', {}).get('allPlays') if isinstance(game_feed.get('liveData', {}), dict) else None
    if isinstance(live_plays, list):
        for p in live_plays:
            # result block may contain either 'eventTypeId' or 'event'
            ev_type = (p.get('result', {}) or {}).get('eventTypeId') or (p.get('result', {}) or {}).get('event')
            if not ev_type:
                continue
            r = str(ev_type).strip().upper()
            if r == 'SHOT' or r == 'GOAL':
                ev_norm = 'SHOT' if r == 'SHOT' else 'GOAL'
            else:
                continue

            # coordinates are usually at p['coordinates'] but some live feeds nest them
            coords = p.get('coordinates', {}) or {}
            x = coords.get('x')
            y = coords.get('y')
            if x is None or y is None:
                # some live feeds use different keys inside 'result' -> details; fall back
                details = p.get('result', {}).get('details', {}) if isinstance(p.get('result', {}), dict) else {}
                x = x or details.get('x') or details.get('xCoord')
                y = y or details.get('y') or details.get('yCoord')
            if x is None or y is None:
                continue

            period = (p.get('about', {}) or {}).get('period')
            period_time = (p.get('about', {}) or {}).get('periodTime')

            # Try to extract the shooter/scorer id from the players block when present
            player_id = None
            players = p.get('players') or []
            if isinstance(players, list):
                for pl in players:
                    if pl.get('playerType') in ('Shooter', 'Scorer'):
                        player_id = (pl.get('player') or {}).get('id') or (pl.get('player') or {}).get('playerId') or (pl.get('player') or {}).get('fullName')
                        break

            team_id = (p.get('team') or {}).get('id') if isinstance(p.get('team', {}), dict) else None
            team_abbrev = (p.get('team') or {}).get('triCode') if isinstance(p.get('team', {}), dict) else None
            # homeTeamDefendingSide may be present in play-level keys in some feeds
            home_side = p.get('homeTeamDefendingSide') or p.get('home_team_defending_side')

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
        if events:
            return events

    # 2) Try api-web style top-level plays list with details/coordinates
    plays = game_feed.get('plays') or game_feed.get('playByPlay', {}).get('plays') if isinstance(game_feed, dict) else None
    if isinstance(plays, list):
        for p in plays:
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
        if events:
            return events

    # nothing found
    print('parse_shot_and_goal_events: no SHOT/GOAL events found in feed; feed keys:', list(game_feed.keys()) if isinstance(game_feed, dict) else type(game_feed))
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


def _season(season: str = '20252026', team: str = 'all', out_path: Optional[
    str] = None) -> pd.DataFrame:
    """Fetch a full season (or a team's season) and return a concatenated
    pandas.DataFrame of shot and goal events suitable for ML.

    Parameters
    - season: season string like '20252026' (default).
    - team: pass a team abbreviation to limit to one club, or 'all' (default).
    - save_csv: optional path to save the resulting DataFrame as CSV.

    Returns a pandas.DataFrame with derived numeric columns (is_goal,
    periodTime_seconds, dist_center, angle_deg) in addition to the raw fields.
    """
    games = nhl_api.get_season(team=team, season=season)

    records: List[Dict[str, Any]] = []

    for game in games:
        # game dictionaries from different endpoints may use different id keys
        game_ID = game.get('id') or game.get('gamePk') or game.get('gameID')
        if game_ID is None:
            continue

        try:
            game_feed = nhl_api.get_game_feed(game_ID)
        except Exception as e:
            logging.warning('Failed to fetch feed for game %s: %s', game_ID, e)
            continue

        try:
            events = _game(game_feed)
        except Exception as e:
            logging.warning('Parser error for game %s: %s', game_ID, e)
            continue

        for ev in events:
            rec = dict(ev)  # shallow copy
            rec['season'] = season
            # Normalize and derive helpful ML features
            rec['is_goal'] = 1 if rec.get('event') == 'GOAL' else 0
            # ensure numeric period when possible
            try:
                rec['period'] = int(rec['period']) if rec.get('period') is not None else None
            except Exception:
                rec['period'] = None
            rec['periodTime_seconds'] = _period_time_to_seconds(rec.get('periodTime'))

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

            # attach game-level metadata when available
            rec['game_pk'] = game_ID
            rec['game_date'] = game.get('date') or game.get('gameDate') or game.get('startTimeUTC')

            records.append(rec)

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


if __name__ == '__main__':
    df = _season(out_path='/Users/harrisonmcadams/PycharmProjects/new_puck/static')
    print('Season dataframe shape:', df.shape)
    if not df.empty:
        print(df.head().to_string())
    else:
        print('No events found for season')

