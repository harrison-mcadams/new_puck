def parse_shot_and_goal_events(game_feed):
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
            ev_type = (p.get('result', {}) or {}).get('eventTypeId') or (p.get('result', {}) or {}).get('event')
            if not ev_type:
                continue
            r = str(ev_type).strip().upper()
            if r == 'SHOT' or r == 'GOAL':
                ev_norm = 'SHOT' if r == 'SHOT' else 'GOAL'
            else:
                continue
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
            })
        if events:
            return events

    # 2) Try api-web style top-level plays list with details/coordinates
    plays = game_feed.get('plays') or game_feed.get('playByPlay', {}).get('plays') if isinstance(game_feed, dict) else None
    if isinstance(plays, list):
        for p in plays:
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
            })
        if events:
            return events

    # nothing found
    print('parse_shot_and_goal_events: no SHOT/GOAL events found in feed; feed keys:', list(game_feed.keys()) if isinstance(game_feed, dict) else type(game_feed))
    return events
