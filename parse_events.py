def parse_shot_and_goal_events(game_feed):
    """Extract shot and goal events with coordinates from a game feed.

    Returns a list of dicts with keys: event, x, y, period, periodTime, player
    """
    plays = game_feed.get("plays", {})
    events = []
    for p in plays:
        # Saving this as an example of the contents of p: {'eventId': 100, 'periodDescriptor': {'number': 1, 'periodType': 'REG', 'maxRegulationPeriods': 3}, 'timeInPeriod': '03:31', 'timeRemaining': '16:29', 'situationCode': '1551', 'homeTeamDefendingSide': 'right', 'typeCode': 506, 'typeDescKey': 'shot-on-goal', 'sortOrder': 57, 'details': {'xCoord': 63, 'yCoord': 34, 'zoneCode': 'O', 'shotType': 'snap', 'shootingPlayerId': 8475287, 'goalieInNetId': 8478435, 'eventOwnerTeamId': 18, 'awaySOG': 2, 'homeSOG': 0}}

        ev_type = p.get("typeDescKey", {})
        if ev_type not in ('shot-on-goal', 'missed-shot', 'goal', 'blocked-shot'):
            continue
        coords = p.get("coordinates", {})
        x = p['details'].get("xCoord")
        y = p['details'].get("yCoord")
        if x is None or y is None:
            continue
        event = {
            "event": ev_type,
            "x": x,
            "y": y,
            "period": p.get("periodDescriptor", {}).get("number"),
            "periodTime": p.get("timeRemaining", {}),
            "player": None,
        }
        # attempt to get the shooter/scorer
        event['playerID'] = p.get("details", {}).get('shootingPlayerId') or []

        events.append(event)
    return events

