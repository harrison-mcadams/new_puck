import parse

feed = {
    'plays': [
        {
            'type': {'description': 'shot-on-goal'},
            'details': {'xCoord': 50, 'yCoord': 5, 'playerId': 123, 'eventOwnerTeamId': 1},
            'coordinates': {'x': 50, 'y': 5},
            'period': 1,
            'timeRemaining': '10:00',
            'team': {'triCode': 'PHI', 'id': 1},
            'homeTeamDefendingSide': 'left',
            'situationCode': '1010'
        },
        {
            'type': {'description': 'goal'},
            'details': {'xCoord': -45, 'yCoord': -2, 'playerId': 456, 'eventOwnerTeamId': 2},
            'coordinates': {'x': -45, 'y': -2},
            'period': 1,
            'timeInPeriod': '05:00',
            'team': {'triCode': 'NYR', 'id': 2},
            'situationCode': '0101'
        }
    ],
    'homeTeam': {'id': 1, 'abbrev': 'PHI'},
    'awayTeam': {'id': 2, 'abbrev': 'NYR'},
    'id': 2025010036
}

df = parse._game(feed)
print('DF shape:', df.shape)
print(df[['event','x','y','team_id','home_team_defending_side']].to_string(index=False))

