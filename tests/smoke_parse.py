import parse
import pandas as pd

# Minimal synthetic game feed matching expected structure
feed = {
    'homeTeam': {'id': 10, 'abbrev': 'HOME'},
    'awayTeam': {'id': 20, 'abbrev': 'AWAY'},
    'plays': [
        {
            'type': {'description': 'shot-on-goal'},
            'details': {
                'xCoord': -80,
                'yCoord': 5,
                'shootingPlayerId': 123,
                'eventOwnerTeamId': 10,
                'situationCode': ('0','5','5','1')
            },
            'period': 1,
            'timeInPeriod': '12:34',
            'homeTeamDefendingSide': 'left',
            'team': {'triCode': 'HOME', 'id': 10}
        },
        {
            'type': {'description': 'goal'},
            'details': {
                'xCoord': 75,
                'yCoord': -3,
                'shootingPlayerId': 321,
                'eventOwnerTeamId': 20,
                'situationCode': ('0','5','5','1')
            },
            'period': 2,
            'timeInPeriod': '05:10',
            'homeTeamDefendingSide': 'left',
            'team': {'triCode': 'AWAY', 'id': 20}
        }
    ]
}

print('Running parse._game(feed)')
df = parse._game(feed)
print(df)

print('\nRunning parse._elaborate(df)')
df2 = parse._elaborate(df)
print(df2)

print('\nSmoke test complete')

