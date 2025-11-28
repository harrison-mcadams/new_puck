import pandas as pd
import analyze

# Build toy dataframe with two games and several timestamps
rows = [
    {'game_id': 'G1', 'total_time_elapsed_seconds': 5, 'x': -40, 'y': 0, 'event': 'shot-on-goal', 'team_id': 'PHI', 'home_id': 'PHI', 'away_id': 'NYR', 'home_abb': 'PHI', 'away_abb': 'NYR'},
    {'game_id': 'G1', 'total_time_elapsed_seconds': 12, 'x': -30, 'y': 5, 'event': 'shot-on-goal', 'team_id': 'PHI', 'home_id': 'PHI', 'away_id': 'NYR', 'home_abb': 'PHI', 'away_abb': 'NYR'},
    {'game_id': 'G1', 'total_time_elapsed_seconds': 35, 'x': -20, 'y': -5, 'event': 'shot-on-goal', 'team_id': 'PHI', 'home_id': 'PHI', 'away_id': 'NYR', 'home_abb': 'PHI', 'away_abb': 'NYR'},
    {'game_id': 'G1', 'total_time_elapsed_seconds': 50, 'x': 10, 'y': 2, 'event': 'shot-on-goal', 'team_id': 'NYR', 'home_id': 'PHI', 'away_id': 'NYR', 'home_abb': 'PHI', 'away_abb': 'NYR'},
    {'game_id': 'G2', 'total_time_elapsed_seconds': 100, 'x': -10, 'y': 1, 'event': 'shot-on-goal', 'team_id': 'PHI', 'home_id': 'PHI', 'away_id': 'BOS', 'home_abb': 'PHI', 'away_abb': 'BOS'},
]

df = pd.DataFrame(rows)

# intervals: for G1, team intersection intervals are [10-20] and [30-40]; for G2 none
intervals_obj = {
    'per_game': {
        'G1': {
            'sides': {
                'team': {
                    'intersection_intervals': [[10,20],[30,40]]
                }
            }
        },
        'G2': {
            'sides': {
                'team': {
                    'intersection_intervals': []
                }
            }
        }
    }
}

print('Input df:')
print(df)

out = analyze.xgs_map(season='20252026', data_df=df, use_intervals=True, intervals_input=intervals_obj, condition={'team':'PHI'}, return_heatmaps=False, return_filtered_df=True)
print('\nResult from xgs_map:')
print(out)

# If returned df exists, print it
if isinstance(out, tuple) and len(out) >= 3:
    print('\nFiltered df rows:')
    print(out[2])
else:
    print('\nDid not receive expected return structure')

