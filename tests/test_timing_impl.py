import pandas as pd
import parse

# Build small synthetic dataset with two games
rows = []
# game 1: events at times 0, 10, 20, 30
rows.append({'game_id': 'g1', 'total_time_elapsed_seconds': 0, 'game_state': '5v5', 'is_net_empty': False, 'home_id': 4, 'away_id': 2})
rows.append({'game_id': 'g1', 'total_time_elapsed_seconds': 10, 'game_state': '5v5', 'is_net_empty': False, 'home_id': 4, 'away_id': 2})
rows.append({'game_id': 'g1', 'total_time_elapsed_seconds': 20, 'game_state': '5v4', 'is_net_empty': False, 'home_id': 4, 'away_id': 2})
rows.append({'game_id': 'g1', 'total_time_elapsed_seconds': 30, 'game_state': '5v5', 'is_net_empty': True, 'home_id': 4, 'away_id': 2})
# game 2: events at times 0, 15, 30
rows.append({'game_id': 'g2', 'total_time_elapsed_seconds': 0, 'game_state': '5v5', 'is_net_empty': False, 'home_id': 3, 'away_id': 4})
rows.append({'game_id': 'g2', 'total_time_elapsed_seconds': 15, 'game_state': '4v4', 'is_net_empty': False, 'home_id': 3, 'away_id': 4})
rows.append({'game_id': 'g2', 'total_time_elapsed_seconds': 30, 'game_state': '5v5', 'is_net_empty': False, 'home_id': 3, 'away_id': 4})

df = pd.DataFrame(rows)
print('DF:\n', df)

# 1) AND condition: game_state == '5v5' and is_net_empty == False
cond_and = {'game_state': ['5v5'], 'is_net_empty': False}
ints, totals_per_game, agg = parse._timing_impl(df, condition=cond_and, game_col='game_id', time_col='total_time_elapsed_seconds')
print('\nAND condition results:')
print('intervals_per_game:', ints)
print('totals_per_game:', totals_per_game)
print('aggregate:', agg)

# 2) team condition: team == 4 (should match rows where home_id==4 or away_id==4)
cond_team = {'team': 4}
ints2, totals2, agg2 = parse._timing_impl(df, condition=cond_team, game_col='game_id', time_col='total_time_elapsed_seconds')
print('\nTEAM condition results (team=4):')
print('intervals_per_game:', ints2)
print('totals_per_game:', totals2)
print('aggregate:', agg2)

# 3) Legacy positional args: ('game_state','5v5')
ints3, totals3, agg3 = parse._timing_impl(df, 'game_state', '5v5', game_col='game_id', time_col='total_time_elapsed_seconds')
print('\nLegacy positional args results:')
print('intervals_per_game:', ints3)
print('totals_per_game:', totals3)
print('aggregate:', agg3)

