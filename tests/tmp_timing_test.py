import pandas as pd
import parse

rows = []
for t,s in [(0,'5v5'),(10,'5v5'),(20,'4v4'),(30,'5v5')]:
    rows.append({'game_id':'A','total_time_elapsed_seconds':t,'game_state':s})
for t,s in [(0,'4v4'),(15,'4v4'),(30,'4v4')]:
    rows.append({'game_id':'B','total_time_elapsed_seconds':t,'game_state':s})

df = pd.DataFrame(rows)
print('DF:')
print(df)

print('\nLegacy call: positional args')
shifts_legacy, totals_per_game_legacy, agg_legacy = parse._timing_impl(df, 'game_state', '5v5', game_col='game_id', time_col='total_time_elapsed_seconds')
print('shifts:', shifts_legacy)
print('totals_per_game:', totals_per_game_legacy)
print('agg:', agg_legacy)

print('\nDict condition')
cond = {'game_state':'5v5'}
shifts_new, totals_per_game_new, agg_new = parse._timing_impl(df, condition=cond, game_col='game_id', time_col='total_time_elapsed_seconds')
print('shifts:', shifts_new)
print('totals_per_game:', totals_per_game_new)
print('agg:', agg_new)

print('\nCallable condition')
cond_callable = lambda d: (d['game_state']=='5v5') & (d['total_time_elapsed_seconds']<25)
shifts_call, totals_call, agg_call = parse._timing_impl(df, condition=cond_callable, game_col='game_id', time_col='total_time_elapsed_seconds')
print('shifts:', shifts_call)
print('totals_per_game:', totals_call)
print('agg:', agg_call)

