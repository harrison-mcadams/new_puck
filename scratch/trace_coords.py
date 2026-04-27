import pandas as pd
import numpy as np
from puck import analyze, data_pipeline, correction
df = pd.read_csv(analyze.locate_season_csv('20232024'))
b = df[df['event']=='blocked-shot'].iloc[0:1].copy()

print('RAW X:', b['x'].values[0])
print('RAW EVENT:', b['event'].values[0])

# Fix Attribution
b = correction.fix_blocked_shot_attribution(b)
print('POST CORRECTION X:', b['x'].values[0])
print('POST CORRECTION TEAM_ID:', b['team_id'].values[0])

# Preprocess Features manually
tid_s = b['team_id'].astype(str).str.replace(r'\.0$', '', regex=True)
hid_s = b['home_id'].astype(str).str.replace(r'\.0$', '', regex=True)
b['is_home'] = (tid_s == hid_s).astype(int)

print('IS HOME:', b['is_home'].values[0])

side_str = b['home_team_defending_side'].astype(str).str.lower().str.strip()
def_side_sign = side_str.map({'left': -1, 'right': 1}).fillna(1)
is_home_ser = (b['is_home'] == 1)
side_mult = np.where(is_home_ser.values, -1, 1)

print('SIDE MULT:', side_mult[0])
print('DEF SIDE SIGN:', def_side_sign.values[0])

attacking_side = def_side_sign.values * side_mult
mask_flip = (attacking_side == -1)

print('ATTACKING SIDE:', attacking_side[0])
print('MASK FLIP:', mask_flip[0])

if mask_flip[0]:
    b['x'] *= -1
    if 'x_adj' in b.columns:
        b['x_adj'] *= -1

print('POST FLIP X:', b['x'].values[0])
if 'x_adj' in b.columns:
    print('POST FLIP X ADJ:', b['x_adj'].values[0])
    b['x'] = b['x_adj']
    print('FINAL ASSIGNMENT X:', b['x'].values[0])

