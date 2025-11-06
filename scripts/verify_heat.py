from pathlib import Path
import pandas as pd
import numpy as np

from plot import rink_half_height_at_x

p = Path('data') / '20252026' / '20252026_df.csv'
df = pd.read_csv(p)
if 'xgs' not in df.columns:
    print('No xgs column; abort')
    raise SystemExit(1)

events = df[pd.to_numeric(df['xgs'], errors='coerce').notna()].copy()
print('events with xgs:', len(events))

hm_sigma = 6.0
hm_res = 1.0

# ensure adjusted coords exist
if 'x_a' not in events.columns or 'y_a' not in events.columns:
    events['x_a'] = events['x']
    events['y_a'] = events['y']

xs = pd.to_numeric(events['x_a'], errors='coerce').values
ys = pd.to_numeric(events['y_a'], errors='coerce').values
amps = pd.to_numeric(events['xgs'], errors='coerce').values
team_ids = events['team_id'].astype(str).fillna('').values if 'team_id' in events.columns else None
home_ids = events['home_id'].astype(str).fillna('').values if 'home_id' in events.columns else None

xmin, xmax = -100.0, 100.0
ymin, ymax = -42.5, 42.5

gx = np.arange(xmin, xmax + hm_res, hm_res)
gy = np.arange(ymin, ymax + hm_res, hm_res)
XX, YY = np.meshgrid(gx, gy)
heat_home = np.zeros_like(XX, dtype=float)
heat_away = np.zeros_like(XX, dtype=float)

two_sigma2 = 2.0 * (hm_sigma ** 2)
# normalization so discrete sum approximates integral = ai
norm_factor = (hm_res ** 2) / (2.0 * np.pi * (hm_sigma ** 2))

for xi, yi, ai, tid, hid in zip(xs, ys, amps, team_ids, home_ids):
    if np.isnan(xi) or np.isnan(yi) or np.isnan(ai):
        continue
    dx = XX - xi
    dy = YY - yi
    kern = ai * norm_factor * np.exp(-(dx * dx + dy * dy) / two_sigma2)
    if tid is not None and hid is not None and tid != '' and hid != '':
        is_home = (str(tid) == str(hid))
    else:
        is_home = (xi < 0)
    if is_home:
        heat_home += kern
    else:
        heat_away += kern

sum_heat = float(np.nansum(heat_home) + np.nansum(heat_away))
sum_xgs = float(np.nansum(amps))
cell_area = hm_res * hm_res
approx_integral_sum = sum_heat * cell_area
expected_integral = sum_xgs * 2.0 * np.pi * (hm_sigma ** 2)

from rink import rink_half_height_at_x
mask_rink = np.vectorize(rink_half_height_at_x)(XX) >= np.abs(YY)
sum_masked = float(np.nansum((heat_home + heat_away) * mask_rink))

print('sum_xgs =', sum_xgs)
print('sum_heat (grid values sum) =', sum_heat)
print('sum_masked =', sum_masked)
print('sum_heat * cell_area =', approx_integral_sum)
print('expected_integral (sum_xgs * 2*pi*sigma^2) =', expected_integral)
print('ratio approx_integral / expected =', approx_integral_sum / expected_integral if expected_integral>0 else float('nan'))
print('\nNotes:')
print('- If plot uses norm_factor as implemented, sum_heat should be close to sum_xgs')
print('- Because we then multiply by cell_area to compare to continuous integral, approx_integral should be close to expected_integral (modulo rink mask and edge truncation)')

print('\nDone')

