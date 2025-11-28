import pandas as pd
import analyze

df = pd.DataFrame([{
    'x_a': 0.0,
    'y_a': 0.0,
    'xgs': 0.5,
    'team_id': '1',
    'home_id': '1',
    'away_id': '2',
    'home_abb': 'PHI',
    'away_abb': 'NYR',
    'game_id': '2025010001',
    'total_time_elapsed_seconds': 0
}])

gx, gy, heat, total_xg, total_seconds = analyze.compute_xg_heatmap_from_df(df, grid_res=10.0, sigma=8.0, normalize_per60=True, total_seconds=3600)

print('gx len', len(gx), 'gy len', len(gy))
print('total_xg', total_xg, 'total_seconds', total_seconds)
print('corner (0,0):', repr(heat[0,0]))
print('center index:', len(gy)//2, len(gx)//2, 'value:', repr(heat[len(gy)//2, len(gx)//2]))

import numpy as _np
n_nans = int(_np.sum(_np.isnan(heat)))
print('n_nans', n_nans, 'shape', heat.shape)
print('frac_nans', n_nans / (heat.size))

