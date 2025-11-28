#!/usr/bin/env python3
"""Diagnostic script to test compute_xg_heatmap_from_df outputs.
Run from project root with the venv active:

.venv/bin/python scripts/test_heat.py
"""
from analyze import compute_xg_heatmap_from_df, orient_all
import pandas as pd
import numpy as np
import fit_xgs

CSV = 'data/20252026/20252026_df.csv'
TEAM = 'PHI'

def main():
    df = pd.read_csv(CSV)
    # take a sample for speed
    sample = df.sample(min(2000, len(df)), random_state=42)

    # ensure xgs exist: load clf if available, else train
    clf = None
    feature_names = None
    cat_levels = None
    try:
        clf, feature_names, cat_levels = fit_xgs.get_clf('static/xg_model.joblib', 'load', csv_path=CSV)
        print('Loaded clf')
    except Exception as e:
        print('Could not load clf:', e)
        try:
            clf, feature_names, cat_levels = fit_xgs.get_clf('static/xg_model.joblib', 'train', csv_path=CSV)
            print('Trained clf')
        except Exception as e2:
            print('Failed to train clf:', e2)

    features = ['distance', 'angle_deg', 'game_state', 'is_net_empty']
    try:
        df_model, final_features, cat_map = fit_xgs.clean_df_for_model(sample.copy(), features, fixed_categorical_levels=cat_levels)
    except Exception as e:
        print('clean_df_for_model failed:', e)
        df_model = None
        final_features = None

    sample['xgs'] = 0.0
    if clf is not None and df_model is not None and df_model.shape[0] > 0:
        try:
            X = df_model[(feature_names if feature_names is not None else final_features)].values
            probs = clf.predict_proba(X)[:,1]
            sample.loc[df_model.index, 'xgs'] = probs
            print('Predicted xgs for', len(probs))
        except Exception as e:
            print('Prediction failed:', e)

    # filter by 5v5 and net not empty if present
    cond_mask = sample['game_state'].isin(['5v5'])
    if 'is_net_empty' in sample.columns:
        cond_mask = cond_mask & (pd.to_numeric(sample['is_net_empty'], errors='coerce').fillna(0).astype(int) == 0)
    df_cond = sample.loc[cond_mask].copy()
    print('df_cond rows:', len(df_cond))

    # team mask
    mask_team = ((df_cond.get('home_abb', pd.Series(dtype=object)).astype(str).str.upper() == TEAM) |
                 (df_cond.get('away_abb', pd.Series(dtype=object)).astype(str).str.upper() == TEAM))
    print('team rows count:', int(mask_team.sum()))

    # find games
    try:
        games = df_cond.loc[mask_team, 'game_id'].dropna().unique().tolist()
    except Exception:
        games = []
    print('games count:', len(games))
    if games:
        df_games = df_cond.loc[df_cond['game_id'].isin(games)].copy()
    else:
        df_games = pd.concat([df_cond.loc[mask_team], df_cond.loc[~mask_team]]).copy()
    print('df_games rows:', len(df_games))

    # orient
    try:
        df_oriented = orient_all(df_games, target='left', selected_team=TEAM, selected_role='team')
    except Exception as e:
        print('orient_all failed:', e)
        df_oriented = df_games.copy()

    # compute heatmaps
    gx, gy, team_heat, team_xg, team_secs = compute_xg_heatmap_from_df(df_oriented, grid_res=5.0, sigma=8.0, normalize_per60=False, selected_team=TEAM, selected_role='team')
    _, _, opp_heat, opp_xg, opp_secs = compute_xg_heatmap_from_df(df_oriented, grid_res=5.0, sigma=8.0, normalize_per60=False, selected_team=TEAM, selected_role='other')

    print('\nRESULTS:')
    print('team_xg (sum from kernel):', team_xg)
    print('opp_xg (sum from kernel):', opp_xg)
    print('team_heat dtype, shape:', team_heat.dtype, team_heat.shape)
    print('team_heat nan count, total cells:', int(np.isnan(team_heat).sum()), team_heat.size)
    print('team_heat sum (nansum):', float(np.nansum(team_heat)))
    print('team_heat sample center 3x3:')
    r,c = team_heat.shape[0]//2, team_heat.shape[1]//2
    print(team_heat[r-1:r+2, c-1:c+2])

    print('\nopp_heat dtype, shape:', opp_heat.dtype, opp_heat.shape)
    print('opp_heat nan count, total cells:', int(np.isnan(opp_heat).sum()), opp_heat.size)
    print('opp_heat sum (nansum):', float(np.nansum(opp_heat)))
    print('opp_heat sample center 3x3:')
    print(opp_heat[r-1:r+2, c-1:c+2])

if __name__ == '__main__':
    main()

