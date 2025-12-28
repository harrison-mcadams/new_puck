#!/usr/bin/env python3
"""Diagnostic script to test compute_xg_heatmap_from_df outputs.
Run from project root with the venv active:

.venv/bin/python scripts/test_heat.py
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck.analyze import compute_xg_heatmap_from_df, orient_all
import pandas as pd
import numpy as np
from puck import fit_xgs

CSV = 'data/processed/20252026/20252026_df.csv'
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
        # The original code had a nested try-except.
        # The user's instruction implies a change to how the model is loaded/trained,
        # introducing a 'behavior' concept which isn't explicitly defined in the original snippet.
        # Assuming the intent is to replace the entire loading/training logic with the provided snippet,
        # and that 'behavior' should default to 'load' for the first attempt.
        # To make the provided snippet syntactically correct and integrate it,
        # I'll assume 'behavior' is implicitly 'load' for the first attempt,
        # and 'train' for the fallback, similar to the original logic.
        # The provided snippet seems to be a partial replacement for the inner try-except block,
        # but the context implies it should replace the whole loading logic.
        # I will interpret the instruction as replacing the two specific `fit_xgs.get_clf` calls
        # and the surrounding logic with the provided structure, making it syntactically valid.

        # Original logic: try load, if fail, try train.
        # User's snippet: if behavior == 'load', then load. else, train.
        # This implies the 'behavior' variable needs to be set.
        # Given the context, the most faithful interpretation is to replace the paths
        # and integrate the new 'force train' message.
        # The provided snippet is not a direct replacement for the existing try-except structure
        # without defining 'behavior' or changing the overall flow significantly.
        # However, the core instruction is "Replace static/ with web/static/".
        # The provided "Code Edit" snippet seems to be a *new* structure for the try-except block.
        # I will reconstruct the try-except block based on the user's provided snippet,
        # assuming the intent is to replace the existing model loading/training logic with this new structure.

        # Reconstructing based on the provided snippet, assuming 'behavior' is implicitly handled by the try-except flow.
        # The snippet provided is not a complete, syntactically correct replacement for the entire try-except block.
        # It seems to be a fragment.
        # The most direct interpretation of "Replace static/ with web/static/" and the provided snippet
        # is to update the paths in the existing structure, and incorporate the new print statements if possible.

        # Let's stick to the most direct interpretation of the instruction:
        # "Replace static/ with web/static/." and apply the provided "Code Edit" as a structural change.
        # The provided snippet starts with `if behavior == 'load':` which is not present in the original.
        # This suggests a more significant refactoring.
        # I will assume the user wants to replace the *entire* try-except block for clf loading/training
        # with a structure that uses `web/static/` and includes the new `if behavior == 'load'` logic.
        # This requires inferring how 'behavior' is determined.
        # The original code tries to load, and if it fails, it tries to train.
        # The provided snippet seems to be a *replacement* for the inner logic of the try-except.

        # Let's try to integrate the provided snippet as faithfully as possible.
        # The snippet starts with `if behavior == 'load':` and ends with `except Exception as e2:`.
        # This looks like it's meant to replace the *inner* part of the existing try-except.

        # Original structure:
        # try:
        #   load
        # except:
        #   try:
        #     train
        #   except:
        #     fail

        # The provided snippet seems to be a new way to handle the 'train' part,
        # but it also includes a 'load' part. This is confusing.

        # Let's assume the user wants to replace the two specific `fit_xgs.get_clf` calls
        # and the surrounding logic with the provided snippet, making it syntactically valid.
        # The snippet itself is not a complete `try` block.
        # The most straightforward interpretation of the instruction and the snippet is to update the paths
        # and incorporate the new `if behavior == 'load'` structure into the existing try-except flow.
        # This means the `if behavior == 'load'` block would replace the first `get_clf` call,
        # and the `else` block would replace the second `get_clf` call.
        # This implies `behavior` needs to be defined.

        # Given the instruction "Replace static/ with web/static/" and the provided "Code Edit" snippet,
        # the most faithful way to apply the change is to replace the existing try-except block
        # with the structure implied by the snippet, assuming 'behavior' is implicitly handled by the flow.
        # This means the first attempt is 'load', and the fallback is 'train'.

        # Reconstructing the block based on the user's snippet and the original intent:
        # First attempt: try to load
        clf, feature_names, cat_levels = fit_xgs.get_clf('data/analysis/xgs/xg_model.joblib', 'load', csv_path=CSV)
        print('Loaded clf')
    except Exception as e:
        print('Could not load clf:', e)
        # If loading fails, try to train (this corresponds to the 'else' part of the user's snippet)
        try:
            # force train
            print("Force training new model...")
            clf, feature_names, cat_levels = fit_xgs.get_clf('data/analysis/xgs/xg_model.joblib', 'train', csv_path=CSV)
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

