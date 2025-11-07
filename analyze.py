## Various different analyses


from typing import Optional

def xgs_map(team: Optional[str] = None, season: str = '20252026', *,
            csv_path: Optional[str] = None,
            model_path: str = 'static/xg_model.joblib',
            behavior: str = 'load',
            out_path: str = 'static/xg_map.png',
            orient_all_left: bool = False,
            events_to_plot: Optional[list] = None,
            show: bool = False,
            return_heatmaps: bool = False,
            condition: Optional[object] = None):
    """Create an xG density map for a season and save a plot.

    Steps implemented:
      1. Locate and load the season CSV (defaults to data/{season}/{season}_df.csv or static/{season}_df.csv).
      2. Obtain a classifier via fit_xgs.get_clf(model_path, behavior, csv_path=...).
         If behavior=='load' and loading fails, the function falls back to training.
      3. Clean the loaded season DataFrame into model inputs using fit_xgs.clean_df_for_model
         (providing the categorical levels map returned at training so encoding is stable).
      4. Predict xG probabilities for the events in the loaded DataFrame and attach as 'xgs'.
      5. Optionally orient all shots to the left (rotate 180° those attacking the right) so the
         visualization is in a consistent frame.
      6. Call plot.plot_events(...) and save the generated image to out_path.

    Parameters
    - team: optional team filter (not used for file discovery here)
    - season: season string used to find CSVs (default '20252026')
    - csv_path: explicit path to the CSV to load; if None the function searches standard locations
    - model_path: where to load/save the xG model
    - behavior: 'load' (try load) or 'train' (force train). If 'load' fails, falls back to 'train'.
    - out_path: destination image path
    - orient_all_left: when True, rotate events so all attacks face the left goal
    - events_to_plot: list of events to pass through to plot.plot_events (default includes 'xgs')
    - show: if True, display the matplotlib figure (requires interactive backend)
    """
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import fit_xgs
    import plot as plot_mod

    # 1) find CSV path
    chosen_csv = None
    if csv_path:
        p = Path(csv_path)
        if p.exists():
            chosen_csv = p
    if chosen_csv is None:
        # prefer data/{season}/{season}_df.csv
        candidates = [Path('data') / season / f'{season}_df.csv',
                      Path('data') / f'{season}_df.csv',
                      Path('static') / f'{season}_df.csv',
                      Path('static') / f'{season}.csv',
                      Path('static') / f'{season}.csv',
                      Path('static') / f'{season}_df.csv']
        for c in candidates:
            try:
                if c.exists():
                    chosen_csv = c
                    break
            except Exception:
                continue
    if chosen_csv is None:
        # last resort: search data/ recursively for a file that contains the season string
        data_dir = Path('data')
        if data_dir.exists():
            found = list(data_dir.rglob(f'*{season}*.csv'))
            if found:
                chosen_csv = found[0]
    if chosen_csv is None:
        raise FileNotFoundError(f'Could not locate a CSV for season {season}. Checked candidates and data/ search.')

    print('xgs_map: loading CSV ->', chosen_csv)
    df = pd.read_csv(chosen_csv)

    # --- Team-based resolution & optional season-level filtering (moved before model prep)
    sel_team_id = None
    sel_team_abb = None
    if team:
        tstr = str(team).strip()
        # try integer id match first
        try:
            tid = int(tstr)
            if ('home_id' in df.columns and tid in df['home_id'].dropna().astype(int).unique()) or (
                'away_id' in df.columns and tid in df['away_id'].dropna().astype(int).unique()):
                sel_team_id = tid
        except Exception:
            tid = None

        if sel_team_id is None:
            # try abbreviation match (case-insensitive)
            tupper = tstr.upper()
            if (('home_abb' in df.columns and tupper in df['home_abb'].dropna().astype(str).str.upper().unique()) or
                ('away_abb' in df.columns and tupper in df['away_abb'].dropna().astype(str).str.upper().unique())):
                sel_team_abb = tupper

        # filter the season dataframe to only rows where this team participates
        if sel_team_id is not None or sel_team_abb is not None:
            mask = pd.Series(False, index=df.index)
            if sel_team_id is not None:
                if 'home_id' in df.columns:
                    mask = mask | (df['home_id'].astype(str) == str(sel_team_id))
                if 'away_id' in df.columns:
                    mask = mask | (df['away_id'].astype(str) == str(sel_team_id))
            if sel_team_abb is not None:
                if 'home_abb' in df.columns:
                    mask = mask | (df['home_abb'].astype(str).str.upper() == sel_team_abb)
                if 'away_abb' in df.columns:
                    mask = mask | (df['away_abb'].astype(str).str.upper() == sel_team_abb)

            n_keep = int(mask.sum())
            if n_keep > 0:
                df = df.loc[mask].copy()
                print(f"Filtered season dataframe to {n_keep} events for team '{team}'")
            else:
                print(f"Warning: no events found for team '{team}' — proceeding without team filtering")
    # --- end moved team filtering ---

    # --- Apply flexible condition-based filtering (same contract as parse.build_mask)
    # `condition` may be None, a dict of column->spec, or a callable(df)->bool Series.
    import parse as _parse
    if condition is not None:
        try:
            cond_mask = _parse.build_mask(df, condition)
            # normalize mask and handle empty
            cond_mask = cond_mask.reindex(df.index).fillna(False).astype(bool)
            n_cond = int(cond_mask.sum())
            if n_cond == 0:
                print(f"Warning: condition {condition!r} matched 0 rows; producing an empty plot without training/loading model")
                # produce an empty DataFrame so downstream code skips model operations
                df = df.iloc[0:0].copy()
                df['xgs'] = float('nan')
                # proceed directly to plotting step below
                pass
            else:
                df = df.loc[cond_mask].copy()
                print(f"Filtered season dataframe to {n_cond} events by condition {condition!r}")
        except Exception as e:
            print('Warning: failed to apply condition filter:', e)

    # 2) get or train classifier — only if we need predictions (i.e., df has rows and xgs not prefilled)
    clf = None
    final_features = None
    categorical_levels_map = None
    # Only attempt to prepare/train/load model if we don't already have 'xgs' column
    if 'xgs' not in df.columns or df['xgs'].notna().any():
        # If df is empty, skip model loading/training to avoid expensive operations
        if df.shape[0] == 0:
            print('xgs_map: filtered dataframe is empty — skipping model load/train and plotting empty map')
        else:
            try:
                clf, final_features, categorical_levels_map = fit_xgs.get_clf(model_path, behavior, csv_path=str(chosen_csv))
            except Exception as e:
                print('xgs_map: get_clf failed with', e, '— trying to train a new model')
                clf, final_features, categorical_levels_map = fit_xgs.get_clf(model_path, 'train', csv_path=str(chosen_csv))

    # 3) prepare the season dataframe for model prediction; use the same features
    features = ['distance', 'angle_deg', 'game_state', 'is_net_empty']
    try:
        df_model, final_feature_cols_game, cat_map_game = fit_xgs.clean_df_for_model(df.copy(), features, fixed_categorical_levels=categorical_levels_map)
    except Exception as e:
        raise RuntimeError('Failed to prepare season df for model: ' + str(e))

    if not final_features:
        # fallback to the derived final_feature_cols_game
        final_features = final_feature_cols_game

    # 4) predict xgs and attach to original df aligning indexes (only if clf is available and df non-empty)
    df['xgs'] = np.nan
    if clf is not None and df_model.shape[0] > 0:
        X = df_model[final_features].values
        try:
            probs = clf.predict_proba(X)[:, 1]
        except Exception as e:
            raise RuntimeError('Failed to predict with classifier: ' + str(e))
        # attach to original df using index alignment
        xgs_series = pd.Series(probs, index=df_model.index)
        df.loc[xgs_series.index, 'xgs'] = xgs_series.values

    # 5) orient all shots to the left if requested
    # Prepare helper to infer which goal a row is attacking (re-uses plot logic)
    left_goal_x, right_goal_x = plot_mod.rink_goal_xs()
    def compute_attacked_goal_x(row):
        try:
            team_id = row.get('team_id')
            home_id = row.get('home_id')
            home_def = row.get('home_team_defending_side')
            if pd.isna(team_id) or pd.isna(home_id):
                return right_goal_x
            # if shooter is home, they attack opposite of what home defends
            if str(team_id) == str(home_id):
                if home_def == 'left':
                    return right_goal_x
                elif home_def == 'right':
                    return left_goal_x
                else:
                    return right_goal_x
            else:
                # shooter is away
                if home_def == 'left':
                    return left_goal_x
                elif home_def == 'right':
                    return right_goal_x
                else:
                    return left_goal_x
        except Exception:
            return right_goal_x

    # default: copy unadjusted coords so we always have x_a/y_a
    df['x_a'] = df.get('x')
    df['y_a'] = df.get('y')

    if team and (sel_team_id is not None or sel_team_abb is not None):
        # Step 2: For the 'team' case, force shots from the selected team to the left
        # and the other team's shots to the right.
        attacked_xs = df.apply(compute_attacked_goal_x, axis=1)

        def is_row_selected(row):
            if sel_team_id is not None:
                return str(row.get('team_id')) == str(sel_team_id)
            if sel_team_abb is not None:
                h_abb = str(row.get('home_abb', '')).upper()
                a_abb = str(row.get('away_abb', '')).upper()
                # shooter is home and home_abb matches OR shooter is away and away_abb matches
                if h_abb == sel_team_abb and str(row.get('team_id')) == str(row.get('home_id')):
                    return True
                if a_abb == sel_team_abb and str(row.get('team_id')) == str(row.get('away_id')):
                    return True
            return False

        # Compute desired goal per row: selected team's shots -> left; others -> right
        desired_goal = df.apply(lambda r: left_goal_x if is_row_selected(r) else right_goal_x, axis=1)

        # Rotate rows where attacked goal != desired goal
        mask_rotate = (attacked_xs != desired_goal) & df['x'].notna() & df['y'].notna()
        df.loc[mask_rotate, 'x_a'] = -df.loc[mask_rotate, 'x']
        df.loc[mask_rotate, 'y_a'] = -df.loc[mask_rotate, 'y']
    elif orient_all_left:
        # Original behavior: orient all shots so home team attacks left
        attacked_xs = df.apply(compute_attacked_goal_x, axis=1)
        mask_rotate = (attacked_xs == right_goal_x) & df['x'].notna() & df['y'].notna()
        df.loc[mask_rotate, 'x_a'] = -df.loc[mask_rotate, 'x']
        df.loc[mask_rotate, 'y_a'] = -df.loc[mask_rotate, 'y']
    # else: leave x_a/y_a as copy of x/y

    # 6) call plotting routine
    if events_to_plot is None:
        events_to_plot = ['xgs']

    print('xgs_map: calling plot.plot_events ->', out_path)
    # request heatmaps from plot_events when requested; handle both return shapes
    if return_heatmaps:
        ret = plot_mod.plot_events(df, events_to_plot=events_to_plot, out_path=out_path, return_heatmaps=True)
        # ret is expected to be (fig, ax, {'home':..., 'away':...}) but be defensive
        if isinstance(ret, (tuple, list)):
            if len(ret) >= 3:
                fig, ax, heatmaps = ret[0], ret[1], ret[2]
            elif len(ret) == 2:
                fig, ax = ret[0], ret[1]
                heatmaps = None
            else:
                raise RuntimeError('Unexpected return from plot.plot_events when return_heatmaps=True')
        else:
            raise RuntimeError('Unexpected return type from plot.plot_events when return_heatmaps=True')

    else:
        ret = plot_mod.plot_events(df, events_to_plot=events_to_plot, out_path=out_path)
        if isinstance(ret, (tuple, list)):
            if len(ret) >= 2:
                fig, ax = ret[0], ret[1]
            else:
                raise RuntimeError('Unexpected return from plot.plot_events; expected (fig, ax)')
        else:
            raise RuntimeError('Unexpected return type from plot.plot_events')
        heatmaps = None

    if show:
        try:
            fig.show()
        except Exception:
            pass

    if return_heatmaps:
        return out_path, heatmaps
    return out_path

if __name__ == '__main__':
    import parse
    import nhl_api
    import pandas as pd


    debug_teamwise = False
    if debug_teamwise:
        game_id = '2025020223'
        game_feed = nhl_api.get_game_feed(game_id)
        df = parse._game(game_feed)
        df = parse._elaborate(df)

    df = pd.read_csv('data/20252026/20252026_df.csv')

    # Get timing information

    condition = {'game_state': ['5v5'],
                 'is_net_empty': False}
    shifts, totals_per_game, totals = parse._timing_impl(df,
                                          condition=condition,
                                          game_col='game_id',
                                          time_col='total_time_elapsed_seconds')
    # next step: make parse._timing_impl able to take additional parameters,
    # and not just game state. i'm envisioning a flexible input like a
    # dictionary, with different keys being the parameters to filter, and the
    # entries for each key being the desired values of those parameters.

    ## next next: make xgs_map able to filter events by state according to
    # the same logic as above

    # example usage
    _, league_maps = xgs_map(season='20252026', condition=condition,
                        out_path='static/xg_map_20252026.png',
            orient_all_left=True, show=False, return_heatmaps=True)

    _, team_maps = xgs_map(team='PHI', season='20252026',
                        out_path='static/PHI_xg_map_20252026.png',
            orient_all_left=False, show=False, return_heatmaps=True)

    print('We are done analyzing')