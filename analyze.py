## Various different analyses

from typing import Optional

def xgs_map(team: Optional[str] = None, season: str = '20252026', *,
            csv_path: Optional[str] = None,
            model_path: str = 'static/xg_model.joblib',
            behavior: str = 'load',
            out_path: str = 'static/xg_map.png',
            orient_all_left: bool = False,
            events_to_plot: Optional[list] = None,
            show: bool = False):
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
    import os
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

    # 2) get or train classifier
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

    # 4) predict xgs and attach to original df aligning indexes
    X = df_model[final_features].values
    try:
        probs = clf.predict_proba(X)[:, 1]
    except Exception as e:
        raise RuntimeError('Failed to predict with classifier: ' + str(e))

    # attach to original df using index alignment
    df['xgs'] = np.nan
    xgs_series = pd.Series(probs, index=df_model.index)
    df.loc[xgs_series.index, 'xgs'] = xgs_series.values

    # 5) orient all shots to the left if requested
    if orient_all_left:
        # implement orientation: if an event is attacking the right, rotate 180deg
        left_goal_x, right_goal_x = plot_mod.rink_goal_xs()
        def compute_attacked_goal_x(row):
            # replicate logic from parse._elaborate / plot.adjust_xy_for_homeaway
            try:
                team_id = row.get('team_id')
                home_id = row.get('home_id')
                home_def = row.get('home_team_defending_side')
                if pd.isna(team_id) or pd.isna(home_id):
                    return right_goal_x
                if int(team_id) == int(home_id):
                    # shooter is home -> attacking goal opposite of what home defends
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

        attacked_xs = df.apply(compute_attacked_goal_x, axis=1)
        # If attacked_x == right_goal_x, rotate 180: (x,y)->(-x,-y)
        df['x_a'] = df['x']
        df['y_a'] = df['y']
        mask_rotate = (attacked_xs == right_goal_x) & df['x'].notna() & df['y'].notna()
        df.loc[mask_rotate, 'x_a'] = -df.loc[mask_rotate, 'x']
        df.loc[mask_rotate, 'y_a'] = -df.loc[mask_rotate, 'y']
    else:
        # rely on plot.adjust_xy_for_homeaway to construct x_a/y_a as needed
        df['x_a'] = df.get('x')
        df['y_a'] = df.get('y')

    # 6) call plotting routine
    if events_to_plot is None:
        events_to_plot = ['xgs']

    print('xgs_map: calling plot.plot_events ->', out_path)
    fig, ax = plot_mod.plot_events(df, events_to_plot=events_to_plot, out_path=out_path)

    if show:
        try:
            fig.show()
        except Exception:
            pass

    return out_path

if __name__ == '__main__':
    # example usage
    xgs_map(season='20252026', out_path='static/xg_map_20252026.png',
            orient_all_left=True, show=True)