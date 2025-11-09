## Various different analyses


from typing import Optional

def xgs_map(season: str = '20252026', *,
             csv_path: Optional[str] = None,
             model_path: str = 'static/xg_model.joblib',
             behavior: str = 'load',
             out_path: str = 'static/xg_map.png',
             orient_all_left: bool = False,
             events_to_plot: Optional[list] = None,
             show: bool = False,
             return_heatmaps: bool = True,
             # when True, return the filtered dataframe used to create the map
             return_filtered_df: bool = True,
             condition: Optional[object] = None):
    """Create an xG density map for a season and save a plot.

    This function is intentionally written as a clear sequence of steps with
    small local helpers so it's easy to read and maintain. The external
    behavior (return value and saved image) is unchanged.

    High level steps (implemented below):
      1) Locate and load a season CSV.
      2) Apply an optional flexible `condition` filter (uses `parse.build_mask`).
         If `condition` is a dict with a 'team' key, that is treated as a
         request to match either home or away team (by id or abbreviation).
      3) If needed, obtain/train an xG classifier and predict probabilities
         on the filtered rows (attach as column 'xgs'). Prediction only runs
         when the filtered df is non-empty and 'xgs' either doesn't exist or
         contains only NaN values.
      4) Optionally orient coordinates so selected shots face left (or all).
      5) Call the plotting routine and save/return outputs.

    Parameters mirror the previous implementation; returning (out_path, heatmaps)
    when `return_heatmaps=True` and just `out_path` otherwise.
    """

    # Local imports placed here to avoid module-level side-effects when importing analyze
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import fit_xgs
    import plot as plot_mod
    import parse as _parse

    # --- Helpers ------------------------------------------------------------
    def _locate_csv() -> Path:
        """Find a CSV for the given season using a prioritized list of
        candidates, then a recursive search of data/ if necessary.
        """
        if csv_path:
            p = Path(csv_path)
            if p.exists():
                return p
        candidates = [Path('data') / season / f'{season}_df.csv',
                      Path('data') / f'{season}_df.csv',
                      Path('static') / f'{season}_df.csv',
                      Path('static') / f'{season}.csv']
        for c in candidates:
            try:
                if c.exists():
                    return c
            except Exception:
                continue
        # fallback: find any CSV under data/ matching season
        data_dir = Path('data')
        if data_dir.exists():
            found = list(data_dir.rglob(f'*{season}*.csv'))
            if found:
                return found[0]
        raise FileNotFoundError(f'Could not locate a CSV for season {season}.')

    def _apply_condition(df: pd.DataFrame):
        """Apply `condition` to df and return (filtered_df, team_val).

        Accepts None, dict, or callable. If a 'team' key is present in the
        dict it is pulled out and used to filter by team (home or away).
        Key normalization is tolerant of common naming variants.
        """
        # copy condition if dict to avoid mutation
        cond_work = condition.copy() if isinstance(condition, dict) else condition

        team_val = None
        if isinstance(cond_work, dict) and 'team' in cond_work:
            team_val = cond_work.pop('team', None)

        # normalize dict keys to dataframe columns (tolerant mapping)
        if isinstance(cond_work, dict):
            def _norm(k: str) -> str:
                return ''.join(ch.lower() for ch in str(k) if ch.isalnum())
            col_map = { _norm(c): c for c in df.columns }
            corrected = {}
            for k, v in cond_work.items():
                nk = _norm(k)
                if nk in col_map:
                    corrected[col_map[nk]] = v
                else:
                    corrected[k] = v
            cond_work = corrected

        # build mask using parse.build_mask (handles None, dict, or callable)
        try:
            base_mask = pd.Series(True, index=df.index) if cond_work is None else _parse.build_mask(df, cond_work).reindex(df.index).fillna(False).astype(bool)
        except Exception as e:
            print('Warning: failed to apply condition filter:', e)
            base_mask = pd.Series(False, index=df.index)

        # if team specified, further restrict to rows where home or away matches
        if team_val is not None:
            tstr = str(team_val).strip()
            tid = None
            try:
                tid = int(tstr)
            except Exception:
                pass
            team_mask = pd.Series(False, index=df.index)
            if tid is not None:
                if 'home_id' in df.columns:
                    team_mask = team_mask | (df['home_id'].astype(str) == str(tid))
                if 'away_id' in df.columns:
                    team_mask = team_mask | (df['away_id'].astype(str) == str(tid))
            else:
                tupper = tstr.upper()
                if 'home_abb' in df.columns:
                    team_mask = team_mask | (df['home_abb'].astype(str).str.upper() == tupper)
                if 'away_abb' in df.columns:
                    team_mask = team_mask | (df['away_abb'].astype(str).str.upper() == tupper)
            final_mask = base_mask & team_mask
        else:
            final_mask = base_mask

        if int(final_mask.sum()) == 0:
            empty = df.iloc[0:0].copy()
            empty['xgs'] = float('nan')
            return empty, team_val
        return df.loc[final_mask].copy(), team_val

    def _predict_xgs(df_filtered: pd.DataFrame):
        """Load/train classifier if needed and predict xgs for df rows; returns (df_with_xgs, clf, meta).

        Meta is (final_feature_names, categorical_levels_map) to be reused by callers.
        """
        df = df_filtered
        if df.shape[0] == 0:
            return df, None, None

        need_predict = ('xgs' not in df.columns) or (df['xgs'].isna().all())
        if not need_predict:
            return df, None, None

        # get classifier (respect behavior, fallback to train on failure)
        try:
            clf, feature_names, cat_levels = fit_xgs.get_clf(model_path, behavior, csv_path=str(chosen_csv))
        except Exception as e:
            print('xgs_map: get_clf failed with', e, '— trying to train a new model')
            clf, feature_names, cat_levels = fit_xgs.get_clf(model_path, 'train', csv_path=str(chosen_csv))

        # Prepare the model DataFrame using canonical feature list
        features = ['distance', 'angle_deg', 'game_state', 'is_net_empty']
        df_model, final_feature_cols_game, cat_map_game = fit_xgs.clean_df_for_model(df.copy(), features, fixed_categorical_levels=cat_levels)

        # prefer classifier's expected features when available
        final_features = feature_names if feature_names is not None else final_feature_cols_game

        # ensure xgs column exists
        df['xgs'] = np.nan

        # predict probabilities when possible
        if clf is not None and df_model.shape[0] > 0 and final_features:
            try:
                X = df_model[final_features].values
                probs = clf.predict_proba(X)[:, 1]
                df.loc[df_model.index, 'xgs'] = probs
            except Exception:
                pass

        return df, clf, (final_features, cat_levels)

    def _orient_coordinates(df_in: pd.DataFrame, team_val_local):
        """Produce x_a/y_a columns for plotting according to orientation rules.

        This preserves previous logic while being slightly more compact.
        """
        df = df_in
        left_goal_x, right_goal_x = plot_mod.rink_goal_xs()

        def attacked_goal_x_for_row(team_id, home_id, home_def):
            # Returns the x-coordinate of the goal being attacked for the shooter
            try:
                if pd.isna(team_id) or pd.isna(home_id):
                    return right_goal_x
                if str(team_id) == str(home_id):
                    # shooter is home: they attack opposite of home's defended side
                    return right_goal_x if home_def == 'left' else (left_goal_x if home_def == 'right' else right_goal_x)
                else:
                    return left_goal_x if home_def == 'left' else (right_goal_x if home_def == 'right' else left_goal_x)
            except Exception:
                return right_goal_x

        df['x_a'] = df.get('x')
        df['y_a'] = df.get('y')

        # compute attacked_x for each row once
        attacked_x = df.apply(lambda r: attacked_goal_x_for_row(r.get('team_id'), r.get('home_id'), r.get('home_team_defending_side')), axis=1)

        if team_val_local is not None:
            tstr = str(team_val_local).strip()
            try:
                tid = int(tstr)
            except Exception:
                tid = None
            tupper = None if tid is not None else tstr.upper()

            def is_selected(row):
                try:
                    if tid is not None:
                        return str(row.get('team_id')) == str(tid)
                    shooter_id = row.get('team_id')
                    if pd.isna(shooter_id):
                        return False
                    if str(shooter_id) == str(row.get('home_id')) and row.get('home_abb') is not None:
                        return str(row.get('home_abb')).upper() == tupper
                    if str(shooter_id) == str(row.get('away_id')) and row.get('away_abb') is not None:
                        return str(row.get('away_abb')).upper() == tupper
                except Exception:
                    return False
                return False

            desired_goal = df.apply(lambda r: left_goal_x if is_selected(r) else right_goal_x, axis=1)
            mask_rotate = (attacked_x != desired_goal) & df['x'].notna() & df['y'].notna()
            df.loc[mask_rotate, ['x_a', 'y_a']] = -df.loc[mask_rotate, ['x', 'y']]

        elif orient_all_left:
            mask_rotate = (attacked_x == right_goal_x) & df['x'].notna() & df['y'].notna()
            df.loc[mask_rotate, ['x_a', 'y_a']] = -df.loc[mask_rotate, ['x', 'y']]

        return df

    # ------------------- Main flow -----------------------------------------
    chosen_csv = _locate_csv()
    print('xgs_map: loading CSV ->', chosen_csv)
    df_all = pd.read_csv(chosen_csv)

    # Apply condition and return filtered dataframe + team_val
    df_filtered, team_val = _apply_condition(df_all)
    if df_filtered.shape[0] == 0:
        print(f"Warning: condition {condition!r} (team={team_val!r}) matched 0 rows; producing an empty plot without training/loading model")
    else:
        print(f"Filtered season dataframe to {len(df_filtered)} events by condition {condition!r} team={team_val!r}")

    # Predict xgs only when needed and possible
    df_with_xgs, clf, clf_meta = _predict_xgs(df_filtered)

    # Orientation deprecation: plotting routines now decide orientation and
    # splitting (team vs not-team or home vs away). Do not perform an
    # explicit coordinate rotation here; leave adjusted coordinates to
    # `plot.plot_events` which will call `adjust_xy_for_homeaway` only when
    # needed. If the user requested the legacy `orient_all_left` behavior,
    # emit a DeprecationWarning and ignore it here (plotting may still
    # emulate it if requested via plotting options).
    import warnings
    if orient_all_left:
        warnings.warn("'orient_all_left' is deprecated in xgs_map and ignored; control orientation via plot.plot_events options.", DeprecationWarning)

    # Use df_with_xgs directly; plot.plot_events will compute x_a/y_a if missing.
    df_to_plot = df_with_xgs

    # prepare events to plot
    if events_to_plot is None:
        events_to_plot = ['xgs']

    # Compute timing and xG summary now so we can optionally display it on the plot.
    import timing
    try:
        timing_result = timing.demo_for_export(df_filtered, condition)
    except Exception as e:
        print('Warning: timing.demo_for_export failed:', e)
        timing_result = {'per_game': {}, 'aggregate': {'intersection_pooled_seconds': {'team': 0.0, 'other': 0.0}}}

    # Compute xG totals from df_with_xgs (sum of 'xgs' per group)
    team_xgs = 0.0
    other_xgs = 0.0
    try:
        xgs_series = pd.to_numeric(df_with_xgs.get('xgs', pd.Series([], dtype=float)), errors='coerce').fillna(0.0)
        if team_val is not None:
            # determine membership using home/away or ids
            def _is_team_row(r):
                try:
                    t = team_val
                    tid = int(t) if str(t).strip().isdigit() else None
                except Exception:
                    tid = None
                try:
                    if tid is not None:
                        return str(r.get('team_id')) == str(tid)
                    tupper = str(team_val).upper()
                    if r.get('home_abb') is not None and str(r.get('home_abb')).upper() == tupper:
                        return str(r.get('team_id')) == str(r.get('home_id'))
                    if r.get('away_abb') is not None and str(r.get('away_abb')).upper() == tupper:
                        return str(r.get('team_id')) == str(r.get('away_id'))
                except Exception:
                    return False
                return False
            mask = df_with_xgs.apply(_is_team_row, axis=1)
            team_xgs = float(xgs_series[mask].sum())
            other_xgs = float(xgs_series[~mask].sum())
        else:
            # legacy home/away split
            if 'home_id' in df_with_xgs.columns and 'team_id' in df_with_xgs.columns:
                mask = df_with_xgs['team_id'].astype(str) == df_with_xgs['home_id'].astype(str)
                team_xgs = float(xgs_series[mask].sum())
                other_xgs = float(xgs_series[~mask].sum())
            else:
                team_xgs = float(xgs_series[xgs_series.index].sum())
                other_xgs = 0.0
    except Exception:
        team_xgs = other_xgs = 0.0

    # extract seconds from timing_result aggregate
    try:
        agg = timing_result.get('aggregate', {}) if isinstance(timing_result, dict) else {}
        inter = agg.get('intersection_pooled_seconds', {}) if isinstance(agg, dict) else {}
        team_seconds = float(inter.get('team') or 0.0)
        other_seconds = float(inter.get('other') or 0.0)
    except Exception:
        team_seconds = other_seconds = 0.0

    team_xg_per60 = (team_xgs / team_seconds * 3600.0) if team_seconds > 0 else 0.0
    other_xg_per60 = (other_xgs / other_seconds * 3600.0) if other_seconds > 0 else 0.0

    summary_stats = {
        'team_xgs': team_xgs,
        'other_xgs': other_xgs,
        'team_seconds': team_seconds,
        'other_seconds': other_seconds,
        'team_xg_per60': team_xg_per60,
        'other_xg_per60': other_xg_per60,
    }

    print('xgs_map: calling plot.plot_events ->', out_path)

    # Decide heatmap split mode: if a team was specified, show team vs not_team;
    # otherwise fall back to legacy home vs away mode.
    heatmap_mode = 'team_not_team' if team_val is not None else 'home_away'

    # Call plot_events and handle both return shapes and the optional heatmap return
    if return_heatmaps:
        # For heatmap generation, ask plot_events to use the selected mode and pass team_val
        ret = plot_mod.plot_events(
            df_to_plot,
            events_to_plot=events_to_plot,
            out_path=out_path,
            return_heatmaps=True,
            heatmap_split_mode=heatmap_mode,
            team_for_heatmap=team_val,
            summary_stats=summary_stats,
        )
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
        # When not requesting heatmaps, still ensure plotting uses the same
        # mode so visuals match the heatmap logic: use heatmap_mode and team_val.
        ret = plot_mod.plot_events(
            df_to_plot,
            events_to_plot=events_to_plot,
            out_path=out_path,
            heatmap_split_mode=heatmap_mode,
            team_for_heatmap=team_val,
        )
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

    # Determine return structure: always return (out_path, heatmaps_or_None, filtered_df_or_None)
    ret_heat = heatmaps if return_heatmaps else None
    ret_df = df_filtered.copy() if ('df_filtered' in locals() and return_filtered_df) else None
    return out_path, ret_heat, ret_df


# ----------------- xG heatmap helpers (moved above the CLI so they are available)
# These helpers implement the next_steps plan: compute Gaussian-smoothed xG
# heatmaps and aggregate per-team maps for a season. They are intentionally
# simple and readable — we can optimize later (FFT convolution, parallelism).


def compute_xg_heatmap_from_df(
    df,
    grid_res: float = 1.0,
    sigma: float = 6.0,
    x_col: str = 'x_a',
    y_col: str = 'y_a',
    amp_col: str = 'xgs',
    rink_mask_fn=None,
    normalize_per60: bool = False,
    total_seconds: float = None,
    selected_team: Optional[object] = None,
    selected_role: str = 'team',
):
    """Compute an xG heatmap on a fixed rink grid from an events DataFrame.

    Returns (gx, gy, heat, total_xg, total_seconds_used)
    - gx: 1D array of x grid centers
    - gy: 1D array of y grid centers
    - heat: 2D array shape (len(gy), len(gx)) with summed xG (or xG/60 if normalized)
    """
    import numpy as np
    import pandas as pd
    from rink import rink_half_height_at_x

    if df is None or df.shape[0] == 0:
        # return empty grid centered on rink extents
        gx = np.arange(-100.0, 100.0 + grid_res, grid_res)
        gy = np.arange(-42.5, 42.5 + grid_res, grid_res)
        heat = np.zeros((len(gy), len(gx)), dtype=float)
        return gx, gy, heat, 0.0, 0.0

    # We'll compute adjusted coordinates (x_a,y_a) here so callers may pass
    # the full df and ask for team-specific subsets via selected_team/selected_role.
    # ensure coordinates exist (x_a/y_a) or compute them from raw x,y
    df_work = df.copy()
    if x_col not in df_work.columns or y_col not in df_work.columns:
        try:
            import plot as _plot
            df_work = _plot.adjust_xy_for_homeaway(df_work)
        except Exception:
            # fall back to raw x/y passthrough
            df_work[x_col] = df_work.get('x')
            df_work[y_col] = df_work.get('y')
    else:
        # ensure presence
        df_work[x_col] = df_work.get(x_col)
        df_work[y_col] = df_work.get(y_col)

    xs_all = pd.to_numeric(df_work.get(x_col, pd.Series([], dtype=float)), errors='coerce')
    ys_all = pd.to_numeric(df_work.get(y_col, pd.Series([], dtype=float)), errors='coerce')
    amps_all = pd.to_numeric(df_work.get(amp_col, pd.Series([], dtype=float)), errors='coerce').fillna(0.0)

    # Determine a boolean mask for the subset we want: if selected_team provided,
    # include only rows matching (selected_role == 'team') or the inverse for 'other'.
    def _is_selected_row(r, team_val):
        try:
            if team_val is None:
                return False
            tstr = str(team_val).strip()
            try:
                tid = int(tstr)
            except Exception:
                tid = None
            if tid is not None:
                return str(r.get('team_id')) == str(tid)
            tupper = tstr.upper()
            # compare abbreviations if present
            if r.get('home_abb') is not None and str(r.get('home_abb')).upper() == tupper and str(r.get('team_id')) == str(r.get('home_id')):
                return True
            if r.get('away_abb') is not None and str(r.get('away_abb')).upper() == tupper and str(r.get('team_id')) == str(r.get('away_id')):
                return True
            # fallback: compare team_id to home/away ids when abb not available
            if r.get('home_id') is not None and r.get('team_id') is not None and str(r.get('team_id')) == str(r.get('home_id')) and r.get('home_abb') is not None and str(r.get('home_abb')).upper() == tupper:
                return True
            return False
        except Exception:
            return False

    if selected_team is not None:
        sel_mask = df_work.apply(lambda r: _is_selected_row(r, selected_team), axis=1)
        if selected_role == 'team':
            use_mask = sel_mask
        else:
            use_mask = ~sel_mask
    else:
        use_mask = pd.Series(True, index=df_work.index)

    # as part of masking, censor out rows that have None or non-finite values in critical parts
    xs_temp = xs_all[use_mask]
    ys_temp = ys_all[use_mask]
    amps_temp = amps_all[use_mask]
    # build a validity mask (non-null and finite)
    try:
        valid_mask = xs_temp.notna() & ys_temp.notna() & amps_temp.notna()
        valid_mask &= xs_temp.apply(np.isfinite) & ys_temp.apply(np.isfinite) & amps_temp.apply(np.isfinite)
    except Exception:
        valid_mask = (~xs_temp.isna()) & (~ys_temp.isna()) & (~amps_temp.isna())
    xs = xs_temp[valid_mask]
    ys = ys_temp[valid_mask]
    amps = amps_temp[valid_mask]

    gx = np.arange(-100.0, 100.0 + grid_res, grid_res)
    gy = np.arange(-42.5, 42.5 + grid_res, grid_res)
    XX, YY = np.meshgrid(gx, gy)

    heat = np.zeros_like(XX, dtype=float)

    if xs.size == 0:
        total_xg = 0.0
    else:
        total_xg = float(amps.sum())
        two_sigma2 = 2.0 * (sigma ** 2)
        norm_factor = (grid_res ** 2) / (2.0 * np.pi * (sigma ** 2))
        # accumulate kernels per event (loop acceptable for typical shot counts)
        for xi, yi, ai in zip(xs, ys, amps):
            dx = XX - float(xi)
            dy = YY - float(yi)
            kern = ai * norm_factor * np.exp(-(dx * dx + dy * dy) / two_sigma2)
            heat += kern

    # mask outside rink: compute a boolean rink_mask and set outside cells to 0.0
    try:
        rink_mask = np.vectorize(rink_half_height_at_x)(XX) >= np.abs(YY)
        # where rink_mask is False, explicitly set heat to 0.0
        heat = np.where(rink_mask, heat, 0.0)
    except Exception:
        try:
            # ensure numeric even if mask fails
            heat = np.asarray(heat, dtype=float)
        except Exception:
            pass

    # Ensure heat is a numeric float array; replace any remaining NaNs with 0.0
    heat = np.asarray(heat, dtype=float)
    heat = np.nan_to_num(heat, nan=0.0)

    # Determine total_seconds_used: prefer explicit input, else try to infer
    total_seconds_used = None
    if total_seconds is not None:
        try:
            total_seconds_used = float(total_seconds)
        except Exception:
            total_seconds_used = None

    if total_seconds_used is None:
        # try to infer from a timing column present in the dataframe
        try:
            times = pd.to_numeric(df_work.get('total_time_elapsed_seconds', pd.Series(dtype=float)), errors='coerce').dropna()
            if len(times) >= 2:
                total_seconds_used = float(times.max() - times.min())
            else:
                total_seconds_used = 0.0
        except Exception:
            total_seconds_used = 0.0

    # If requested, normalize heat and total_xg to per-60 units using total_seconds_used
    if normalize_per60 and total_seconds_used and total_seconds_used > 0.0:
        try:
            scale = 3600.0 / float(total_seconds_used)
            heat = heat * scale
            total_xg = float(total_xg) * scale
        except Exception:
            # fall back to no scaling
            pass

    # Final safety: ensure no NaNs remain after scaling
    heat = np.nan_to_num(heat, nan=0.0)

    return gx, gy, heat, float(total_xg), float(total_seconds_used or 0.0)


def xg_maps_for_season(season_or_df, condition=None, grid_res: float = 1.0, sigma: float = 6.0, out_dir: str = 'static/league_vs_team_maps', min_events: int = 5, model_path: str = 'static/xg_model.joblib', behavior: str = 'load', csv_path: str = None):
    """Compute league and per-team xG maps for a season (or events DataFrame).

    Saves PNG and JSON summary per team into out_dir/{season}/
    Returns a dict with league_map info and per-team summaries.
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import json
    import matplotlib.pyplot as plt
    from rink import draw_rink, rink_half_height_at_x
    import parse as _parse

    # load season df or accept a provided DataFrame
    if isinstance(season_or_df, str):
        import timing
        df = timing.load_season_df(season_or_df)
        season = season_or_df
    else:
        df = season_or_df.copy()
        # try to infer season name
        season = getattr(df, 'season', None) or 'season'

    if df is None or df.shape[0] == 0:
        raise ValueError('No events data available for xg_maps_for_season')

    # apply condition filter using parse.build_mask when condition provided
    if condition is None:
        df_cond = df.copy()
    else:
        try:
            mask = _parse.build_mask(df, condition)
            mask = mask.reindex(df.index).fillna(False).astype(bool)
            df_cond = df.loc[mask].copy()
        except Exception:
            df_cond = df.copy()

    # ensure adjusted coords exist
    try:
        import plot as _plot
        df_cond = _plot.adjust_xy_for_homeaway(df_cond)
    except Exception:
        pass

    # If xgs are missing or all zeros, try to predict using the xG classifier
    try:
        need_predict = ('xgs' not in df_cond.columns) or (pd.to_numeric(df_cond.get('xgs', pd.Series(dtype=float)), errors='coerce').fillna(0.0).sum() == 0)
    except Exception:
        need_predict = True

    if need_predict:
        try:
            import fit_xgs
            # use get_or_train_clf which caches/trains if needed
            force_retrain = True if (behavior or '').strip().lower() == 'train' else False
            try:
                clf, feature_names, cat_levels = fit_xgs.get_or_train_clf(force_retrain=force_retrain, csv_path=csv_path)
            except Exception:
                # fallback to older get_clf
                clf, feature_names, cat_levels = fit_xgs.get_clf(model_path, behavior, csv_path=csv_path)

            features = ['distance', 'angle_deg', 'game_state', 'is_net_empty']
            df_model, final_feature_cols_game, cat_map_game = fit_xgs.clean_df_for_model(df_cond.copy(), features, fixed_categorical_levels=cat_levels)

            # ensure xgs column exists
            df_cond['xgs'] = pd.to_numeric(df_cond.get('xgs', pd.Series(dtype=float)), errors='coerce').fillna(0.0)

            final_features = feature_names if feature_names is not None else final_feature_cols_game
            if clf is not None and df_model.shape[0] > 0 and final_features:
                try:
                    X = df_model[final_features].values
                    probs = clf.predict_proba(X)[:, 1]
                    # Apply predictions back to the original df_cond by index
                    df_cond.loc[df_model.index, 'xgs'] = probs
                    print(f'xg_maps_for_season: predicted xgs for {len(probs)} events')
                except Exception as e:
                    print('xg_maps_for_season: prediction failed:', e)
        except Exception as e:
            print('xg_maps_for_season: failed to predict xgs:', e)

    def orient_all(df_in, target: str = 'left', selected_team: Optional[object] = None, selected_role: str = 'team'):
        """Return a copy of df_in with x_a/y_a rotated according to either a fixed target
        ('left' or 'right') or relative to a `selected_team`.

        If `selected_team` is provided, rows where the shooter matches
        `selected_team` will be oriented to the left and all other rows to the
        right. If `selected_team` is None, behavior falls back to the simpler
        `target`-based orientation (all left or all right).
        """
        import copy
        df2 = df_in.copy()
        try:
            from plot import rink_goal_xs
            left_goal_x, right_goal_x = rink_goal_xs()
        except Exception:
            # fallback numeric goal x positions
            left_goal_x, right_goal_x = -89.0, 89.0

        def attacked_goal_x_for_row(r):
            # replicate logic from plot.adjust_xy_for_homeaway
            try:
                team_id = r.get('team_id')
                home_id = r.get('home_id')
                home_def = r.get('home_team_defending_side')
                if pd.isna(team_id) or pd.isna(home_id):
                    return right_goal_x
                if str(team_id) == str(home_id):
                    # shooter is home: they attack opposite of home's defended side
                    if home_def == 'left':
                        return right_goal_x
                    elif home_def == 'right':
                        return left_goal_x
                    else:
                        return right_goal_x
                else:
                    if home_def == 'left':
                        return left_goal_x
                    elif home_def == 'right':
                        return right_goal_x
                    else:
                        return left_goal_x
            except Exception:
                return right_goal_x

        # compute attacked goal series
        attacked = df2.apply(attacked_goal_x_for_row, axis=1)

        # determine desired goal per-row
        if selected_team is not None:
            tstr = str(selected_team).strip()
            tid = None
            try:
                tid = int(tstr)
            except Exception:
                tid = None

            def is_selected(row):
                try:
                    if tid is not None:
                        return str(row.get('team_id')) == str(tid)
                    shooter_id = row.get('team_id')
                    if pd.isna(shooter_id):
                        return False
                    if row.get('home_id') is not None and str(row.get('home_id')) == str(row.get('team_id')):
                        # home team membership
                        if row.get('home_abb') is not None and str(row.get('home_abb')).upper() == tstr.upper():
                            return True
                    if row.get('away_id') is not None and str(row.get('away_id')) == str(row.get('team_id')):
                        if row.get('away_abb') is not None and str(row.get('away_abb')).upper() == tstr.upper():
                            return True
                    # fallback: compare abbreviations if present
                    if row.get('home_abb') is not None and str(row.get('home_abb')).upper() == tstr.upper() and str(row.get('team_id')) == str(row.get('home_id')):
                        return True
                    if row.get('away_abb') is not None and str(row.get('away_abb')).upper() == tstr.upper() and str(row.get('team_id')) == str(row.get('away_id')):
                        return True
                except Exception:
                    return False
                return False

            # selected_role determines which rows are treated as the "selected"
            # group. If selected_role == 'team', rows matching selected_team are
            # oriented to the left and others to the right. If
            # selected_role == 'other', the logic is flipped: rows matching
            # selected_team are oriented to the right and others to the left.
            if selected_role == 'team':
                desired = df2.apply(lambda r: left_goal_x if is_selected(r) else right_goal_x, axis=1)
            else:
                desired = df2.apply(lambda r: right_goal_x if is_selected(r) else left_goal_x, axis=1)
        else:
            desired = left_goal_x if target == 'left' else right_goal_x

        # produce adjusted coords
        xcol = 'x'
        ycol = 'y'
        df2['x_a'] = df2.get('x')
        df2['y_a'] = df2.get('y')
        mask = (attacked != desired) & df2[xcol].notna() & df2[ycol].notna()
        try:
            df2.loc[mask, ['x_a', 'y_a']] = -df2.loc[mask, [xcol, ycol]].values
        except Exception:
            # fallback loop if vectorized assignment fails
            for idx in df2.loc[mask].index:
                try:
                    df2.at[idx, 'x_a'] = -float(df2.at[idx, xcol])
                    df2.at[idx, 'y_a'] = -float(df2.at[idx, ycol])
                except Exception:
                    df2.at[idx, 'x_a'] = df2.at[idx, xcol]
                    df2.at[idx, 'y_a'] = df2.at[idx, ycol]
        return df2

    def rotate_heat_180(heat):
        import numpy as _np
        if heat is None:
            return None
        return _np.flipud(_np.fliplr(heat))

    # Derive total_seconds for normalization from timing.demo_for_export using the input condition.
    # This gives a more accurate denominator for normalize_per60 than inferring from observed timestamp ranges.
    try:
        import timing
        timing_res = timing.demo_for_export(df_cond, condition)
        agg = timing_res.get('aggregate', {}) if isinstance(timing_res, dict) else {}
        inter = agg.get('intersection_pooled_seconds', {}) if isinstance(agg, dict) else {}
        team_secs = float(inter.get('team') or 0.0)
        other_secs = float(inter.get('other') or 0.0)
        total_seconds_cond = float(team_secs + other_secs) if (team_secs or other_secs) else None
    except Exception:
        total_seconds_cond = None

    # compute league map with all shots oriented to the LEFT (so league baseline is offense-left)
    df_league_left = orient_all(df_cond, target='left')
    gx, gy, league_heat, league_xg, league_seconds = compute_xg_heatmap_from_df(
        df_league_left, grid_res=grid_res, sigma=sigma, normalize_per60=True, total_seconds=total_seconds_cond)

    # prepare output dir
    base_out = Path(out_dir) / str(season)
    base_out.mkdir(parents=True, exist_ok=True)

    # save league map figure
    try:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        draw_rink(ax=ax)
        extent = (gx[0] - grid_res / 2.0, gx[-1] + grid_res / 2.0, gy[0] - grid_res / 2.0, gy[-1] + grid_res / 2.0)
        cmap = plt.get_cmap('viridis')
        im = ax.imshow(league_heat, extent=extent, origin='lower', cmap=cmap, zorder=1, alpha=0.8)
        fig.colorbar(im, ax=ax, label='xG per hour (approx)')
        out_png = base_out / f'{season}_league_map.png'
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
    except Exception:
        pass

    # determine team list (use abbreviations when possible)
    teams = set()
    if 'home_abb' in df_cond.columns:
        teams.update([a for a in df_cond['home_abb'].dropna().astype(str).unique().tolist()])
    if 'away_abb' in df_cond.columns:
        teams.update([a for a in df_cond['away_abb'].dropna().astype(str).unique().tolist()])
    # fallback to team ids if no abbs
    if not teams and 'home_id' in df_cond.columns:
        teams.update([str(a) for a in df_cond['home_id'].dropna().unique().tolist()])

    results = {'season': season, 'league': {'gx': list(gx), 'gy': list(gy), 'xg_total': league_xg, 'seconds': league_seconds}}
    results['teams'] = {}

    for team in sorted(teams):
        # team membership mask
        try:
            t = str(team).strip().upper()
            mask_team = ((df_cond.get('home_abb', pd.Series(dtype=object)).astype(str).str.upper() == t) | (df_cond.get('away_abb', pd.Series(dtype=object)).astype(str).str.upper() == t))
        except Exception:
            mask_team = pd.Series(False, index=df_cond.index)

        df_team = df_cond.loc[mask_team]
        df_opp = df_cond.loc[~mask_team]
        n_events = int(df_team.shape[0])
        if n_events < min_events:
            # skip low-sample teams
            continue

        # Derive per-team timing seconds via timing.demo_for_export using the base condition + team
        team_secs_t = None
        other_secs_t = None
        try:
            import timing
            # build a per-team condition dict based on the original `condition`
            if isinstance(condition, dict):
                team_condition = condition.copy()
            else:
                team_condition = {} if condition is None else dict(condition)
            team_condition['team'] = team
            timing_res_team = timing.demo_for_export(df_cond, team_condition)
            agg_t = timing_res_team.get('aggregate', {}) if isinstance(timing_res_team, dict) else {}
            inter_t = agg_t.get('intersection_pooled_seconds', {}) if isinstance(agg_t, dict) else {}
            team_secs_t = float(inter_t.get('team') or 0.0)
            other_secs_t = float(inter_t.get('other') or 0.0)
        except Exception:
            # fall back to season-level total_seconds_cond
            team_secs_t = None
            other_secs_t = None

        # Prepare sensible fallbacks for normalization: prefer per-team seconds when present,
        # otherwise fall back to the season-level total_seconds_cond (if available), else None.
        team_total_seconds_for_heat = team_secs_t if team_secs_t and team_secs_t > 0 else (total_seconds_cond if 'total_seconds_cond' in locals() else None)
        opp_total_seconds_for_heat = other_secs_t if other_secs_t and other_secs_t > 0 else (total_seconds_cond if 'total_seconds_cond' in locals() else None)

        # Filter the season-level df to only games involving this team. This ensures
        # both heatmaps (team and opponents) are derived from the same set of games.
        games_for_team = []
        try:
            games_for_team = df_cond.loc[mask_team, 'game_id'].dropna().unique().tolist()
        except Exception:
            games_for_team = []

        if games_for_team:
            df_games = df_cond.loc[df_cond['game_id'].isin(games_for_team)].copy()
        else:
            # fallback: if no game ids, just use the per-team rows and their opponents
            df_games = pd.concat([df_team, df_opp], ignore_index=False).copy()

        # Orient the combined games such that selected team shots face LEFT and
        # opponents face RIGHT. This produces x_a/y_a coordinates suitable for
        # both subsequent heatmap computations.
        try:
            df_oriented = orient_all(df_games, target='left', selected_team=team, selected_role='team')
        except Exception:
            # last-resort: use df_games as-is
            df_oriented = df_games.copy()

        # Team heat: compute from oriented df but ask function to restrict to the selected team
        gx_t, gy_t, team_heat, team_xg, team_seconds = compute_xg_heatmap_from_df(
            df_oriented, grid_res=grid_res, sigma=sigma, normalize_per60=True,
            total_seconds=team_total_seconds_for_heat, selected_team=team, selected_role='team')

        # Opponent heat: compute from the same oriented df but restrict to opponents
        gx_o, gy_o, opp_heat, opp_xg, opp_seconds = compute_xg_heatmap_from_df(
            df_oriented, grid_res=grid_res, sigma=sigma, normalize_per60=True,
            total_seconds=opp_total_seconds_for_heat, selected_team=team, selected_role='other')

        import numpy as np
        eps = 1e-9

        # Determine blue-line x positions (same as rink.draw_rink uses: ±25.0)
        blue_x = 25.0
        left_blue_x = -blue_x
        right_blue_x = blue_x

        # Create x masks for grid columns: team offensive zone is gx <= left_blue_x; opponent offense is gx >= right_blue_x
        gx_arr = np.array(gx)
        mask_team_zone = gx_arr <= left_blue_x
        mask_opp_zone = gx_arr >= right_blue_x

        # Mask league heat to left and right zones accordingly
        league_left_zone = np.full_like(league_heat, np.nan)
        league_left_zone[:, mask_team_zone] = league_heat[:, mask_team_zone]

        league_right = rotate_heat_180(league_heat)
        league_right_zone = np.full_like(league_right, np.nan)
        league_right_zone[:, mask_opp_zone] = league_right[:, mask_opp_zone]

        # Mask team and opponent heat to offensive zones
        team_zone = np.full_like(team_heat, np.nan)
        team_zone[:, mask_team_zone] = team_heat[:, mask_team_zone]

        opp_zone = np.full_like(opp_heat, np.nan)
        opp_zone[:, mask_opp_zone] = opp_heat[:, mask_opp_zone]

        # Compute percent-difference maps only within masked zones (NaNs elsewhere)
        try:
            pct_team = (team_zone - league_left_zone) / (league_left_zone + eps) * 100.0
        except Exception:
            pct_team = np.full_like(team_zone, np.nan)

        try:
            pct_opp = (opp_zone - league_right_zone) / (league_right_zone + eps) * 100.0
        except Exception:
            pct_opp = np.full_like(opp_zone, np.nan)

        # Combined summary plot: single rink with team (left) and opponents (right)
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            # draw single rink
            draw_rink(ax=ax)

            # determine shared vmax from both pct arrays (ignore naNs)
            vals = np.hstack([
                pct_team[~np.isnan(pct_team)].ravel() if np.any(~np.isnan(pct_team)) else np.array([0.0]),
                pct_opp[~np.isnan(pct_opp)].ravel() if np.any(~np.isnan(pct_opp)) else np.array([0.0])
            ])
            if vals.size == 0:
                vmax = 1.0
            else:
                vmax = max(abs(np.nanmin(vals)), abs(np.nanmax(vals)), 1.0)

            cmap = plt.get_cmap('RdBu_r')
            # Render NaN (masked) cells as transparent so the rink beneath is visible
            try:
                cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
            except Exception:
                # fallback to white if transparency unsupported
                cmap.set_bad(color='white')

            # ensure `extent` is defined (may have been created earlier for league map)
            extent = (gx[0] - grid_res / 2.0, gx[-1] + grid_res / 2.0,
                      gy[0] - grid_res / 2.0, gy[-1] + grid_res / 2.0)

            # plot team (left zone) then opponents (right zone) on same axes; NaNs are transparent
            try:
                m_pct_team = np.ma.masked_invalid(pct_team)
                m_pct_opp = np.ma.masked_invalid(pct_opp)
            except Exception:
                m_pct_team = pct_team
                m_pct_opp = pct_opp
            im_team = ax.imshow(m_pct_team, extent=extent, origin='lower', cmap=cmap, vmin=-vmax, vmax=vmax, zorder=2)
            im_opp = ax.imshow(m_pct_opp, extent=extent, origin='lower', cmap=cmap, vmin=-vmax, vmax=vmax, zorder=2)

            # add small textual labels for clarity
            ax.text(-80, 40, f"{team} (offense → left)", fontsize=10, fontweight='bold', ha='left')
            ax.text(20, 40, f"Opponents (offense → right)", fontsize=10, fontweight='bold', ha='left')

            # shared colorbar
            cbar = fig.colorbar(im_opp, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('pct change vs league (%)')

            out_png_summary = base_out / f'{season}_{team}_summary.png'
            fig.tight_layout()
            fig.savefig(out_png_summary, dpi=150)
            plt.close(fig)
        except Exception as e:
            print('failed to create combined summary plot for', team, e)
            out_png_summary = None

        # save JSON summary
        summary = {
            'team': team,
            'n_events': n_events,
            'team_xg': team_xg,
            'team_seconds': team_seconds,
            'team_xg_per60': (team_xg / team_seconds * 3600.0) if team_seconds > 0 else None,
            'out_png_summary': str(out_png_summary) if out_png_summary is not None else None,
        }
        # write JSON
        try:
            out_json = base_out / f'{season}_{team}_summary.json'
            with open(out_json, 'w') as fh:
                json.dump(summary, fh, indent=2)
        except Exception:
            pass

        results['teams'][team] = summary

    # save a small league summary JSON
    try:
        out_league_json = base_out / f'{season}_league_summary.json'
        with open(out_league_json, 'w') as fh:
            json.dump(results['league'], fh, indent=2)
    except Exception:
        pass

    return results


if __name__ == '__main__':
    """Command-line example for running xgs_map for a season and a team.

    Usage examples (from project root):
      python analyze.py --season 20252026 --team PHI
      python analyze.py --season 20252026 --team PHI --out static/PHI_map.png --orient-all-left

    Notes:
      - This example constructs a minimal `condition` dictionary with the
        requested team (``{'team': TEAM}``) and passes it to xgs_map. The
        heavy lifting (CSV lookup, optional model loading/training, plotting)
        is performed by xgs_map.
      - If you prefer to run non-interactively from a script, call
        xgs_map(...) directly from Python, passing a `condition` dict.

    The parser below also supports a quick "run all teams" mode which calls
    `xg_maps_for_season`. That quick-run mode is intentionally simple and
    easy to edit: change the `condition` and parameters below to control what
    is computed for the full season.
    """

    import argparse

    parser = argparse.ArgumentParser(prog='analyze.py', description='Run xgs_map for a season and team (example).')
    parser.add_argument('--season', default='20252026', help='Season string (e.g. 20252026)')
    parser.add_argument('--team', required=False, help='Team abbreviation or id to filter (e.g. PHI)')
    parser.add_argument('--out', default='static/xg_map_example.png', help='Output image path')
    parser.add_argument('--orient-all-left', action='store_true', help='Orient all shots to the left')
    parser.add_argument('--behavior', choices=['load', 'train'], default='load', help='Classifier behavior for xg model (load or train)')
    # by default we return heatmaps; use --no-heatmaps to disable
    parser.add_argument('--no-heatmaps', dest='return_heatmaps', action='store_false', help='Do not return heatmaps (enabled by default)')
    parser.set_defaults(return_heatmaps=True)
    parser.add_argument('--csv-path', default=None, help='Optional explicit CSV path to use (overrides season search)')
    # NEW: quick-run flag to generate per-team xG pct maps for the whole season
    parser.add_argument('--run-all', action='store_true', help='Run '
                                                               'full-season '
                                                               'xG maps for '
                                                               'all teams ('
                                                               'simple '
                                                               'demo)',
                        default=True)
    args = parser.parse_args()

    # Default condition used by both single-game xgs_map and the full-season run.
    # Edit this block directly to change the condition used in the quick full-season run.
    condition = {'game_state': ['5v5'],
                 'is_net_empty': [0]}
    if args.team:
        condition['team'] = args.team

    print(f"Running xgs_map for season={args.season} team={args.team!r} out={args.out} behavior={args.behavior}")

    try:
        import parse

        # If --run-all was provided, run the season-level mapping across all teams
        if args.run_all:
            # Parameters for the full-season maps. Tune these for speed/quality:
            #  - grid_res: 1.0 (1 ft) is fine-quality; use 2.0/5.0 for faster runs
            #  - sigma: gaussian kernel width in feet (6.0 is a reasonable default)
            #  - min_events: skip teams with fewer than this many events
            grid_res = 1.0
            sigma = 6.0
            min_events = 20
            out_dir = 'static/league_vs_team_maps'

            print(f"Running full-season xG maps for season {args.season} with condition={condition}")
            results = xg_maps_for_season(args.season, condition=condition, grid_res=grid_res, sigma=sigma, out_dir=out_dir, min_events=min_events)
            print('Full-season xG maps completed. Results keys:', list(results.keys()))
            # Print a small summary for convenience
            try:
                print('League total xG:', results['league'].get('xg_total'))
            except Exception:
                pass
            raise SystemExit(0)

        # Otherwise run the existing one-game/one-season plotting example via xgs_map
        out_path, heatmaps, df_filtered = xgs_map(season=args.season,
                         csv_path=args.csv_path,
                         model_path='static/xg_model.joblib',
                         behavior=args.behavior,
                         out_path=args.out,
                         orient_all_left=args.orient_all_left,
                         events_to_plot=None,
                         show=False,
                         return_heatmaps=args.return_heatmaps,
                         condition=condition)
        print('xgs_map completed. out=', out_path)
        print('heatmaps:', bool(heatmaps))

        import timing
        # Pass the same `condition` dict used to filter the season into
        # `demo_for_export`. `demo_for_export` will internally derive the
        # analysis conditions from this `condition` (or fall back to defaults).
        timing_result = timing.demo_for_export(df_filtered, condition)

        # Print summary stats about xGs per 60 minutes
        def _safe_heat_sum(hm, key):
            try:
                if hm is None:
                    return 0.0
                if isinstance(hm, dict) and key in hm and hm[key] is not None:
                    import numpy as _np
                    return float(_np.nansum(hm[key]))
            except Exception:
                pass
            return 0.0

        # Heatmap keys may be ('team','not_team') or ('home','away') depending on split mode
        team_xgs = 0.0
        other_xgs = 0.0
        try:
            if isinstance(heatmaps, dict):
                if 'team' in heatmaps or 'not_team' in heatmaps:
                    team_xgs = _safe_heat_sum(heatmaps, 'team')
                    other_xgs = _safe_heat_sum(heatmaps, 'not_team')
                else:
                    team_xgs = _safe_heat_sum(heatmaps, 'home')
                    other_xgs = _safe_heat_sum(heatmaps, 'away')
        except Exception:
            team_xgs = other_xgs = 0.0

        # Extract intersection times from timing_result (seconds)
        try:
            agg = timing_result.get('aggregate', {}) if isinstance(timing_result, dict) else {}
            inter = agg.get('intersection_pooled_seconds', {}) if isinstance(agg, dict) else {}
            team_seconds = float(inter.get('team') or 0.0)
            other_seconds = float(inter.get('other') or 0.0)
        except Exception:
            team_seconds = other_seconds = 0.0

        # Compute xG per 60 minutes (xG/60)
        team_xg_per60 = (team_xgs / team_seconds * 3600.0) if team_seconds > 0 else 0.0
        other_xg_per60 = (other_xgs / other_seconds * 3600.0) if other_seconds > 0 else 0.0

        print(f"Summary: team xGS={team_xgs:.3f} over {team_seconds/60:.2f} min -> {team_xg_per60:.3f} xG/60", flush=True)
        print(f"Summary: other xGS={other_xgs:.3f} over {other_seconds/60:.2f} min -> {other_xg_per60:.3f} xG/60", flush=True)


        if df_filtered is not None:
            print('Filtered df:', df_filtered.shape)
            if df_filtered.shape[0] > 0:
                print(df_filtered[['x', 'y', 'xgs']].describe())
    except FileNotFoundError as fe:
        print('Error: could not find season CSV or required files:', fe)
    except SystemExit:
        # allow clean exit from run-all
        pass
    except Exception as e:
        import traceback as _tb
        print('xgs_map failed:', type(e).__name__, e)
        _tb.print_exc()
