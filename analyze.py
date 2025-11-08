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

        The `condition` contract is flexible: None, dict (column->spec), or
        callable(df)->boolean Series. If a team is specified in a dict via
        the 'team' key, it is treated specially and interpreted as matching
        either home or away (by id or abbreviation).
        """
        # Defensive copy of condition dict so caller's input isn't mutated
        cond_work = None
        if isinstance(condition, dict):
            cond_work = condition.copy()
        else:
            cond_work = condition

        team_val = None
        if isinstance(cond_work, dict) and 'team' in cond_work:
            try:
                team_val = cond_work.pop('team')
            except Exception:
                team_val = None

        # Build base mask using parse.build_mask (supports dict, callable, None)
        try:
            if cond_work is None:
                base_mask = pd.Series(True, index=df.index)
            else:
                base_mask = _parse.build_mask(df, cond_work).reindex(df.index).fillna(False).astype(bool)
        except Exception as e:
            print('Warning: failed to apply condition filter:', e)
            base_mask = pd.Series(False, index=df.index)

        # Build team mask if requested
        if team_val is not None:
            tstr = str(team_val).strip()
            try:
                tid = int(tstr)
            except Exception:
                tid = None

            team_mask = pd.Series(False, index=df.index)
            if tid is not None:
                if 'home_id' in df.columns:
                    team_mask |= df['home_id'].astype(str) == str(tid)
                if 'away_id' in df.columns:
                    team_mask |= df['away_id'].astype(str) == str(tid)
            else:
                tupper = tstr.upper()
                if 'home_abb' in df.columns:
                    team_mask |= df['home_abb'].astype(str).str.upper() == tupper
                if 'away_abb' in df.columns:
                    team_mask |= df['away_abb'].astype(str).str.upper() == tupper

            final_mask = base_mask & team_mask
        else:
            final_mask = base_mask

        n = int(final_mask.sum())
        if n == 0:
            # preserve prior behavior: return empty df with xgs column present
            empty = df.iloc[0:0].copy()
            empty['xgs'] = float('nan')
            return empty, team_val
        return df.loc[final_mask].copy(), team_val

    def _predict_xgs(df_filtered: pd.DataFrame):
        """Load/train classifier if needed and predict xgs for df rows; returns df with 'xgs'.

        Prediction is attempted only when df_filtered is non-empty and either
        (a) column 'xgs' doesn't exist, or (b) it exists but all values are NaN.
        This avoids re-predicting over already-populated xgs.
        """
        df = df_filtered
        if df.shape[0] == 0:
            return df, None, None

        # Decide whether we need to load/train a classifier
        need_predict = ('xgs' not in df.columns) or (df['xgs'].isna().all())
        clf = None
        final_features = None
        categorical_levels_map = None

        if not need_predict:
            # nothing to do
            return df, clf, (final_features, categorical_levels_map)

        # Attempt to obtain classifier according to requested behavior
        try:
            clf, final_features, categorical_levels_map = fit_xgs.get_clf(model_path, behavior, csv_path=str(chosen_csv))
        except Exception as e:
            # fallback: force training
            print('xgs_map: get_clf failed with', e, '— trying to train a new model')
            clf, final_features, categorical_levels_map = fit_xgs.get_clf(model_path, 'train', csv_path=str(chosen_csv))

        # Prepare model inputs using the same features used at training
        features = ['distance', 'angle_deg', 'game_state', 'is_net_empty']
        df_model, final_feature_cols_game, cat_map_game = fit_xgs.clean_df_for_model(df.copy(), features, fixed_categorical_levels=categorical_levels_map)

        if final_features is None:
            final_features = final_feature_cols_game

        # Ensure xgs column exists
        df['xgs'] = np.nan

        if clf is not None and df_model.shape[0] > 0:
            X = df_model[final_features].values
            probs = clf.predict_proba(X)[:, 1]
            xgs_series = pd.Series(probs, index=df_model.index)
            df.loc[xgs_series.index, 'xgs'] = xgs_series.values

        return df, clf, (final_features, categorical_levels_map)

    def _orient_coordinates(df_in: pd.DataFrame, team_val_local):
        """Produce x_a/y_a columns for plotting according to orientation rules.

        When `team_val_local` is supplied we force shots from that team to face
        the left goal and opponents to face the right goal. When `orient_all_left`
        is True (and no team is given), we orient all shots so they face left.
        """
        df = df_in
        left_goal_x, right_goal_x = plot_mod.rink_goal_xs()

        def compute_attacked_goal_x(row):
            try:
                team_id = row.get('team_id')
                home_id = row.get('home_id')
                home_def = row.get('home_team_defending_side')
                if pd.isna(team_id) or pd.isna(home_id):
                    return right_goal_x
                if str(team_id) == str(home_id):
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

        df['x_a'] = df.get('x')
        df['y_a'] = df.get('y')

        if team_val_local is not None:
            # compute identity of selected team
            tstr = str(team_val_local).strip()
            try:
                tid = int(tstr)
            except Exception:
                tid = None
            tupper = None if tid is not None else tstr.upper()

            attacked_xs = df.apply(compute_attacked_goal_x, axis=1)

            def is_row_selected(row):
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
                    return False
                except Exception:
                    return False

            desired_goal = df.apply(lambda r: left_goal_x if is_row_selected(r) else right_goal_x, axis=1)
            mask_rotate = (attacked_xs != desired_goal) & df['x'].notna() & df['y'].notna()
            df.loc[mask_rotate, 'x_a'] = -df.loc[mask_rotate, 'x']
            df.loc[mask_rotate, 'y_a'] = -df.loc[mask_rotate, 'y']

        elif orient_all_left:
            attacked_xs = df.apply(compute_attacked_goal_x, axis=1)
            mask_rotate = (attacked_xs == right_goal_x) & df['x'].notna() & df['y'].notna()
            df.loc[mask_rotate, 'x_a'] = -df.loc[mask_rotate, 'x']
            df.loc[mask_rotate, 'y_a'] = -df.loc[mask_rotate, 'y']

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
    import timing_brainstorming
    try:
        timing_result = timing_brainstorming.demo_for_export(df_filtered, condition)
    except Exception as e:
        print('Warning: timing_brainstorming.demo_for_export failed:', e)
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
    args = parser.parse_args()

    condition = {'game_state': ['5v5', '5v4', '4v5']}
    if args.team:
        condition['team'] = args.team

    print(f"Running xgs_map for season={args.season} team={args.team!r} out={args.out} behavior={args.behavior}")

    try:
        import parse


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

        import timing_brainstorming
        # Pass the same `condition` dict used to filter the season into
        # `demo_for_export`. `demo_for_export` will internally derive the
        # analysis conditions from this `condition` (or fall back to defaults).
        timing_result = timing_brainstorming.demo_for_export(df_filtered, condition)

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
    except Exception as e:
        import traceback as _tb
        print('xgs_map failed:', type(e).__name__, e)
        _tb.print_exc()
