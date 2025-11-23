## Various different analyses


from typing import Optional
import pandas as pd

# Import timing module with preference for timing_new
import sys
try:
    import timing_new as timing
    sys.modules['timing'] = timing
except ImportError:
    try:
        import timing
    except ImportError:
        timing = None

def _league(season: Optional[str] = '20252026',
            csv_path: Optional[str] = None,
            teams: Optional[list] = None):
    """
    Compute league-wide xG heatmaps by pooling all teams' left-side results.

    Optional `teams` allows limiting which teams to process (useful for testing).

    Behavior:
      - Loads the team list from static/teams.json.
      - Calls `xgs_map` per team (with the team set in the condition) and
        extracts the 'left' heatmap (team-facing) and the per-team seconds.
      - Sums the left heatmaps across teams and sums the left_seconds.
      - Normalizes the combined left heatmap to xG per 60 minutes by using the
        summed left_seconds as the denominator.
      - Saves the combined heatmap as a .npy and a PNG under static/.

    Returns:
      dict with keys:
        - combined_norm: 2D numpy array (normalized to xG per 60)
        - total_left_seconds: float
        - total_left_xg: float (integral of raw summed left heat)
        - stats: summary dict
    """
    import json
    import numpy as np
    import os
    import matplotlib.pyplot as plt

    # Load team list from static/teams.json unless explicit `teams` provided
    if teams is None:
        teams_path = os.path.join('static', 'teams.json')
        with open(teams_path, 'r') as f:
            teams_data = json.load(f)
        team_list = [t.get('abbr') for t in teams_data if 'abbr' in t]
    else:
        team_list = list(teams)

    condition = {'game_state': ['5v5'], 'is_net_empty': [0]}

    left_maps = []
    left_seconds = []
    pooled_output = {}

    for team in team_list:
        condition['team'] = team
        # call xgs_map robustly and unpack returned elements
        try:
            out_path, ret_heat, ret_df, summary_stats = xgs_map(
                season=season,
                csv_path=csv_path,
                model_path='static/xg_model.joblib',
                behavior='load',
                out_path=f'static/{team}_xg_map.png',
                orient_all_left=False,
                events_to_plot=None,
                show=False,
                return_heatmaps=True,
                condition=condition
            )
        except Exception as e:
            # skip team on failure but record error
            pooled_output[team] = {'error': str(e)}
            continue

        # unpack safely
        heatmaps = ret_heat
        df_filtered = ret_df

        # extract left heatmap (team-facing). Accept keys 'team' or 'home' as fallback
        left_h = None
        if isinstance(heatmaps, dict):
            # avoid using `or` on arrays (truth-value ambiguous)
            if heatmaps.get('team') is not None:
                left_h = heatmaps.get('team')
            elif heatmaps.get('home') is not None:
                left_h = heatmaps.get('home')
        elif heatmaps is not None:
            # if it's an array directly
            left_h = heatmaps

        # extract per-team seconds if available from summary_stats
        sec = None
        try:
            sec = float(summary_stats.get('team_seconds')) if isinstance(summary_stats, dict) and summary_stats.get('team_seconds') is not None else None
        except Exception:
            sec = None

        # fallback: try to estimate seconds from df_filtered using timing.demo_for_export
        if (sec is None or sec == 0.0) and df_filtered is not None:
            try:
                import timing
                t_res = timing.demo_for_export(df_filtered, {'team': team})
                agg = t_res.get('aggregate', {}) if isinstance(t_res, dict) else {}
                inter = agg.get('intersection_pooled_seconds', {}) if isinstance(agg, dict) else {}
                sec = float(inter.get('team') or 0.0)
            except Exception:
                sec = 0.0

        # record
        pooled_output[team] = {
            'left_map': left_h,
            'seconds': sec,
            'out_path': out_path,
            'df_filtered_shape': df_filtered.shape if df_filtered is not None else None,
            'summary_stats': summary_stats,
        }

        if left_h is not None:
            left_maps.append(left_h)
            left_seconds.append(float(sec or 0.0))

    # sum left heatmaps
    if left_maps:
        # ensure consistent shapes: find first non-None shape
        base_shape = None
        for h in left_maps:
            if h is not None:
                base_shape = np.array(h).shape
                break
        # re-stack with nan for missing cells
        aligned = []
        for h in left_maps:
            if h is None:
                aligned.append(np.full(base_shape, np.nan))
            else:
                arr = np.asarray(h, dtype=float)
                if arr.shape != base_shape:
                    # try to broadcast or resize: raise for now
                    raise ValueError(f'Incompatible heatmap shape for team: expected {base_shape}, got {arr.shape}')
                aligned.append(arr)
        left_sum = np.nansum(np.stack(aligned, axis=0), axis=0)
    else:
        left_sum = None

    total_left_seconds = float(sum(left_seconds)) if left_seconds else 0.0

    # Avoid divide-by-zero by using at least 1.0 second if zero (but warn)
    if total_left_seconds <= 0.0:
        total_left_seconds = 1.0

    # Normalize to xG per 60min
    combined_norm = None
    total_left_xg = 0.0
    if left_sum is not None:
        combined_norm = left_sum / total_left_seconds * 3600.0
        # total raw xG (integral of left_sum)
        total_left_xg = float(np.nansum(left_sum))

    # Save out combined heatmap to static
    try:
        out_dir = os.path.join('static')
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f'{season}_league_left_combined.npy'), combined_norm)
        # also save a PNG visual using masked array
        if combined_norm is not None:
            try:
                m = np.ma.masked_invalid(combined_norm)
                fig, ax = plt.subplots(figsize=(8, 4.5))
                from rink import draw_rink
                draw_rink(ax=ax)
                extent = None
                # try to infer extent from gx/gy in code that generated heatmaps (fallback to rink extents)
                gx = np.arange(-100.0, 100.0 + 1.0, 1.0)
                gy = np.arange(-42.5, 42.5 + 1.0, 1.0)
                extent = (gx[0] - 0.5, gx[-1] + 0.5, gy[0] - 0.5, gy[-1] + 0.5)
                cmap = plt.get_cmap('viridis')
                try:
                    cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
                except Exception:
                    pass
                ax.imshow(m, extent=extent, origin='lower', cmap=cmap)
                fig.savefig(os.path.join(out_dir, f'{season}_league_left_combined.png'), dpi=150)
                plt.close(fig)
            except Exception:
                pass
    except Exception:
        pass

    stats = {
        'season': season,
        'total_left_seconds': total_left_seconds,
        'total_left_xg': total_left_xg,
        'xg_per60': float(np.nansum(combined_norm)) if combined_norm is not None else 0.0,
        'n_teams': len(team_list),
    }

    return {'combined_norm': combined_norm, 'total_left_seconds': total_left_seconds, 'total_left_xg': total_left_xg, 'stats': stats, 'per_team': pooled_output}


def xgs_map(season: Optional[str] = '20252026', *,
            game_id: Optional[str] = None,

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
              condition: Optional[object] = None,
              # heatmap-only mode: compute and return heatmap arrays instead of plotting
              heatmap_only: bool = False,
              grid_res: float = 1.0,
              sigma: float = 6.0,
              normalize_per60: bool = False,
              selected_role: str = 'team', data_df: Optional['pd.DataFrame'] = None,
              # new interval filtering behavior
              use_intervals: bool = True,

              intervals_input: Optional[object] = None,
              interval_time_col: str = 'total_time_elapsed_seconds'):
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

    # Helper: apply a condition dict to a dataframe and return (filtered_df, team_val)
    def _apply_condition(df: pd.DataFrame):
        """Apply `condition` to df and return (filtered_df, team_val).

        Accepts None or a dict. If condition contains 'team' that key is used to
        filter rows where home or away matches that team (by abb or id).
        """
        cond_work = condition.copy() if isinstance(condition, dict) else condition

        team_val_local = None
        if isinstance(cond_work, dict) and 'team' in cond_work:
            team_val_local = cond_work.pop('team', None)

        # Normalize keys to column names where possible
        if isinstance(cond_work, dict):
            # Normalize key to alphanumeric lowercase
            col_map = {''.join(ch.lower() for ch in str(c) if ch.isalnum()): c for c in df.columns}
            corrected = {}
            for k, v in cond_work.items():
                nk = ''.join(ch.lower() for ch in str(k) if ch.isalnum())
                corrected[col_map.get(nk, k)] = v
            cond_work = corrected

        # Build mask via parse.build_mask when a dict is provided
        try:
            base_mask = pd.Series(True, index=df.index) if cond_work is None else _parse.build_mask(df, cond_work).reindex(df.index).fillna(False).astype(bool)
        except Exception as e:
            print('Warning: failed to apply condition filter:', e)
            base_mask = pd.Series(False, index=df.index)

        # Apply team filter if requested
        if team_val_local is not None:
            tstr = str(team_val_local).strip()
            # Try to parse as int ID, otherwise use as abbreviation
            try:
                tid = int(tstr)
                team_mask = pd.Series(False, index=df.index)
                if 'home_id' in df.columns:
                    team_mask |= df['home_id'].astype(str) == str(tid)
                if 'away_id' in df.columns:
                    team_mask |= df['away_id'].astype(str) == str(tid)
            except ValueError:
                tupper = tstr.upper()
                team_mask = pd.Series(False, index=df.index)
                if 'home_abb' in df.columns:
                    team_mask |= df['home_abb'].astype(str).str.upper() == tupper
                if 'away_abb' in df.columns:
                    team_mask |= df['away_abb'].astype(str).str.upper() == tupper
            final_mask = base_mask & team_mask
        else:
            final_mask = base_mask

        if int(final_mask.sum()) == 0:
            empty = df.iloc[0:0].copy()
            empty['xgs'] = float('nan')
            return empty, team_val_local
        return df.loc[final_mask].copy(), team_val_local

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
            # prefer explicit csv_path if provided; otherwise pass None when using data_df
            clf, feature_names, cat_levels = fit_xgs.get_clf(model_path, behavior, csv_path=csv_path or None)
        except Exception as e:
            print('xgs_map: get_clf failed with', e, '— trying to train a new model')
            clf, feature_names, cat_levels = fit_xgs.get_clf(model_path, 'train', csv_path=csv_path or None)

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
            df.loc[mask_rotate, ['x_a', 'y_a']] = -df.loc[mask_rotate, ['x', 'y']].values

        elif orient_all_left:
            mask_rotate = (attacked_x == right_goal_x) & df['x'].notna() & df['y'].notna()
            df.loc[mask_rotate, ['x_a', 'y_a']] = -df.loc[mask_rotate, ['x', 'y']].values

        return df

    def _apply_intervals(df_in: pd.DataFrame, intervals_obj, time_col: str = 'total_time_elapsed_seconds', team_val: Optional[object] = None, condition: Optional[dict] = None) -> pd.DataFrame:
        """
        Filters the input dataframe to include only rows where the event time falls within the specified intervals.

        Args:
            df_in (pd.DataFrame): The input dataframe containing event data.
            intervals_obj (dict): The intervals object containing per-game intersection intervals (as produced by timing.demo_for_export).
            time_col (str): The column name in df_in representing the event time.
            team_val (Optional[object]): The team identifier to extract team-specific intervals when appropriate.
            condition (Optional[dict]): Additional per-row conditions to enforce (e.g. {'game_state': ['5v5'], 'is_net_empty':[0]}).

        Returns:
            pd.DataFrame: A filtered dataframe containing only rows within the specified intervals and satisfying `condition` when provided.
        """
        import timing as _timing  # local import to avoid top-level circular deps

        filtered_rows = []
        skipped_games = []

        per_game = intervals_obj.get('per_game', {}) if isinstance(intervals_obj, dict) else {}

        # Iterate over each game_id in the intervals object
        for game_id, game_data in per_game.items():
            # normalize game id to string for robust matching with df values
            gid_str = str(game_id)

            # Determine which team perspective to use when evaluating game_state_relative_to_team
            team_for_game = team_val if team_val is not None else (game_data.get('selected_team') if isinstance(game_data, dict) else None)

            # Obtain intervals for this game in a few supported shapes
            team_intervals = []
            try:
                if isinstance(game_data, dict):
                    # preferred shape: game_data['sides']['team']['intersection_intervals']
                    sides = game_data.get('sides')
                    if isinstance(sides, dict):
                        team_side = sides.get('team') or {}
                        team_intervals = team_side.get('intersection_intervals') or team_side.get('pooled_intervals') or []
                    # fallback: top-level intersection_intervals
                    if not team_intervals:
                        team_intervals = game_data.get('intersection_intervals') or game_data.get('merged_intervals') or []
                    # another fallback: intervals_per_condition when only one condition present
                    if not team_intervals and isinstance(game_data.get('intervals_per_condition'), dict) and condition:
                        # try to pick the intersection across requested condition keys
                        ipc = game_data.get('intervals_per_condition')
                        # if the intervals_per_condition contains keys matching condition, merge them
                        candidate_lists = []
                        for k in (condition.keys() if isinstance(condition, dict) else []):
                            if k in ipc and isinstance(ipc.get(k), list):
                                candidate_lists.append(ipc.get(k))
                        if candidate_lists:
                            # intersect multiple lists
                            from math import isfinite
                            def _intersect_two_local(a,b):
                                res=[]
                                i=j=0
                                a_sorted=sorted(a)
                                b_sorted=sorted(b)
                                while i<len(a_sorted) and j<len(b_sorted):
                                    s1,e1=a_sorted[i]
                                    s2,e2=b_sorted[j]
                                    s=max(s1,s2); e=min(e1,e2)
                                    if e> s:
                                        res.append((s,e))
                                    if e1<e2:
                                        i+=1
                                    else:
                                        j+=1
                                return res
                            inter = candidate_lists[0]
                            for lst in candidate_lists[1:]:
                                inter = _intersect_two_local(inter, lst)
                            team_intervals = inter
                else:
                    # game_data might be a plain list of intervals
                    if isinstance(game_data, (list, tuple)):
                        team_intervals = list(game_data)
            except Exception:
                team_intervals = []

            # Debug print: show per-game summary to help trace empty-filtering
            try:
                rows_in_game = 0
                if 'game_id' in df_in.columns:
                    rows_in_game = int((df_in['game_id'].astype(str) == gid_str).sum())
                print(f"_apply_intervals: game {gid_str} rows_in_game={rows_in_game} parsed_intervals={len(team_intervals)} team_for_game={team_for_game}")
            except Exception:
                pass

            # If no intervals found for this game, skip it
            if not team_intervals:
                # try to continue -- nothing to apply for this game
                skipped_games.append((gid_str, 'no_intervals'))
                continue

            # Subset rows for this game (robust to int/str mismatch)
            try:
                if 'game_id' in df_in.columns:
                    df_game = df_in[df_in['game_id'].astype(str) == gid_str]
                else:
                    df_game = df_in.copy().iloc[0:0]
            except Exception:
                df_game = df_in[df_in.get('game_id') == game_id]

            try:
                print(f"_apply_intervals: game {gid_str} df_game_rows={0 if df_game is None else int(df_game.shape[0])}")
            except Exception:
                pass

            if df_game is None or df_game.empty:
                # no rows for this game in df_in
                skipped_games.append((gid_str, 'no_rows'))
                continue

            # If we need to test game_state, prepare a df with game_state_relative_to_team
            need_game_state = False
            gs_series = None
            if condition and isinstance(condition, dict) and 'game_state' in condition:
                need_game_state = True
                try:
                    # If timing module provides helper to add relative game state, use it
                    if hasattr(_timing, 'add_game_state_relative_column'):
                        df_game_rel = _timing.add_game_state_relative_column(df_game.copy(), team_for_game)
                        gs_series = df_game_rel.get('game_state_relative_to_team') if isinstance(df_game_rel, dict) is False else None
                        # if helper returned a DataFrame-like object, try attribute
                        if gs_series is None and isinstance(df_game_rel, pd.DataFrame):
                            gs_series = df_game_rel.get('game_state_relative_to_team')
                    else:
                        gs_series = None
                except Exception:
                    gs_series = None

            # Vectorized selection: collect indices of rows whose time_col falls into any interval
            try:
                matched_indices = []
                # coerce time column to numeric once for this game's rows
                # Ensure times is always a Series aligned with df_game.index
                if time_col in df_game.columns:
                    times = pd.to_numeric(df_game[time_col], errors='coerce')
                else:
                    import numpy as _np
                    times = pd.Series(_np.nan, index=df_game.index)
                for (start, end) in team_intervals:
                    try:
                        s = float(start); e = float(end)
                    except Exception:
                        continue
                    mask = times.notna() & (times >= s) & (times <= e)
                    if mask.any():
                        matched_indices.extend(df_game.loc[mask].index.tolist())
                # deduplicate while preserving order
                if matched_indices:
                    seen = set()
                    unique_idx = []
                    for ii in matched_indices:
                        if ii not in seen:
                            seen.add(ii); unique_idx.append(ii)
                    # append matched rows to filtered list by index reference
                    for ii in unique_idx:
                        try:
                            filtered_rows.append(df_game.loc[ii])
                        except Exception:
                            continue
                else:
                    skipped_games.append((gid_str, 'no_matches'))
            except Exception:
                skipped_games.append((gid_str, 'match_error'))

        # Create a new dataframe from the filtered rows
        if not filtered_rows:
            # Diagnostic output to help debugging why nothing matched
            try:
                print('\n_apply_intervals debug: no rows matched any intervals')
                print('intervals_obj per_game count =', len(per_game))
                # show up to 10 games summary
                i = 0
                for gk, gd in list(per_game.items())[:10]:
                    try:
                        # count intervals found by our parsing
                        s = None
                        sides = gd.get('sides') if isinstance(gd, dict) else None
                        if isinstance(sides, dict) and sides.get('team'):
                            team_section = sides.get('team') or {}
                            if isinstance(team_section, dict) and team_section.get('intersection_intervals'):
                                s = len(team_section.get('intersection_intervals') or [])
                            else:
                                s = len(team_section.get('pooled_intervals') or []) if isinstance(team_section, dict) else 0
                        elif isinstance(gd, dict) and gd.get('intersection_intervals'):
                            s = len(gd.get('intersection_intervals') or [])
                        else:
                            s = 0
                    except Exception:
                        s = '?'
                    print(' game', gk, 'intervals_parsed=', s)
                    i += 1
                print('skipped_games (sample):', skipped_games[:20])
                # print a little info about df_in shape/type for clarity
                try:
                    if df_in is None:
                        print('df_in is None')
                        cols = []
                    else:
                        try:
                            cols = list(df_in.columns)
                            print('df_in shape:', getattr(df_in, 'shape', None), 'columns_count=', len(cols))
                        except Exception:
                            cols = []
                            print('df_in present but columns not introspectable; type=', type(df_in))
                except Exception:
                    cols = []
            except Exception:
                cols = []
            # Return an empty DataFrame with the same columns as input when possible
            try:
                return pd.DataFrame(columns=cols)
            except Exception:
                return pd.DataFrame()

        return pd.DataFrame(filtered_rows, columns=df_in.columns)

    # ------------------- Main flow -----------------------------------------
    # Allow caller to specify a single game_id: fetch and parse that game's feed
    df_all = None
    chosen_csv = None
    if game_id is not None:
        try:
            import nhl_api as _nhl_api
            print(f"xgs_map: game_id provided ({game_id}) - fetching live feed...", flush=True)
            
            # Try fetching game feed (coerce to int or str as needed)
            feed = None
            for gid in [int(game_id), str(game_id)]:
                try:
                    feed = _nhl_api.get_game_feed(gid)
                    break
                except (ValueError, TypeError, Exception):
                    continue
            
            if feed:
                try:
                    # use parse helpers to build the events dataframe for the single game
                    ev_df = _parse._game(feed)
                    if ev_df is not None and not ev_df.empty:
                        try:
                            df_game = _parse._elaborate(ev_df)
                        except Exception:
                            df_game = ev_df.copy()
                        df_all = df_game.copy()
                        print(f"xgs_map: loaded {len(df_all)} event rows for game {game_id}", flush=True)
                    else:
                        print(f"xgs_map: parsed feed but got empty events for game {game_id}", flush=True)
                except Exception as e:
                    print('xgs_map: parse of live feed failed:', e, flush=True)
        except Exception as e:
            print('xgs_map: failed to fetch live feed for game_id', game_id, e, flush=True)

        # If we loaded a single game's DataFrame and condition doesn't specify a team, infer the home team
        if df_all is not None and not df_all.empty:
            try:
                if condition is None or not isinstance(condition, dict):
                    condition = {} if condition is None else dict(condition)
                if 'team' not in condition:
                    home_abb = None
                    home_id = None
                    if 'home_abb' in df_all.columns:
                        try:
                            home_abb = df_all['home_abb'].dropna().unique().tolist()[0]
                        except Exception:
                            home_abb = None
                    if 'home_id' in df_all.columns and home_abb is None:
                        try:
                            home_id = df_all['home_id'].dropna().unique().tolist()[0]
                        except Exception:
                            home_id = None
                    if home_abb:
                        condition['team'] = home_abb
                        print(f"xgs_map: inferred team='{home_abb}' from live game feed and set it in condition", flush=True)
                    elif home_id is not None:
                        condition['team'] = home_id
                        print(f"xgs_map: inferred team id={home_id} from live game feed and set it in condition", flush=True)
            except Exception:
                pass

    # If no single-game feed requested or fetch failed, fall back to provided DataFrame or CSV
    if df_all is None:
        if data_df is not None:
            df_all = data_df.copy()
            chosen_csv = None
            print('xgs_map: using provided DataFrame (in-memory) -> rows=', len(df_all))
        else:
            chosen_csv = _locate_csv()
            print('xgs_map: loading CSV ->', chosen_csv)
            df_all = pd.read_csv(chosen_csv)

    # --- Single timing call: call timing.demo_for_export once on the full dataset
    timing_full = {'per_game': {}, 'aggregate': {'intersection_pooled_seconds': {'team': 0.0, 'other': 0.0}}}
    if timing is not None:
        try:
            timing_full = timing.demo_for_export(df_all, condition)
        except Exception as e:
            print(f'Warning: timing.demo_for_export failed: {e}; using empty timing structure')

    # Apply filtering: either by condition or by intervals
    if use_intervals:
        intervals = intervals_input if intervals_input is not None else timing_full
        team_param = None
        if isinstance(condition, dict):
            team_param = condition.get('team')
        try:
            # debug: print a summary of the intervals object
            try:
                if isinstance(intervals, dict):
                    per_games_count = len(intervals.get('per_game', {}))
                else:
                    per_games_count = len(intervals) if hasattr(intervals, '__len__') else 0
                print(f"_apply_intervals: intervals per_game count={per_games_count}")
            except Exception:
                pass

            df_filtered = _apply_intervals(df_all, intervals, time_col=interval_time_col, team_val=team_param, condition=condition)
            if df_filtered is None:
                print('_apply_intervals returned None; falling back to empty DataFrame')
                df_filtered = pd.DataFrame(columns=df_all.columns if df_all is not None else [])
        except Exception as e:
            print('Exception while applying intervals:', type(e).__name__, e)
            # fallback to empty dataframe
            df_filtered = pd.DataFrame(columns=df_all.columns if df_all is not None else [])
        team_val = team_param
    else:
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
    # Add orientation step here to ensure x_a/y_a are present and correct for plotting
    df_to_plot = _orient_coordinates(df_with_xgs, team_val)


    # Compute timing and xG summary now so we can optionally display it on the plot.
    # Use the single timing result collected earlier (timing_full) as the canonical timing_result
    timing_result = timing_full

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
        inter = agg.get('intersection_seconds_total', {}) if isinstance(agg,
                                                               dict) else {}
        team_seconds = float(inter or 0.0)
        other_seconds = team_seconds
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

    # Decide heatmap split mode:
    # - If a team is specified, use 'team_not_team'.
    # - If a game is specified (but not a team), use 'home_away'.
    # - If neither team nor game is specified, use 'orient_all_left'.
    if team_val is not None:
        heatmap_mode = 'team_not_team'
    elif condition is not None and isinstance(condition, dict) and 'game_id' in condition:
        heatmap_mode = 'home_away'
    else:
        heatmap_mode = 'orient_all_left'


    # Call plot_events and handle both return shapes and the optional heatmap return
    if return_heatmaps:
        # For heatmap generation, ask plot_events to use the selected mode and pass team_val
        ret = plot_mod.plot_events(
            df_to_plot,
            out_path=out_path,
            return_heatmaps=True,
            heatmap_split_mode=heatmap_mode,
            team_for_heatmap=team_val,
            summary_stats=summary_stats,
            events_to_plot=['shot-on-goal', 'goal',
                            'blocked-shot', 'missed-shot',
                            'xGs'],
        )
        # Fix for tuple index out of range
        # Ensure the `ret` object has enough elements before unpacking
        if len(ret) >= 3:
            fig, ax, heatmaps = ret[0], ret[1], ret[2]
        else:
            raise ValueError("Expected at least 3 elements in 'ret', but got fewer.")
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
    return out_path, ret_heat, ret_df, summary_stats


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
    # mask outside rink: compute a boolean rink_mask and set outside cells to NaN
    try:
        rink_mask = np.vectorize(rink_half_height_at_x)(XX) >= np.abs(YY)
        # where rink_mask is False, explicitly set heat to NaN so it will be
        # treated as masked/transparent by plotting routines that mask NaNs.
        heat = np.where(rink_mask, heat, np.nan)
    except Exception:
        # ensure numeric array at minimum
        try:
            heat = np.asarray(heat, dtype=float)
        except Exception:
            pass

    # Ensure heat is a numeric float array; keep NaNs as NaN (do not replace)
    try:
        heat = np.asarray(heat, dtype=float)
    except Exception:
        pass

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

    # Final safety: convert non-numeric entries to float but keep NaNs so
    # plotting routines can mask them (do not replace NaNs with zeros).
    try:
        heat = np.asarray(heat, dtype=float)
    except Exception:
        pass

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
    from rink import draw_rink
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
        if 'xgs' in df_cond.columns:
            xgs_series = pd.to_numeric(df_cond['xgs'], errors='coerce').fillna(0.0)
        else:
            xgs_series = pd.Series(0.0, index=df_cond.index)

        need_predict = ('xgs' not in df_cond.columns) or (xgs_series.sum() == 0)
        df_cond['xgs'] = xgs_series
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
        `selected_team` will be oriented to the left and all other rows to the right. If `selected_team` is None, behavior falls back to the simpler
        `target`-based orientation (all left or all right).
        """
        df2 = df_in.copy()

        # determine goal x positions (try to use helper from plot/rink if available)
        try:
            from plot import rink_goal_xs
            left_goal_x, right_goal_x = rink_goal_xs()
        except Exception:
            # sensible defaults (distance from center to goal line)
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
        # ensure NaNs are masked so outer rink cells render transparent
        try:
            cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
        except Exception:
            try:
                cmap.set_bad(color='white')
            except Exception:
                pass
        try:
            m_league = np.ma.masked_invalid(league_heat)
            im = ax.imshow(m_league, extent=extent, origin='lower', cmap=cmap, zorder=1, alpha=0.8)
        except Exception:
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
            # Reserve space above the rink so summary text sits above and does not overlap
            try:
                fig.subplots_adjust(top=0.78)
            except Exception:
                pass
            # draw single rink
            draw_rink(ax=ax)

            # Fixed colorbar range for percent-change maps across all teams.
            # Base range is ±100 (%), extend it by 25% to give some headroom.
            colorbar_base = 100.0
            colorbar_extension = 1.25
            vmin = -colorbar_base * colorbar_extension
            vmax = colorbar_base * colorbar_extension

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
            im_team = ax.imshow(m_pct_team, extent=extent, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, zorder=2)
            im_opp = ax.imshow(m_pct_opp, extent=extent, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, zorder=2)

            # Structured title and two-column stats placed above the rink.
            try:
                cond_desc = str(condition) if condition is not None else 'All'
            except Exception:
                cond_desc = 'All'

            main_title = f"{team} — {cond_desc}"

            # Layout positions for title/subtitle/stats in figure coordinates
            title_y = 0.94
            subtitle_y = 0.90
            stats_y_start = 0.86
            line_gap = 0.03

            # Derive numeric summaries used in the top text block. These
            # variables are computed here to avoid unresolved-reference errors
            # while keeping the calculations compact and robust to missing data.
            # Prefer per-team computed values (team_seconds / opp_seconds), then
            # fall back to earlier aggregate values if unavailable.
            # Derive numeric summaries used in the top text block. Prefer per-team values
            try:
                if 'team_seconds' in locals() and team_seconds is not None:
                    t_secs = float(team_seconds or 0.0)
                elif 'team_secs_t' in locals() and team_secs_t is not None:
                    t_secs = float(team_secs_t or 0.0)
                elif 'total_seconds_cond' in locals() and total_seconds_cond is not None:
                    t_secs = float(total_seconds_cond or 0.0)
                else:
                    t_secs = 0.0
            except Exception:
                t_secs = 0.0

            try:
                if 'opp_seconds' in locals() and opp_seconds is not None:
                    o_secs = float(opp_seconds or 0.0)
                elif 'other_secs_t' in locals() and other_secs_t is not None:
                    o_secs = float(other_secs_t or 0.0)
                elif 'total_seconds_cond' in locals() and total_seconds_cond is not None:
                    o_secs = float(total_seconds_cond or 0.0)
                else:
                    o_secs = 0.0
            except Exception:
                o_secs = 0.0

            t_min = t_secs / 60.0 if t_secs > 0 else 0.0
            o_min = o_secs / 60.0 if o_secs > 0 else 0.0

            # compute per-60 rates: when normalize_per60=True was used earlier,
            # the returned team_xg/opp_xg already represent per-60 values.
            try:
                t_xg_per60 = float(team_xg or 0.0)
            except Exception:
                t_xg_per60 = 0.0
            try:
                o_xg_per60 = float(opp_xg or 0.0)
            except Exception:
                o_xg_per60 = 0.0

            # total xG over the interval (derived from per-60 rate)
            try:
                t_xg_total = t_xg_per60 * (t_secs / 3600.0)
            except Exception:
                t_xg_total = 0.0
            try:
                o_xg_total = o_xg_per60 * (o_secs / 3600.0)
            except Exception:
                o_xg_total = 0.0

            # vs-league difference in per-60 units (team - league)
            try:
                league_per60 = float(league_xg or 0.0)
            except Exception:
                league_per60 = 0.0
            try:
                t_vs_league = t_xg_per60 - league_per60
            except Exception:
                t_vs_league = 0.0
            try:
                o_vs_league = o_xg_per60 - league_per60
            except Exception:
                o_vs_league = 0.0

            # Title centered
            fig.text(0.5, title_y, main_title, fontsize=12, fontweight='bold', ha='center')
            # Subtitles for left/right (positioned in figure coords)
            fig.text(0.25, subtitle_y, 'Offense', fontsize=10, fontweight='semibold', ha='center')
            fig.text(0.75, subtitle_y, 'Defense', fontsize=10, fontweight='semibold', ha='center')

            # Build the lines of summary text for the left (team) and right (opponent)
            left_lines = [
                f"Time: {t_min:.1f} min",
                f"xG: {t_xg_total:.2f}",
                f"xG/60: {t_xg_per60:.3f}",
                f"vs league: {t_vs_league:+.3f} xG/60",
            ]
            right_lines = [
                f"Time: {o_min:.1f} min",
                f"xG: {o_xg_total:.2f}",
                f"xG/60: {o_xg_per60:.3f}",
                f"vs league: {o_vs_league:+.3f} xG/60",
            ]

            for i, (l, r) in enumerate(zip(left_lines, right_lines)):
                y = stats_y_start - i * line_gap
                fig.text(0.25, y, l, fontsize=9, fontweight='bold' if i == 0 else 'normal', ha='center')
                fig.text(0.75, y, r, fontsize=9, fontweight='bold' if i == 0 else 'normal', ha='center')
            # shared colorbar
            cbar = fig.colorbar(im_opp, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('pct change vs league (%)')

            out_png_summary = base_out / f'{season}_{team}_summary.png'
            # Do not call tight_layout here; we intentionally reserved top margin
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
    parser.add_argument('--run-all', action='store_true', help='Run full-season xG maps for all teams')
    
    args = parser.parse_args()
    
    # Execute based on arguments - only runs when called as main script
    if args.run_all:
        condition = {'game_state': ['5v5'], 'is_net_empty': [0]}
        xg_maps_for_season(args.season, condition=condition, behavior=args.behavior, csv_path=args.csv_path)
    elif args.team:
        condition = {'game_state': ['5v5'], 'is_net_empty': [0], 'team': args.team}
        xgs_map(season=args.season, condition=condition, behavior=args.behavior, 
                out_path=args.out, orient_all_left=args.orient_all_left,
                return_heatmaps=args.return_heatmaps, csv_path=args.csv_path)
    else:
        # Example: run league routine
        _league(season=args.season, csv_path=args.csv_path)

