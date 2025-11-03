"""fit_xgs.py

Simple, well-documented script to fit a lightweight xG-like model (shot -> goal
probability) using one primary feature (distance to goal). The goal is clarity
and a good starting point for extension.

Usage (from project root):
    python fit_xgs.py

The script will read 'static/20252026.csv' by default, train a Random Forest on
`dist_center` -> `is_goal`, print evaluation metrics, and save a calibration
plot to 'static/xg_likelihood.png'.
"""

# Keep the implementation intentionally simple and readable.

# typing imports intentionally minimal; Optional not required here

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from pathlib import Path

# Optional imports that may not be in the minimal requirements; we don't hard-fail
# at import time so the script can be inspected even if sklearn isn't available.
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        log_loss,
        roc_auc_score,
        brier_score_loss,
    )
    from sklearn.calibration import calibration_curve
except Exception:  # pragma: no cover - graceful fallback for environment missing sklearn
    RandomForestClassifier = None
    train_test_split = None
    accuracy_score = log_loss = roc_auc_score = brier_score_loss = None
    calibration_curve = None

def clean_df_for_model(df: pd.DataFrame, feature_cols, fixed_categorical_levels: dict = None):
    """Filter events, encode categorical features as integer codes, and
    coerce numeric types. Returns (df_clean, final_feature_cols, categorical_map).

    - df: filtered and preprocessed DataFrame
    - final_feature_cols: list of feature column names present in the returned df
    - categorical_map: mapping categorical input -> list of generated code column(s)
    """
    shot_attempt_types = ['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot']
    df = df[df['event'].isin(shot_attempt_types)].copy()

    # define is_goal as a boolean: True when event equals 'goal'
    df['is_goal'] = df['event'].eq('goal')

    # Determine which requested features are categorical (object dtype)
    categorical_cols = [col for col in feature_cols if col in df.columns and df[col].dtype == object]
    categorical_dummies_map = {}
    final_feature_cols = list(feature_cols)

    # Encode categorical columns as integer codes (one column per category).
    # If `fixed_categorical_levels` is provided (mapping cat -> list of levels),
    # we will use that to ensure encoding is compatible with training.
    categorical_levels_map = {}
    if categorical_cols:
        df, categorical_dummies_map, categorical_levels_map = one_hot_encode(
            df, categorical_cols, prefix_sep='_', fill_value='', fixed_mappings=fixed_categorical_levels
        )
        # Replace categorical names in final_feature_cols with the generated code column names
        for cat in categorical_cols:
            new_cols = categorical_dummies_map.get(cat, [])
            if cat in final_feature_cols:
                idx = final_feature_cols.index(cat)
                # splice in new_cols
                final_feature_cols = final_feature_cols[:idx] + new_cols + final_feature_cols[idx+1:]
            else:
                final_feature_cols.extend(new_cols)

    # Coerce final feature columns to numeric where possible
    numeric_cols = [c for c in final_feature_cols]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Ensure is_goal is integer 0/1
    df['is_goal'] = pd.to_numeric(df['is_goal'], errors='coerce').astype('Int64')

    # Drop rows with missing required values
    df = df[final_feature_cols + ['is_goal']].dropna().copy()
    df['is_goal'] = df['is_goal'].astype(int)

    return df, final_feature_cols, categorical_levels_map

def one_hot_encode(df: pd.DataFrame, categorical_cols, prefix_sep: str = '_', fill_value: str = '', fixed_mappings: dict = None):
    """Encode categorical columns as integer codes instead of binary dummies.

    This function keeps the same return shape as before: (df_out, categorical_dummies_map)
    where categorical_dummies_map maps categorical column name -> list of newly created
    column names. For integer encoding we create a single column per categorical
    input named '{col}_code' and return that in the list.
    """
    if isinstance(categorical_cols, str):
        categorical_cols = [categorical_cols]
    # ensure requested columns are present; if missing, create empty string column
    for c in categorical_cols:
        if c not in df.columns:
            df[c] = fill_value

    categorical_dummies_map = {}
    categorical_levels_map = {}

    for cat in categorical_cols:
        ser = df[cat].fillna(fill_value).astype(str)
        if fixed_mappings and cat in fixed_mappings and fixed_mappings[cat] is not None:
            # Use the provided training levels. Unknown values become '__other__'
            allowed = [str(x) for x in fixed_mappings[cat]]
            ser_clean = ser.where(ser.isin(allowed), other='__other__')
            levels = allowed + ['__other__']
            cat_obj = pd.Categorical(ser_clean, categories=levels)
        else:
            # build levels from the data (sorted for determinism)
            levels = sorted(ser.unique().tolist())
            cat_obj = pd.Categorical(ser, categories=levels)

        codes = cat_obj.codes
        new_col = f"{cat}_code"
        df[new_col] = codes
        df = df.drop(columns=[cat])
        categorical_dummies_map[cat] = [new_col]
        # store the levels used for this categorical column (excluding any placeholder)
        categorical_levels_map[cat] = [lv for lv in levels if lv is not None]

    return df, categorical_dummies_map, categorical_levels_map


def load_data(path: str = 'static/20252026.csv'):
    """Load the season CSV and return a cleaned DataFrame.

    Parameters
    - path: CSV path (default 'static/20252026.csv')
    - feature_cols: a column name or list of column names to use as features
      (default ['dist_center', 'angle_deg', 'game_state', 'is_net_empty']). The function will try common
      alternate names if the requested columns are missing.

    Returns a DataFrame with at least the feature column and `is_goal` target.
    The function will drop rows missing the required fields.
    """
    df = pd.read_csv(path)


    return df


def fit_model(
    df: pd.DataFrame,
    feature_cols=None,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
):
    """Fit a RandomForest on the specified feature and return the trained
    model along with the held-out test split.

    Returns (clf, X_test, y_test).
    """
    if RandomForestClassifier is None:
        raise RuntimeError('scikit-learn is required to run training. Please install scikit-learn.')

    # default feature set if not provided
    if feature_cols is None:
        feature_cols = ['distance', 'angle_deg', 'game_state', 'is_net_empty']
    if isinstance(feature_cols, str):
        feature_cols = [feature_cols]

    X = df[feature_cols].values
    y = df['is_goal'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=(y if len(np.unique(y)) > 1 else None),
    )

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)

    return clf, X_test, y_test


def evaluate_model(clf, X_test, y_test):
    """Given a fitted classifier and test split, compute probabilities and a
    small set of evaluation metrics. Returns (y_prob, y_pred, metrics).
    """
    # predicted probability of positive class
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Defensive metric helpers
    def _safe_call(func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TypeError:
            try:
                return func(*args)
            except Exception:
                return float('nan')
        except Exception:
            return float('nan')

    accuracy = _safe_call(accuracy_score, y_test, y_pred) if accuracy_score is not None else float('nan')

    logloss = float('nan')
    if log_loss is not None:
        try:
            logloss = log_loss(y_test, y_prob, eps=1e-15)
        except TypeError:
            try:
                logloss = log_loss(y_test, y_prob)
            except Exception:
                logloss = float('nan')

    rocauc = float('nan')
    if roc_auc_score is not None:
        try:
            rocauc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else float('nan')
        except Exception:
            rocauc = float('nan')

    brier = _safe_call(brier_score_loss, y_test, y_prob) if brier_score_loss is not None else float('nan')

    plot_calibration(y_test, y_prob, path='static/xg_likelihood.png',
                     n_bins= 10)

    metrics = {
        'accuracy': accuracy,
        'log_loss': logloss,
        'roc_auc': rocauc,
        'brier': brier,
    }

    return y_prob, y_pred, metrics





def plot_calibration(y_test, y_prob, path: str = 'static/xg_likelihood.png', n_bins: int = 10):
    """Generate and save a simple calibration/reliability plot.

    The plot shows observed frequency vs predicted probability in bins, and a
    diagonal reference line.
    """
    if calibration_curve is None:
        raise RuntimeError('scikit-learn is required to create calibration plots.')

    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=n_bins, strategy='uniform')

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    plt.xlabel('Predicted probability')
    plt.ylabel('Observed frequency')
    plt.title('Calibration curve — xG simple model')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()




def debug_model(clf, feature_cols=None, goal_side: str = 'left',
                    x_res: float = 2.0, y_res: float = 2.0,
                    out_path: str = 'static/xg_heatmap.png', cmap='viridis',
                    alpha: float = 0.8, verbose: bool = True,
                    game_state_values=None, is_net_empty_values=None,
                    categorical_levels_map: dict = None,
                    fixed_category_values: dict = None):
    """Simulate shots across the rink, predict xG using clf, and plot heatmaps for
    multiple combinations of non-location features.

    Parameters
    - clf: trained classifier with `predict_proba`
    - feature_cols: list of feature names (for example ['distance','angle_deg','game_state_code','is_net_empty'])
    - goal_side: 'left' or 'right' — which goal the shots are directed at
    - x_res, y_res: grid resolution in feet
    - out_path: base path to save images; the function will append suffixes for combos
    - cmap, alpha: retained for compatibility (not used when producing RGBA overlays)
    - game_state_values: iterable of game_state values to evaluate (strings)
    - is_net_empty_values: iterable of values to evaluate (e.g., [0,1])
    - categorical_levels_map: mapping from categorical name -> list of levels (used to convert category -> code index)
    - fixed_category_values: deprecated alias; kept for backward compatibility

    Returns dict mapping (game_state, is_net_empty) -> heat (2D numpy array)
    """
    import matplotlib.pyplot as plt
    from rink import draw_rink, rink_half_height_at_x, rink_bounds, rink_goal_xs

    if feature_cols is None:
        feature_cols = ['distance', 'angle_deg']
    if isinstance(feature_cols, str):
        feature_cols = [feature_cols]

    # normalize combination value inputs to lists
    if game_state_values is None:
        game_state_values = ['5v5']
    elif isinstance(game_state_values, (str, int)):
        game_state_values = [game_state_values]
    if is_net_empty_values is None:
        is_net_empty_values = [0]
    elif isinstance(is_net_empty_values, (str, int)):
        is_net_empty_values = [is_net_empty_values]

    # allow legacy fixed_category_values to populate a single-value run
    fixed_map = fixed_category_values or {}

    # get rink bounds
    xmin, xmax, ymin, ymax = rink_bounds()

    gx = np.arange(xmin, xmax + x_res, x_res)
    gy = np.arange(ymin, ymax + y_res, y_res)
    XX, YY = np.meshgrid(gx, gy)

    # mask outside rink
    mask = np.vectorize(rink_half_height_at_x)(XX) >= np.abs(YY)

    # choose attacked goal x-coordinate using rink helper
    left_goal_x, right_goal_x = rink_goal_xs()
    goal_x = left_goal_x if goal_side == 'left' else right_goal_x

    # precompute distances and angles for all valid grid points (flattened)
    pts = []
    coord_indices = []  # list of (i,j)
    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            if not mask[i, j]:
                continue
            x = float(XX[i, j])
            y = float(YY[i, j])
            dist = math.hypot(x - goal_x, y - 0.0)
            vx = x - goal_x
            vy = y - 0.0
            # rotate reference depending on goal side so angle convention matches parse
            if goal_x < 0:
                rx, ry = 0.0, 1.0
            else:
                rx, ry = 0.0, -1.0
            cross = rx * vy - ry * vx
            dot = rx * vx + ry * vy
            angle_rad_ccw = math.atan2(cross, dot)
            angle_deg = (-math.degrees(angle_rad_ccw)) % 360.0
            pts.append({'x': x, 'y': y, 'distance': dist, 'angle_deg': angle_deg})
            coord_indices.append((i, j))

    if len(pts) == 0:
        raise RuntimeError('No valid grid points found inside rink for heatmap generation.')

    # Convert pts to arrays for fast vectorized feature construction
    xs = np.array([p['x'] for p in pts])
    ys = np.array([p['y'] for p in pts])
    dists = np.array([p['distance'] for p in pts])
    angles = np.array([p['angle_deg'] for p in pts])

    results = {}

    # helper to convert categorical value to code index when feature uses *_code naming
    def category_value_to_code(feature_name: str, cat_value):
        # feature_name expected like 'game_state_code' -> base 'game_state'
        if not feature_name.endswith('_code'):
            # not a categorical code col
            return None
        base = feature_name[:-5]
        if categorical_levels_map and base in categorical_levels_map:
            levels = list(categorical_levels_map[base])
            try:
                return int(levels.index(str(cat_value)))
            except ValueError:
                # fallback to '__other__' if present
                if '__other__' in levels:
                    return int(levels.index('__other__'))
                # else fallback to 0
                return 0
        # if no mapping provided, try to cast directly
        try:
            return int(cat_value)
        except Exception:
            return 0

    # iterate combinations
    for gs in game_state_values:
        for nne in is_net_empty_values:
            # build feature matrix for each grid point according to feature_cols ordering
            Xgrid = []
            for k in range(len(xs)):
                row_feats = []
                for f in feature_cols:
                    if f == 'distance':
                        row_feats.append(dists[k])
                    elif f in ('angle_deg', 'angle'):
                        row_feats.append(angles[k])
                    elif f == 'dist_center':
                        row_feats.append(math.hypot(xs[k], ys[k]))
                    elif f == 'is_net_empty':
                        row_feats.append(int(nne))
                    elif f.endswith('_code'):
                        # determine categorical code for this combo
                        base = f[:-5]
                        if base == 'game_state':
                            code = category_value_to_code(f, gs)
                            row_feats.append(code)
                        else:
                            # no specific combo specified for this categorical variable
                            # try to use fixed_map or 0
                            val = fixed_map.get(base, None)
                            code = category_value_to_code(f, val) if val is not None else 0
                            row_feats.append(code)
                    else:
                        # unknown feature: try to default to NaN
                        row_feats.append(float('nan'))
                Xgrid.append(row_feats)
            Xgrid = np.array(Xgrid)

            # predict probabilities
            try:
                probs = clf.predict_proba(Xgrid)[:, 1]
            except Exception as e:
                raise RuntimeError('Model does not support predict_proba or X shape mismatch: ' + str(e))

            # fill heat grid
            heat = np.full(XX.shape, np.nan)
            for (i, j), p in zip(coord_indices, probs):
                heat[i, j] = p

            # Save image for this combination
            fig, ax = plt.subplots(figsize=(8, 4.5))
            draw_rink(ax=ax)
            extent = (gx[0] - x_res / 2.0, gx[-1] + x_res / 2.0, gy[0] - y_res / 2.0, gy[-1] + y_res / 2.0)
            im = ax.imshow(heat, extent=extent, origin='lower', cmap=cmap, alpha=alpha, zorder=1)
            ax.set_title(f'xG heatmap — game_state={gs} is_net_empty={nne}')
            ax.axis('off')
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            # sanitize gs for filename
            gs_tag = str(gs).replace(' ', '_')
            base, ext = (out_path, '') if out_path.endswith('.png') else (out_path + '.png', '')
            save_path = out_path.replace('.png', f'_gs-{gs_tag}_net-{nne}.png')
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            if verbose:
                print('Saved xG heatmap to', save_path)

            results[(gs, nne)] = heat

    return results
def analyze_game(game_id, clf=None):
    import nhl_api
    import parse
    # Default CSV and feature set for analysis; ensure final_features available
    csv_path = 'static/20252026.csv'
    features = ['distance', 'angle_deg', 'game_state', 'is_net_empty']

    if clf is None:
        # Load season data and preprocess for training. Capture the categorical
        # levels map so we can apply the exact same encoding to a single game.
        print('Demo: loading season data from', csv_path)
        season_df = load_data(csv_path)
        print('Demo: cleaning season data', season_df.shape)
        season_model_df, final_features, categorical_levels_map = clean_df_for_model(season_df, features)

        # Fit model on the cleaned season dataset
        print('Demo: fitting model...')
        clf, _, _ = fit_model(season_model_df, feature_cols=final_features)
    else:
        # If a classifier is provided, derive the feature ordering and category
        # levels from the canonical season CSV so we can encode the game data
        # consistently with the training setup.
        season_df = load_data(csv_path)
        _, final_features, categorical_levels_map = clean_df_for_model(season_df, features)

    # Assemble the game data and preprocess with the same feature set
    game_feed = nhl_api.get_game_feed(game_id)
    events = parse._game(game_feed)
    events = parse._elaborate(events)
    df = pd.DataFrame.from_records(events)

    # Use the same `features` list used for training to clean/encode the game df
    # and pass the categorical_levels_map obtained from training so encoding is
    # stable (unknown categories will be mapped to '__other__').
    df_model, final_feature_cols_game, categorical_cols_dummy_map = clean_df_for_model(
        df, features, fixed_categorical_levels=categorical_levels_map
    )

    # Now extract feature matrix
    X = df_model[final_features].values
    y = df_model['is_goal'].values

    # Evaluate model on game in question
    xgs, y_pred, metrics = evaluate_model(clf, X, y)

    # Map the predicted xG probabilities back onto the original game-level
    # DataFrame (`df`) using the index of `df_model`. Rows that were filtered
    # out during preprocessing will retain NaN for 'xgs'. This preserves the
    # original event ordering and makes downstream analysis simpler.
    # ensure original df exists in this scope (it was created earlier)
    df['xgs'] = np.nan
    # build a Series indexed by the df_model index so assignment aligns rows
    xgs_series = pd.Series(xgs, index=df_model.index)
    df.loc[xgs_series.index, 'xgs'] = xgs_series.values

    # Optionally, also attach predicted label (binary) if desired
    try:
        df['xg_pred'] = np.nan
        y_pred_series = pd.Series(y_pred, index=df_model.index)
        df.loc[y_pred_series.index, 'xg_pred'] = y_pred_series.values
    except Exception:
        # if y_pred isn't available or lengths mismatch, silently continue
        pass

    return df


if __name__ == '__main__':

    #analyze_game('2025020196')

    debug = True
    if debug:

        # Load data
        csv_path = 'static/20252026.csv'
        features = ['distance', 'angle_deg', 'game_state', 'is_net_empty']
        print('Demo: loading data from', csv_path)
        df = load_data(csv_path)
        print('Demo: cleaning data', df.shape)
        df, final_features, _ = clean_df_for_model(df, features)

        # Fit model
        print('Demo: fitting model...')
        clf, X_test, y_test = fit_model(df, feature_cols=final_features)

        # Evaluate model
        print('Demo: evaluating model...')
        y_prob, y_pred, metrics = evaluate_model(clf, X_test, y_test)
        print('Demo: metrics:')
        for k, v in metrics.items():
            try:
                print(f'  {k}: {v:.4f}')
            except Exception:
                print(f'  {k}: {v}')

        # Debug model
        print('Demo: debugging model...')
        debug_model(clf, feature_cols=final_features)
