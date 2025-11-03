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

def clean_df_for_model(df: pd.DataFrame, feature_cols):
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

    # Encode categorical columns as integer codes (one column per category)
    if categorical_cols:
        df, categorical_dummies_map = one_hot_encode(df, categorical_cols, prefix_sep='_', fill_value='')
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

    return df, final_feature_cols, categorical_dummies_map

def one_hot_encode(df: pd.DataFrame, categorical_cols, prefix_sep: str = '_', fill_value: str = ''):
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

    # For each categorical column, create an integer code column and drop original
    for cat in categorical_cols:
        # convert to string and fill missing with fill_value
        ser = df[cat].fillna(fill_value).astype(str)
        # create a Categorical and use its codes (0..n-1)
        cat_obj = pd.Categorical(ser)
        codes = cat_obj.codes
        new_col = f"{cat}_code"
        df[new_col] = codes
        # drop the original textual column
        df = df.drop(columns=[cat])
        # return a list containing the single new column name for compatibility
        categorical_dummies_map[cat] = [new_col]

    return df, categorical_dummies_map


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
                    fixed_game_state='5v5', fixed_is_net_empty=0, categorical_dummies_map=None,
                    fixed_category_values: dict = None):
    """Simulate shots across the rink, predict xG using clf, and plot heatmap.

    Parameters:
    - clf: trained classifier with predict_proba
    - feature_cols: list of feature names; default ['distance','angle_deg']
    - goal_side: 'left' or 'right' — which goal the shots are directed at
    - x_res, y_res: grid resolution in feet
    - out_path: file path to save image
    """
    import matplotlib.pyplot as plt
    from rink import draw_rink, rink_half_height_at_x, rink_bounds, rink_goal_xs

    if feature_cols is None:
        feature_cols = ['distance', 'angle_deg']
    if isinstance(feature_cols, str):
        feature_cols = [feature_cols]

    # get rink bounds from rink.py to avoid hard-coded geometry
    xmin, xmax, ymin, ymax = rink_bounds()

    xs = np.arange(xmin, xmax + x_res, x_res)
    ys = np.arange(ymin, ymax + y_res, y_res)
    xx, yy = np.meshgrid(xs, ys)

    # mask points outside rink using the canonical rink helper (vectorized)
    mask = np.vectorize(rink_half_height_at_x)(xx) >= np.abs(yy)

    # choose attacked goal x-coordinate using rink helper
    left_goal_x, right_goal_x = rink_goal_xs()
    goal_x = left_goal_x if goal_side == 'left' else right_goal_x

    # prepare feature matrix
    pts = []
    coords = []
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            if not mask[i, j]:
                continue
            x = float(xx[i, j])
            y = float(yy[i, j])
            # distance to goal
            dist = math.hypot(x - goal_x, y - 0.0)
            # angle: same convention as parse._elaborate
            vx = x - goal_x
            vy = y - 0.0
            if goal_x < 0:
                rx, ry = 0.0, 1.0
            else:
                rx, ry = 0.0, -1.0
            cross = rx * vy - ry * vx
            dot = rx * vx + ry * vy
            angle_rad_ccw = math.atan2(cross, dot)
            angle_deg = (-math.degrees(angle_rad_ccw)) % 360.0
            feat = []
            for f in feature_cols:
                if f == 'distance':
                    feat.append(dist)
                elif f in ('angle_deg', 'angle'):
                    feat.append(angle_deg)
                elif f == 'dist_center':
                    feat.append(math.hypot(x, y))
                elif f == 'is_net_empty':
                    try:
                        nne = int(fixed_is_net_empty)
                    except Exception:
                        nne = 0
                    feat.append(nne)
                elif categorical_dummies_map and any(f in lst for lst in categorical_dummies_map.values()):
                    # determine which categorical variable this dummy belongs to
                    dummy_to_cat = {d: cat for cat, lst in categorical_dummies_map.items() for d in lst}
                    cat = dummy_to_cat.get(f)
                    # determine the fixed category value for this categorical variable
                    fixed_map = fixed_category_values or {}
                    if cat and cat not in fixed_map:
                        # provide backwards-compatible default for 'game_state'
                        if cat == 'game_state':
                            fixed_map[cat] = fixed_game_state
                    fixed_val = fixed_map.get(cat)
                    # build expected dummy name for the fixed value
                    expected_dummy = f"{cat}_{fixed_val}" if fixed_val is not None else None
                    if expected_dummy is not None and f == expected_dummy:
                        feat.append(1)
                    else:
                        feat.append(0)
                else:
                    # unknown feature; try to default to NaN
                    feat.append(float('nan'))
            pts.append(feat)
            coords.append((i, j))

    if not pts:
        raise RuntimeError('No valid grid points found inside rink.')

    Xgrid = np.array(pts)
    try:
        probs = clf.predict_proba(Xgrid)[:, 1]
    except Exception as e:
        raise RuntimeError('Model does not support predict_proba or X shape mismatch: ' + str(e))

    # fill heat array
    heat = np.full(xx.shape, np.nan)
    for (i, j), p in zip(coords, probs):
        heat[i, j] = p

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    draw_rink(ax=ax)

    # Overlay heatmap; set extent to match xs, ys
    extent = (xs[0] - x_res/2.0, xs[-1] + x_res/2.0, ys[0] - y_res/2.0, ys[-1] + y_res/2.0)
    im = ax.imshow(heat, extent=extent, origin='lower', cmap=cmap, alpha=alpha, zorder=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('xG')

    plt.title('xG heatmap (simulated shots)')
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    if verbose:
        print('Saved xG heatmap to', out_path)


def analyze_game(game_id, clf=None):
    import nhl_api
    import parse
    # Default CSV and feature set for analysis; ensure final_features available
    csv_path = 'static/20252026.csv'
    features = ['distance', 'angle_deg', 'game_state', 'is_net_empty']

    if clf is None:
        # Load data and train if no classifier provided
        print('Demo: loading data from', csv_path)
        df = load_data(csv_path)
        print('Demo: cleaning data', df.shape)
        df, final_features, _ = clean_df_for_model(df, features)

        # Fit model
        print('Demo: fitting model...')
        clf, _, _ = fit_model(df, feature_cols=final_features)
    else:
        # If a classifier is provided, load dataset just to obtain feature ordering
        # Note: callers providing a pre-fit clf should ensure it was trained on the
        # same `final_features` ordering; here we derive a compatible ordering from
        # the canonical CSV using the requested features.
        _, final_features, gs_dummies = load_data(csv_path, feature_cols=features)

    # Assemble the game data and preprocess with the same feature set
    game_feed = nhl_api.get_game_feed(game_id)
    events = parse._game(game_feed)
    events = parse._elaborate(events)
    df = pd.DataFrame.from_records(events)

    # Use the same `features` list used for training to clean/encode the game df
    df, final_feature_cols_game, categorical_cols_dummy_map = clean_df_for_model(df, features)

    # Now extract feature matrix
    X = df[final_features].values
    y = df['is_goal'].values

    # Evaluate model on game in question
    xgs, y_pred, metrics = evaluate_model(clf, X, y)

    # concatenate xG results back to game DataFrame for further analysis if desired
    df['xgs'] = xgs

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
