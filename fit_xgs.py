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


def load_data(path: str = 'static/20252026.csv', feature_cols=None, categorical_cols=None):
    """Load the season CSV and return a cleaned DataFrame.

    Parameters
    - path: CSV path (default 'static/20252026.csv')
    - feature_cols: a column name or list of column names to use as features
      (default ['dist_center', 'angle_deg']). The function will try common
      alternate names if the requested columns are missing.

    Returns a DataFrame with at least the feature column and `is_goal` target.
    The function will drop rows missing the required fields.
    """
    df = pd.read_csv(path)

    # Normalize feature_cols to a list
    if feature_cols is None:
        feature_cols = ['distance', 'angle_deg', 'game_state', 'is_net_empty']
    if isinstance(feature_cols, str):
        feature_cols = [feature_cols]

    # Select only necessary columns and drop NA after preprocessing
    cols_to_use = list(feature_cols) + ['is_goal']
    # ensure any missing contextual columns are added as defaults
    for c in cols_to_use:
        if c not in df.columns and c != 'is_goal':
            # default game_state -> '5v5', is_net_empty -> 0, others -> NaN
            if c == 'game_state':
                df[c] = '5v5'
            elif c == 'is_net_empty':
                df[c] = 0
            else:
                df[c] = pd.NA

    df = df[cols_to_use].copy()

    # Determine categorical columns to one-hot encode
    if categorical_cols is None:
        # auto-detect categorical columns among requested feature_cols by dtype
        categorical_cols = [c for c in feature_cols if c in df.columns and df[c].dtype == object]
    else:
        # ensure it's a list
        if isinstance(categorical_cols, str):
            categorical_cols = [categorical_cols]

    final_feature_cols = list(feature_cols)
    categorical_dummies_map = {}
    if categorical_cols:
        # make sure columns exist and fill defaults for known contextual ones
        for c in categorical_cols:
            if c not in df.columns:
                if c == 'game_state':
                    df[c] = '5v5'
                else:
                    df[c] = df[c].fillna('') if c in df.columns else ''
        # create dummies for all categorical_cols at once
        dummies = pd.get_dummies(df[categorical_cols].fillna(''), prefix=categorical_cols, prefix_sep='_')
        # concat dummies and drop original categorical columns
        df = pd.concat([df.drop(columns=categorical_cols), dummies], axis=1)
        # update final_feature_cols: replace each categorical col with its dummy names
        for cat in categorical_cols:
            # find the dummy columns generated for this cat (prefix 'cat_')
            prefix = f"{cat}_"
            cat_dummies = [col for col in dummies.columns if col.startswith(prefix)]
            categorical_dummies_map[cat] = cat_dummies
            if cat in final_feature_cols:
                idx = final_feature_cols.index(cat)
                final_feature_cols = final_feature_cols[:idx] + cat_dummies + final_feature_cols[idx+1:]
            else:
                # if the categorical column wasn't requested as a feature but dummies exist,
                # append them to the end of the feature list
                final_feature_cols.extend(cat_dummies)

    # Ensure numeric types for remaining (non-dummy) features
    numeric_cols = [c for c in final_feature_cols if not any(c.startswith(pref + '_') for pref in (categorical_cols or []))]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df['is_goal'] = pd.to_numeric(df['is_goal'], errors='coerce').astype('Int64')

    # Drop rows with missing required values
    df = df[final_feature_cols + ['is_goal']].dropna().copy()
    df['is_goal'] = df['is_goal'].astype(int)

    return df, final_feature_cols, categorical_dummies_map


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




if __name__ == '__main__':

    debug = True
    if debug:

        # Load data
        csv_path = 'static/20252026.csv'
        features = ['distance', 'angle_deg', 'game_state', 'is_net_empty']
        print('Demo: loading data from', csv_path)
        df, final_features, gs_dummies = load_data(csv_path, feature_cols=features)
        print('Demo: data shape after cleaning', df.shape)

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


