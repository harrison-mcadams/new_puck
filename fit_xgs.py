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


def load_data(path: str = 'static/20252026.csv', feature_cols=None) -> pd.DataFrame:
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
        feature_cols = ['distance', 'angle_deg']
    if isinstance(feature_cols, str):
        feature_cols = [feature_cols]

    # Map of canonical feature -> possible alternate names to try
    alternates = {
        'dist_center': ['dist_center', 'distance', 'distance_to_goal', 'dist', 'distance_m'],
        'angle_deg': ['angle_deg', 'angle', 'angle_degree', 'angle_degs'],
    }

    found_cols = []
    for feat in feature_cols:
        # prefer exact match first
        if feat in df.columns:
            found_cols.append(feat)
            continue
        # look up alternates if we have a mapping
        alt_names = alternates.get(feat, [feat])
        found = None
        for n in alt_names:
            if n in df.columns:
                found = n
                break
        if found is None:
            raise ValueError(f'Could not find feature column for {feat}. Tried: {alt_names}')
        # copy into canonical name so downstream code can rely on standard names
        if found != feat:
            df[feat] = df[found]
        found_cols.append(feat)

    # target expected as 'is_goal' (0/1)
    if 'is_goal' not in df.columns:
        for alt in ('goal', 'isGoal', 'is_goal_flag'):
            if alt in df.columns:
                df['is_goal'] = df[alt]
                break
    if 'is_goal' not in df.columns:
        raise ValueError('Could not find target column `is_goal` in CSV')

    # Select only necessary columns and drop NA
    cols_to_use = list(found_cols) + ['is_goal']
    df = df[cols_to_use].copy()
    # Ensure numeric types for features
    df[found_cols] = df[found_cols].apply(pd.to_numeric, errors='coerce')
    df['is_goal'] = pd.to_numeric(df['is_goal'], errors='coerce').astype('Int64')
    df = df.dropna()
    df['is_goal'] = df['is_goal'].astype(int)

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
        feature_cols = ['distance', 'angle_deg']
    if isinstance(feature_cols, str):
        feature_cols = [feature_cols]

    X = df[feature_cols].values
    y = df['is_goal'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
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

    metrics = {
        'accuracy': accuracy,
        'log_loss': logloss,
        'roc_auc': rocauc,
        'brier': brier,
    }

    return y_prob, y_pred, metrics


def fit_and_evaluate(
    df: pd.DataFrame,
    feature_cols=None,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
):
    """Backward compatible wrapper: fits the model and returns the same
    result dict as before.
    """
    clf, X_test, y_test = fit_model(df, feature_cols=feature_cols, test_size=test_size, random_state=random_state, n_estimators=n_estimators)
    y_prob, y_pred, metrics = evaluate_model(clf, X_test, y_test)

    results = {
        'model': clf,
        'X_test': X_test,
        'y_test': y_test,
        'y_prob': y_prob,
        'metrics': metrics,
    }
    return results


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


def run(
    csv_path: str = 'static/20252026.csv',
    feature_cols=None,
    plot_path: str = 'static/xg_likelihood.png',
    **fit_kwargs,
):
    """Convenience runner: load data, fit, evaluate, and plot results.

    Returns the training results dictionary from `fit_and_evaluate`.
    """
    print(f'Loading data from: {csv_path}')
    df = load_data(csv_path, feature_cols=feature_cols)
    print('Data shape (after cleaning):', df.shape)

    print('Features used:', feature_cols or ['distance', 'angle_deg'])
    results = fit_and_evaluate(df, feature_cols=feature_cols, **fit_kwargs)

    print('Metrics:')
    for k, v in results['metrics'].items():
        print(f'  {k}: {v:.4f}')

    try:
        plot_calibration(results['y_test'], results['y_prob'], path=plot_path)
        print('Saved calibration plot to', plot_path)
    except Exception as e:
        print('Failed to save calibration plot:', e)

    return results


def _rink_half_height_at_x(x: float) -> float:
    """Return half-height of rink at x (same geometry as rink.draw_rink).

    Uses RINK_LENGTH=200, RINK_WIDTH=85 geometry.
    """
    R = 85.0 / 2.0
    half_length = 200.0 / 2.0
    straight_half = half_length - R
    left_center_x = -straight_half
    right_center_x = straight_half
    if left_center_x <= x <= right_center_x:
        return R
    # else inside semicircle region
    center = left_center_x if x < left_center_x else right_center_x
    dx = abs(x - center)
    if dx > R:
        return 0.0
    return math.sqrt(max(0.0, R * R - dx * dx))


def plot_xg_heatmap(clf, feature_cols=None, goal_side: str = 'left',
                    x_res: float = 2.0, y_res: float = 2.0,
                    out_path: str = 'static/xg_heatmap.png', cmap='viridis',
                    alpha: float = 0.8, verbose: bool = True):
    """Simulate shots across the rink, predict xG using clf, and plot heatmap.

    Parameters:
    - clf: trained classifier with predict_proba
    - feature_cols: list of feature names; default ['distance','angle_deg']
    - goal_side: 'left' or 'right' — which goal the shots are directed at
    - x_res, y_res: grid resolution in feet
    - out_path: file path to save image
    """
    import matplotlib.pyplot as plt
    from rink import draw_rink

    if feature_cols is None:
        feature_cols = ['distance', 'angle_deg']
    if isinstance(feature_cols, str):
        feature_cols = [feature_cols]

    # define grid bounds roughly matching rink extents
    xmin, xmax = -89.0, 89.0
    ymin, ymax = -42.5, 42.5

    xs = np.arange(xmin, xmax + x_res, x_res)
    ys = np.arange(ymin, ymax + y_res, y_res)
    xx, yy = np.meshgrid(xs, ys)

    # mask points outside rink
    mask = np.vectorize(lambda x: _rink_half_height_at_x(x))(xx) >= np.abs(yy)

    # choose goal_x
    goal_x = -89.0 if goal_side == 'left' else 89.0

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
    extent = [xs[0] - x_res/2.0, xs[-1] + x_res/2.0, ys[0] - y_res/2.0, ys[-1] + y_res/2.0]
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


def train_and_plot_xg_heatmap(csv_path: str = 'static/20252026.csv', feature_cols=None, plot_path: str = 'static/xg_heatmap.png',
                              model_kwargs=None, grid_res: float = 2.0, goal_side: str = 'left', verbose: bool = True):
    """Convenience: train model on CSV then plot heatmap using trained model."""
    model_kwargs = model_kwargs or {}
    df = load_data(csv_path, feature_cols=feature_cols)
    clf, X_test, y_test = fit_model(df, feature_cols=feature_cols, **model_kwargs)
    plot_xg_heatmap(clf, feature_cols=feature_cols, goal_side=goal_side, x_res=grid_res, y_res=grid_res, out_path=plot_path, verbose=verbose)


if __name__ == '__main__':
    # Demo: train on default CSV and produce an xG heatmap
    try:
        csv_path = 'static/20252026.csv'
        features = ['distance', 'angle_deg']
        print('Demo: loading data from', csv_path)
        df = load_data(csv_path, feature_cols=features)
        print('Demo: data shape after cleaning', df.shape)

        print('Demo: fitting model...')
        results = fit_and_evaluate(df, feature_cols=features, n_estimators=200)
        print('Demo: metrics:')
        for k, v in results['metrics'].items():
            try:
                print(f'  {k}: {v:.4f}')
            except Exception:
                print(f'  {k}: {v}')

        print('Demo: generating xG heatmap...')
        train_and_plot_xg_heatmap(csv_path=csv_path, feature_cols=features, plot_path='static/xg_heatmap.png', model_kwargs={'n_estimators':200}, grid_res=2.0, goal_side='left', verbose=True)
        print('Demo completed.')
    except Exception as e:
        print('Demo failed:', e)
