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


def load_data(path: str = 'static/20252026.csv', feature_col: str = 'dist_center') -> pd.DataFrame:
    """Load the season CSV and return a cleaned DataFrame.

    Parameters
    - path: CSV path (default 'static/20252026.csv')
    - feature_col: name of the distance feature column to use (default 'dist_center')

    Returns a DataFrame with at least the feature column and `is_goal` target.
    The function will drop rows missing the required fields.
    """
    df = pd.read_csv(path)

    # Normalize expected column names if present under alternate names
    alt_feature_names = [feature_col, 'distance', 'distance_to_goal', 'dist']
    found_feature = None
    for n in alt_feature_names:
        if n in df.columns:
            found_feature = n
            break
    if found_feature is None:
        raise ValueError(f'Could not find a distance column in CSV. Tried: {alt_feature_names}')
    if found_feature != feature_col:
        df[feature_col] = df[found_feature]

    # target expected as 'is_goal' (0/1)
    if 'is_goal' not in df.columns:
        # try common alternates
        for alt in ('goal', 'isGoal', 'is_goal_flag'):
            if alt in df.columns:
                df['is_goal'] = df[alt]
                break
    if 'is_goal' not in df.columns:
        raise ValueError('Could not find target column `is_goal` in CSV')

    # Drop rows missing crucial fields
    df = df[[feature_col, 'is_goal']].dropna()

    # Ensure numeric types
    df[feature_col] = pd.to_numeric(df[feature_col], errors='coerce')
    df['is_goal'] = pd.to_numeric(df['is_goal'], errors='coerce').astype('Int64')
    df = df.dropna()

    # Convert target to 0/1 ints
    df['is_goal'] = df['is_goal'].astype(int)

    return df



def fit_model(
    df: pd.DataFrame,
    feature_col: str = 'dist_center',
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

    X = df[[feature_col]].values
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
    feature_col: str = 'dist_center',
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
):
    """Backward compatible wrapper: fits the model and returns the same
    result dict as before.
    """
    clf, X_test, y_test = fit_model(df, feature_col=feature_col, test_size=test_size, random_state=random_state, n_estimators=n_estimators)
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
    feature_col: str = 'dist_center',
    plot_path: str = 'static/xg_likelihood.png',
    **fit_kwargs,
):
    """Convenience runner: load data, fit, evaluate, and plot results.

    Returns the training results dictionary from `fit_and_evaluate`.
    """
    print(f'Loading data from: {csv_path}')
    df = load_data(csv_path, feature_col=feature_col)
    print('Data shape (after cleaning):', df.shape)

    results = fit_and_evaluate(df, feature_col=feature_col, **fit_kwargs)

    print('Metrics:')
    for k, v in results['metrics'].items():
        print(f'  {k}: {v:.4f}')

    try:
        plot_calibration(results['y_test'], results['y_prob'], path=plot_path)
        print('Saved calibration plot to', plot_path)
    except Exception as e:
        print('Failed to save calibration plot:', e)

    return results


if __name__ == '__main__':
    # default quick run
    run()
