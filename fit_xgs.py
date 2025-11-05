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
import joblib
import json
import sys
import time

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

# --- Simple module-level caching for the trained classifier ---
_GLOBAL_CLF = None
_GLOBAL_FINAL_FEATURES = None
_GLOBAL_CATEGORICAL_LEVELS_MAP = None


def get_clf(out_path: str = 'static/xg_model.joblib', behavior: str = 'load', *,
            csv_path: str = 'static/20252026.csv',
            features=None,
            random_state: int = 42,
            n_estimators: int = 200):
    """Train or load a RandomForest classifier for xG.

    Parameters
    - out_path: path to save/load the classifier (joblib file)
    - behavior: 'train' to train & save, 'load' to load from disk
    - csv_path/features/random_state/n_estimators: training params used when behavior='train'

    Returns: (clf, final_features, categorical_levels_map)
    """
    # normalize behavior
    b = (behavior or '').strip().lower()
    if b not in ('train', 'load'):
        raise ValueError("behavior must be 'train' or 'load'")

    meta_path = out_path + '.meta.json'

    if b == 'load':
        # Try to load model and metadata from disk
        try:
            clf = joblib.load(out_path)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load classifier from {out_path}: {e}")
        # try to load metadata (features + categorical levels)
        final_features = None
        categorical_levels_map = None
        try:
            with open(meta_path, 'r', encoding='utf-8') as fh:
                meta = json.load(fh)
                final_features = meta.get('final_features')
                categorical_levels_map = meta.get('categorical_levels_map')
        except Exception:
            # metadata missing is not fatal; caller may re-derive
            final_features = None
            categorical_levels_map = None
        return clf, final_features, categorical_levels_map

    # else: train
    # default features
    if features is None:
        features = ['distance', 'angle_deg', 'game_state', 'is_net_empty']

    # load data and prepare
    season_df = load_data(csv_path)
    season_model_df, final_features, categorical_levels_map = clean_df_for_model(season_df, features)

    # fit model
    clf, X_test, y_test = fit_model(season_model_df, feature_cols=final_features,
                                    random_state=random_state, n_estimators=n_estimators)

    # evaluate model and produce calibration plot (keeps previous behavior)
    try:
        y_prob, y_pred, metrics = evaluate_model(clf, X_test, y_test)
    except Exception:
        # evaluation shouldn't block saving the model
        metrics = None

    # persist classifier and metadata
    try:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, out_path)
        meta = {'final_features': final_features, 'categorical_levels_map': categorical_levels_map}
        with open(meta_path, 'w', encoding='utf-8') as fh:
            json.dump(meta, fh)
    except Exception as e:
        # if saving fails, warn but return the in-memory clf
        try:
            print(f'Warning: failed to save classifier or metadata to {out_path}: {e}')
        except Exception:
            pass

    return clf, final_features, categorical_levels_map

# --- end of module-level caching helpers ---

def get_or_train_clf(force_retrain: bool = False,
                     csv_path: str = 'static/20252026.csv',
                     features=None,
                     random_state: int = 42,
                     n_estimators: int = 200):
    """Return a trained classifier and associated metadata.

    This helper will train a RandomForest on the season CSV the first time
    it's called and cache the classifier (and the final feature column list
    and categorical levels map). Subsequent calls return the cached object
    unless force_retrain=True.

    Returns (clf, final_features, categorical_levels_map).
    """
    global _GLOBAL_CLF, _GLOBAL_FINAL_FEATURES, _GLOBAL_CATEGORICAL_LEVELS_MAP
    if _GLOBAL_CLF is not None and not force_retrain:
        return _GLOBAL_CLF, _GLOBAL_FINAL_FEATURES, _GLOBAL_CATEGORICAL_LEVELS_MAP

    # Default features if none provided
    if features is None:
        features = ['distance', 'angle_deg', 'game_state', 'is_net_empty']

    # Train a fresh model and cache metadata
    season_df = load_data(csv_path)
    season_model_df, final_features, categorical_levels_map = clean_df_for_model(season_df, features)

    clf, _, _ = fit_model(season_model_df, feature_cols=final_features,
                          random_state=random_state, n_estimators=n_estimators)

    _GLOBAL_CLF = clf
    _GLOBAL_FINAL_FEATURES = final_features
    _GLOBAL_CATEGORICAL_LEVELS_MAP = categorical_levels_map
    return _GLOBAL_CLF, _GLOBAL_FINAL_FEATURES, _GLOBAL_CATEGORICAL_LEVELS_MAP

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
    progress: bool = False,
    progress_steps: int = 20,
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

    # Simple console progress helper
    def _print_progress(prefix: str, i: int, total: int, width: int = 40):
        frac = float(i) / float(total)
        filled = int(round(width * frac))
        bar = '#' * filled + '-' * (width - filled)
        sys.stdout.write(f"\r{prefix} |{bar}| {i}/{total} ({frac*100:5.1f}%)")
        sys.stdout.flush()

    if not progress:
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        clf.fit(X_train, y_train)
        return clf, X_test, y_test

    # Progress mode: train in chunks using warm_start to reveal progress
    if RandomForestClassifier is None:
        raise RuntimeError('scikit-learn is required to run training with progress.')

    # Determine chunk size (at most progress_steps updates)
    steps = max(1, int(progress_steps))
    chunk = max(1, n_estimators // steps)
    clf = RandomForestClassifier(n_estimators=0, warm_start=True, random_state=random_state)
    trained = 0
    total = n_estimators
    try:
        while trained < total:
            to_add = min(chunk, total - trained)
            clf.n_estimators = trained + to_add
            # Fit will add `to_add` new trees when warm_start=True
            clf.fit(X_train, y_train)
            trained = clf.n_estimators
            _print_progress('Training RF', trained, total)
            # small sleep to ensure progress is visible on fast machines
            time.sleep(0.01)
        # finish line
        _print_progress('Training RF', total, total)
        sys.stdout.write('\n')
    except Exception:
        # fallback to single-shot fit if incremental fails
        sys.stdout.write('\n')
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
                    fixed_category_values: dict = None,
                    interactive: bool = False):
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
    - interactive: bool, if True will display an interactive GUI to browse heatmaps

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
        if categorical_levels_map and 'game_state' in categorical_levels_map:
            game_state_values = list(categorical_levels_map['game_state'])
        else:
            game_state_values = ['5v5', '5v4', '4v5']
    elif isinstance(game_state_values, (str, int)):
        game_state_values = [game_state_values]
    if is_net_empty_values is None:
        is_net_empty_values = [0, 1]
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

    # Build explicit list of combinations (cartesian product) so we're sure we
    # compute and save the same set in order.
    from itertools import product
    combos = list(product(game_state_values, is_net_empty_values))
    if verbose:
        try:
            print(f"debug_model: computing heatmaps for {len(combos)} combinations")
        except Exception:
            pass

    # First pass: compute heat arrays for all combinations and store them
    for gs, nne in combos:
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
                        val = fixed_map.get(base, None)
                        code = category_value_to_code(f, val) if val is not None else 0
                        row_feats.append(code)
                else:
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

        results[(gs, nne)] = heat

    # Determine a common color scale across all combinations.
    # By default use the reference condition (game_state='5v5', is_net_empty=0)
    # to set the upper bound — this narrows the colorbar to a meaningful
    # operational range. If that reference is missing or has zero max, fall
    # back to the global maximum across all combos.
    all_max = 0.0
    for h in results.values():
        try:
            mv = float(np.nanmax(h))
            if mv > all_max:
                all_max = mv
        except Exception:
            continue

    # reference key and its max value (if available)
    ref_key = ('5v5', 0)
    ref_max = None
    if ref_key in results:
        try:
            ref_max = float(np.nanmax(results[ref_key]))
        except Exception:
            ref_max = None

    vmin = 0.0
    # Prefer the reference max if it exists and is > 0; otherwise use global max.
    if ref_max is not None and ref_max > 0:
        vmax = ref_max
    else:
        # ensure a sensible small positive fallback to avoid degenerate range
        vmax = max(all_max, 1.0e-6)

    # Second pass: render and save each heatmap using the common vmin/vmax and add colorbar
    saved_paths = []
    for gs, nne in combos:
        heat = results.get((gs, nne))
        if heat is None:
            # if compute failed or missing, create an empty (nan) heat grid to save a placeholder
            heat = np.full(XX.shape, np.nan)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        draw_rink(ax=ax)
        extent = (gx[0] - x_res / 2.0, gx[-1] + x_res / 2.0, gy[0] - y_res / 2.0, gy[-1] + y_res / 2.0)
        im = ax.imshow(heat, extent=extent, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, zorder=1)
        # readable title
        ax.set_title(f'xG heatmap — game_state: {gs}   |   empty net: {nne}', fontsize=10)
        ax.axis('off')

        # add colorbar with same scaling across all images
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('xG probability')

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        gs_tag = str(gs).replace(' ', '_')
        # build save path reliably using splitext to avoid accidental replace issues
        import os
        base, ext = os.path.splitext(out_path)
        if ext == '':
            ext = '.png'
        save_path = f"{base}_gs-{gs_tag}_net-{nne}{ext}"
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        # Always print saved path so user can inspect output
        try:
            print('Saved xG heatmap to', save_path)
        except Exception:
            pass
        saved_paths.append(save_path)

    try:
        print(f"debug_model: saved {len(saved_paths)} heatmap files")
    except Exception:
        pass

    # If interactive mode requested, attempt to open a small matplotlib GUI
    if interactive:
        try:
            # try to ensure an interactive backend is available. If the current
            # backend is non-interactive (Agg) we attempt to switch to a GUI
            # backend such as TkAgg, Qt5Agg, or MacOSX. This requires the
            # relevant GUI bindings (tkinter/Qt) to be installed on the system.
            import matplotlib as mpl
            import importlib
            non_interactive_names = ('agg', 'pdf', 'svg', 'ps')
            current_backend = mpl.get_backend().lower()
            if any(n in current_backend for n in non_interactive_names):
                # Try common interactive backends until one works
                tried = []
                success = False
                for candidate in ('TkAgg', 'Qt5Agg', 'MacOSX'):
                    try:
                        mpl.use(candidate, force=True)
                        # reload pyplot to pick up backend change
                        importlib.reload(plt)
                        new_backend = mpl.get_backend().lower()
                        if not any(n in new_backend for n in non_interactive_names):
                            success = True
                            break
                    except Exception:
                        tried.append(candidate)
                        continue
                if not success:
                    raise RuntimeError(f'No interactive matplotlib backend available (tried: {tried}). Please install tkinter or Qt and set MPLBACKEND to a GUI backend.')

            # re-check backend: if still non-interactive, bail out with clear message
            final_backend = mpl.get_backend().lower()
            if any(n in final_backend for n in non_interactive_names):
                raise RuntimeError(
                    "Interactive backend not available (current backend='{}').\n".format(final_backend)
                    + "To enable interactive mode: set the environment variable MPLBACKEND to a GUI backend (e.g. 'TkAgg') before starting Python,\n"
                    + "or remove any 'matplotlib.use('Agg')' calls in project modules so a GUI backend can be selected."
                )

            # Prepare a figure for interactive browsing. Reserve a larger left
            # margin so the RadioButtons do not overlap the rink area.
            fig, ax = plt.subplots(figsize=(8, 4.5))
            # leave more room on the left for controls (increase left margin)
            fig.subplots_adjust(left=0.22, right=0.95, top=0.92)

            extent = (gx[0] - x_res / 2.0, gx[-1] + x_res / 2.0, gy[0] - y_res / 2.0, gy[-1] + y_res / 2.0)

            idx = 0
            total = len(combos)
            gs0, nne0 = combos[idx]
            heat0 = results.get((gs0, nne0), np.full(XX.shape, np.nan))
            im = ax.imshow(heat0, extent=extent, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, zorder=1)
            title = ax.set_title(f'xG heatmap — game_state: {gs0}   |   empty net: {nne0}', fontsize=10)
            ax.axis('off')

            # colorbar
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('xG probability')

            # --- INTERACTIVE CONTROLS: RadioButtons for each feature ---
            # Place the control panel on the left side. Two groups:
            #  - game_state (list of strings)
            #  - is_net_empty (list of integers -> convert to str for labels)
            # import widgets here (after backend check)
            from matplotlib import widgets

            # move the control panels further left (smaller x) so they do not
            # overlap with the rink drawing when the left margin is large.
            gs_ax = plt.axes((0.02, 0.3, 0.16, 0.6))
            net_ax = plt.axes((0.02, 0.15, 0.16, 0.12))

            # convert values to strings for display
            gs_labels = [str(v) for v in game_state_values]
            net_labels = [str(v) for v in is_net_empty_values]

            # create RadioButtons; active=0 selects the first item
            rgs = widgets.RadioButtons(gs_ax, gs_labels, active=0)
            rnet = widgets.RadioButtons(net_ax, net_labels, active=0)

            # current selections stored in closure
            selected = {'gs': game_state_values[0], 'net': is_net_empty_values[0]}

            def _show_selection():
                # fetch heat for currently selected combo and update image/title
                gs_val = selected['gs']
                nne_val = selected['net']
                h = results.get((gs_val, nne_val), np.full(XX.shape, np.nan))
                im.set_data(h)
                title.set_text(f'xG heatmap — game_state: {gs_val}   |   empty net: {nne_val}')
                fig.canvas.draw_idle()

            def _on_gs_clicked(label):
                # label is string; convert back to original value if needed
                try:
                    # if original values are ints, try to cast
                    idx = gs_labels.index(label)
                    selected['gs'] = game_state_values[idx]
                except Exception:
                    selected['gs'] = label
                _show_selection()

            def _on_net_clicked(label):
                try:
                    idx = net_labels.index(label)
                    selected['net'] = is_net_empty_values[idx]
                except Exception:
                    # fallback: try numeric cast
                    try:
                        selected['net'] = int(label)
                    except Exception:
                        selected['net'] = label
                _show_selection()

            rgs.on_clicked(_on_gs_clicked)
            rnet.on_clicked(_on_net_clicked)

            # Add a small Reset Colorbar button under the colorbar to rescale
            # the displayed heatmap's color range to its own max. Position it
            # relative to the colorbar axes so it won't overlap the rink.
            try:
                cbbox = cbar.ax.get_position()
                # place button centered under colorbar, small height
                bx = cbbox.x0
                by = max(0.01, cbbox.y0 - 0.055)
                bwidth = cbbox.width
                bheight = 0.04
                reset_ax = plt.axes((bx, by, bwidth, bheight))
            except Exception:
                # fallback to a safe right-side position
                reset_ax = plt.axes((0.88, 0.02, 0.08, 0.04))
            reset_btn = widgets.Button(reset_ax, 'Reset CB')

            def _reset_colorbar(event):
                # get current selection
                gs_val = selected.get('gs', game_state_values[0])
                nne_val = selected.get('net', is_net_empty_values[0])
                h = results.get((gs_val, nne_val), None)
                if h is None:
                    try:
                        print('No heat available to reset colorbar')
                    except Exception:
                        pass
                    return
                try:
                    new_max = float(np.nanmax(h))
                except Exception:
                    new_max = 0.0
                if not (new_max and new_max > 0):
                    new_max = 1.0e-6
                # update mappable and colorbar
                try:
                    im.set_clim(0.0, new_max)
                    try:
                        cbar.update_normal(im)
                    except Exception:
                        pass
                    fig.canvas.draw_idle()
                    print(f'Reset colorbar to vmax={new_max:.6g}')
                except Exception as e:
                    print('Failed to reset colorbar:', e)

            reset_btn.on_clicked(_reset_colorbar)

            print('Interactive mode: use RadioButtons to select game_state and is_net_empty values. Heatmap updates automatically.')
            # Block here until the user closes the figure window so the GUI stays open
            plt.show(block=True)
        except Exception as e:
            print('Interactive mode is unavailable in this environment:', str(e))

    return results

def analyze_game(game_id, clf=None):
    import nhl_api
    import parse
    # Default CSV and feature set for analysis; ensure final_features available
    csv_path = 'static/20252026.csv'
    features = ['distance', 'angle_deg', 'game_state', 'is_net_empty']

    if clf is None:
        # Try to load a persisted classifier; if that fails, train & save one.
        model_path = 'static/xg_model.joblib'
        try:
            clf, final_features, categorical_levels_map = get_clf(model_path, 'load')
        except Exception:
            # If loading failed, train and persist a new model
            clf, final_features, categorical_levels_map = get_clf(model_path, 'train', csv_path=csv_path, features=features)
    else:
        # If a classifier is provided, attempt to load metadata if available
        try:
            _, cached_final_features, cached_categorical = get_or_train_clf(force_retrain=False, csv_path=csv_path, features=features)
            final_features = cached_final_features
            categorical_levels_map = cached_categorical
        except Exception:
            # As before, derive feature ordering and category levels from the canonical season CSV
            season_df = load_data(csv_path)
            _, final_features, categorical_levels_map = clean_df_for_model(season_df, features)

    # Assemble the game data and preprocess with the same feature set
    game_feed = nhl_api.get_game_feed(game_id)
    df = parse._game(game_feed)
    df = parse._elaborate(df)

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

    # Demo / CLI runner: load season-level dataframes from local ``data/{year}/{year}_df.csv``
    # If none are found, fall back to the previous single CSV in ``static/20252026.csv``.
    debug = True
    if debug:
        from pathlib import Path

        data_dir = Path('data')
        candidates = []
        print('Demo: looking for per-year data in', str(data_dir))
        if data_dir.exists() and data_dir.is_dir():
            for p in sorted(data_dir.iterdir()):
                if p.is_dir():
                    year = p.name
                    candidate = p / f"{year}_df.csv"
                    if candidate.exists():
                        candidates.append(candidate)

        # If no per-year files found, try the legacy static path
        if not candidates:
            legacy = Path('static') / '20252026.csv'
            if legacy.exists():
                print('Demo: no per-year CSVs found under data/ — falling back to', str(legacy))
                candidates = [legacy]
            else:
                print('Demo: no data files found (checked data/ and static/). Exiting.')
                raise SystemExit(1)

        # Load and concatenate all candidate CSVs
        frames = []
        for c in candidates:
            try:
                print('Demo: loading', str(c))
                dfc = load_data(str(c))
                frames.append(dfc)
            except Exception as e:
                print('Demo: failed to load', str(c), ' — ', e)

        if not frames:
            print('Demo: no frames loaded successfully. Exiting.')
            raise SystemExit(1)

        df = pd.concat(frames, ignore_index=True)
        print('Demo: combined dataframe shape:', df.shape)

        # Define feature set (same as before)
        features = ['distance', 'angle_deg', 'game_state', 'is_net_empty']

        print('Demo: cleaning data')
        df, final_features, categorical_map = clean_df_for_model(df, features)

        # Fit model
        print('Demo: fitting model...')
        clf, X_test, y_test = fit_model(df, feature_cols=final_features,
                                        progress=True)

        # Evaluate model
        print('Demo: evaluating model...')
        y_prob, y_pred, metrics = evaluate_model(clf, X_test, y_test)
        print('Demo: metrics:')
        for k, v in metrics.items():
            try:
                print(f'  {k}: {v:.4f}')
            except Exception:
                print(f'  {k}: {v}')

        # Persist trained classifier and metadata (so get_clf('load') can reuse it)
        try:
            model_path = Path('static') / 'xg_model.joblib'
            meta_path = str(model_path) + '.meta.json'
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(clf, str(model_path))
            meta = {'final_features': final_features, 'categorical_levels_map': categorical_map}
            with open(meta_path, 'w', encoding='utf-8') as fh:
                json.dump(meta, fh)
            print(f'Demo: saved classifier to {model_path} and metadata to {meta_path}')
        except Exception as e:
            print('Demo: failed to save classifier/meta:', e)

        # Debug model (non-interactive visualization by default in demo)
        print('Demo: debugging model...')
        debug_model(clf, feature_cols=final_features,
                    categorical_levels_map=categorical_map, interactive=False)
