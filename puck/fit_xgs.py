"""fit_xgs.py

Simple, well-documented script to fit a lightweight xG-like model (shot -> goal
probability) using one primary feature (distance to goal). The goal is clarity
and a good starting point for extension.

Usage (from project root):
    python fit_xgs.py

The script will read 'data/processed/20252026/20252026.csv' by default, train a Random Forest on
`dist_center` -> `is_goal`, print evaluation metrics, and save a calibration
plot to 'analysis/xgs/xg_likelihood.png'.
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
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple

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

@dataclass
class ModelConfig:
    """Configuration for an xG model variant."""
    name: str
    features: List[str]
    n_estimators: int = 500  # Increased default for better baseline performance
    max_depth: Optional[int] = None
    min_samples_leaf: int = 1
    description: str = ""
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != 'name'}

def load_all_seasons_data(base_dir: str = 'data/processed') -> pd.DataFrame:
    """Load and concatenate all season CSVs found in data/processed."""
    base_path = Path(base_dir)
    if not base_path.exists():
        # Fallback to current season location if root not found
        fallback = Path('data/processed/20252026/20252026.csv')
        if fallback.exists():
            return pd.read_csv(fallback)
        raise FileNotFoundError(f"Processed data directory not found: {base_dir}")
        
    frames = []
    # Look for {year}/{year}.csv structure
    for year_dir in sorted(base_path.iterdir()):
        if year_dir.is_dir() and year_dir.name.isdigit():
            csv_path = year_dir / f"{year_dir.name}.csv"
            if csv_path.exists():
                print(f"Loading {year_dir.name} from {csv_path}...")
                try:
                    df = pd.read_csv(csv_path)
                    frames.append(df)
                except Exception as e:
                    print(f"Failed to load {csv_path}: {e}")
            else:
                 # Check for old {year}_df.csv style just in case of failed migration or mix
                 legacy_csv = year_dir / f"{year_dir.name}_df.csv"
                 if legacy_csv.exists():
                     print(f"Loading (legacy name) {year_dir.name} from {legacy_csv}...")
                     try:
                        df = pd.read_csv(legacy_csv)
                        frames.append(df)
                     except Exception as e:
                        print(f"Failed to load {legacy_csv}: {e}")
    
    if not frames:
        print("No season data found in data/processed.")
        raise FileNotFoundError("No season data found.")

    full_df = pd.concat(frames, ignore_index=True)
    print(f"Total loaded rows: {len(full_df)}")
    return full_df

def compare_models(configs: List[ModelConfig], 
                   df_train: pd.DataFrame, 
                   df_test: pd.DataFrame,
                   random_state: int = 42) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Train multiple models defined by configs and evaluate them on the same test set.
    
    Returns:
        models: Dict[str, clf] - trained classifiers
        results_df: pd.DataFrame - Comparison metrics
    """
    results = []
    models = {}
    
    print(f"\n--- Comparing {len(configs)} Models ---")
    
    for conf in configs:
        print(f"Training '{conf.name}'...")
        # Prepare data for this specific model configuration
        # Note: We re-clean/encode for each model because features might differ
        # (e.g. one uses cat codes, another uses dummies, or different feature subsets)
        
        # We need to act on the full train/test split to ensure consistent evaluation?
        # Actually, fit_model does splitting internally. 
        # To compare fairly, we should pass explicit Train/Test sets to fit_model or 
        # handle splitting outside. 
        # Let's adjust: taking df_train and df_test inputs allows us to control the split externally.
        
        # Prepare TRAIN
        train_df_mod, final_feats, cat_map = clean_df_for_model(df_train.copy(), conf.features)
        X_train = train_df_mod[final_feats].values
        y_train = train_df_mod['is_goal'].values
        
        # Prepare TEST (using the same categorical map/features)
        test_df_mod, _, _ = clean_df_for_model(df_test.copy(), conf.features, fixed_categorical_levels=cat_map)
        X_test = test_df_mod[final_feats].values
        y_test = test_df_mod['is_goal'].values
        
        # Fit
        clf = RandomForestClassifier(
            n_estimators=conf.n_estimators,
            max_depth=conf.max_depth,
            min_samples_leaf=conf.min_samples_leaf,
            random_state=random_state,
            n_jobs=-1  # Use all cores
        )
        clf.fit(X_train, y_train)
        models[conf.name] = clf
        
        # Evaluate
        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        
        # Metrics
        ll = log_loss(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        
        results.append({
            'Model': conf.name,
            'Log Loss': ll,
            'ROC AUC': auc,
            'Brier': brier,
            'Accuracy': acc,
            'Features': str(final_feats),
            'N Features': len(final_feats)
        })
        print(f"  -> Log Loss: {ll:.4f}, AUC: {auc:.4f}")
        
    return models, pd.DataFrame(results).sort_values('Log Loss')


def get_clf(out_path: str = 'analysis/xgs/xg_model.joblib', behavior: str = 'load', *,
            csv_path: str = 'data/processed/20252026/20252026.csv',
            n_estimators: int = 200,
            features: list = None,
            random_state: int = 42):

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
        features = ['distance', 'angle_deg', 'game_state', 'is_net_empty', 'shot_type']

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
                     csv_path: str = 'data/processed/20252026/20252026.csv',
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
        features = ['distance', 'angle_deg', 'game_state', 'is_net_empty', 'shot_type']

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


def load_data(path: str = 'data/processed/20252026/20252026.csv'):
    """Load the season CSV and return a cleaned DataFrame.

    Parameters
    - path: CSV path (default 'data/processed/20252026/20252026.csv')
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
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=1)
        clf.fit(X_train, y_train)
        return clf, X_test, y_test

    # Progress mode: train in chunks using warm_start to reveal progress
    if RandomForestClassifier is None:
        raise RuntimeError('scikit-learn is required to run training with progress.')

    # Determine chunk size (at most progress_steps updates)
    steps = max(1, int(progress_steps))
    chunk = max(1, n_estimators // steps)
    clf = RandomForestClassifier(n_estimators=0, warm_start=True, random_state=random_state, n_jobs=1)
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
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=1)
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

    plot_calibration(y_test, y_prob, path='analysis/xgs/xg_likelihood.png',
                     n_bins= 10)

    metrics = {
        'accuracy': accuracy,
        'log_loss': logloss,
        'roc_auc': rocauc,
        'brier': brier,
    }

    return y_prob, y_pred, metrics





def plot_calibration(y_test, y_prob, path: str = 'analysis/xgs/xg_likelihood.png', n_bins: int = 10):
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




def debug_model(clf_or_models, feature_cols=None, goal_side: str = 'left',
                x_res: float = 2.0, y_res: float = 2.0,
                out_path: str = 'analysis/xgs/xg_heatmap.png', cmap='viridis',
                alpha: float = 0.8, verbose: bool = True,
                game_state_values=None, is_net_empty_values=None,
                categorical_levels_map: dict = None,
                fixed_category_values: dict = None,
                interactive: bool = False,
                model_configs: Dict[str, ModelConfig] = None):
    """Simulate shots across the rink, predict xG using clf, and plot heatmaps.
    Supports single model or dictionary of models.
    
    Parameters
    - clf_or_models: 
        Single classifier OR 
        Dictionary {model_name: clf}
    - feature_cols: 
        (Legacy/Single) List of feature names. 
        Ignored if model_configs is provided.
    - model_configs: Dict {model_name: ModelConfig} mapping for feature info per model.
    """
    import matplotlib.pyplot as plt
    try:
        from .rink import draw_rink, rink_half_height_at_x, rink_bounds, rink_goal_xs
    except ImportError:
        try:
            from puck.rink import draw_rink, rink_half_height_at_x, rink_bounds, rink_goal_xs
        except ImportError:
            from rink import draw_rink, rink_half_height_at_x, rink_bounds, rink_goal_xs

    # Normalize inputs to handle multiple models
    if isinstance(clf_or_models, dict):
        models = clf_or_models
    else:
        models = {'Default': clf_or_models}

    # If configs not provided, infer minimal config for valid models
    if model_configs is None:
        model_configs = {}
        for name in models:
            # Fallback to passed feature_cols or default
            fcols = feature_cols
            if fcols is None:
                fcols = ['distance', 'angle_deg']
            model_configs[name] = ModelConfig(name=name, features=fcols)

    # Validate that we have config for every model
    possible_models = list(models.keys())
    for name in possible_models:
        if name not in model_configs:
            # should not happen given logic above
            model_configs[name] = ModelConfig(name=name, features=['distance', 'angle_deg'])

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
            return None
        base = feature_name[:-5]
        if categorical_levels_map and base in categorical_levels_map:
            levels = list(categorical_levels_map[base])
            try:
                return int(levels.index(str(cat_value)))
            except ValueError:
                if '__other__' in levels:
                    return int(levels.index('__other__'))
                return 0
        try:
            return int(cat_value)
        except Exception:
            return 0

    from itertools import product
    combos = list(product(game_state_values, is_net_empty_values))
    
    # Iterate over ALL models and compute heatmaps
    for model_name, clf in models.items():
        conf = model_configs.get(model_name)
        model_feats = conf.features if conf else feature_cols
        
        if verbose:
            print(f"debug_model: computing heatmaps for '{model_name}' ({len(combos)} combos)")

        for gs, nne in combos:
            # build feature matrix
            Xgrid = []
            for k in range(len(xs)):
                row_feats = []
                for f in model_feats:
                    if f == 'distance':
                        row_feats.append(dists[k])
                    elif f in ('angle_deg', 'angle'):
                        row_feats.append(angles[k])
                    elif f == 'dist_center':
                        row_feats.append(math.hypot(xs[k], ys[k]))
                    elif f == 'is_net_empty':
                        row_feats.append(int(nne))
                    elif f.endswith('_code'):
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

            try:
                probs = clf.predict_proba(Xgrid)[:, 1]
            except Exception as e:
                print(f"Error predicting for {model_name}: {e}")
                probs = np.zeros(len(Xgrid))

            # fill heat grid
            heat = np.full(XX.shape, np.nan)
            for (i, j), p in zip(coord_indices, probs):
                heat[i, j] = p

            # Store result key: (model_name, gs, nne)
            results[(model_name, gs, nne)] = heat

    # Just calc global max for scaling roughly
    all_max = 0.0
    for h in results.values():
        try:
            mv = float(np.nanmax(h))
            if mv > all_max: all_max = mv
        except Exception: pass
        
    vmin, vmax = 0.0, max(all_max, 0.001)

    # Save static images: only for the FIRST model in the list to preserve backward compatibility behavior
    # or save all? Let's save all with prefix.
    for model_name in models:
        for gs, nne in combos:
            heat = results.get((model_name, gs, nne))
            if heat is None: continue
            
            fig, ax = plt.subplots(figsize=(8, 4.5))
            draw_rink(ax=ax)
            extent = (gx[0] - x_res / 2.0, gx[-1] + x_res / 2.0, gy[0] - y_res / 2.0, gy[-1] + y_res / 2.0)
            im = ax.imshow(heat, extent=extent, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, zorder=1)
            ax.set_title(f'{model_name} xG — gs: {gs} | net: {nne}', fontsize=10)
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Construct logical filename
            import os
            base, ext = os.path.splitext(out_path)
            if ext == '': ext = '.png'
            safe_name = model_name.replace(' ', '_').lower()
            gs_tag = str(gs).replace(' ', '_')
            save_path = f"{base}_{safe_name}_gs-{gs_tag}_net-{nne}{ext}"
            
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(save_path, dpi=150)
            plt.close(fig)

    # Interactive Mode
    if interactive:
        try:
            import matplotlib as mpl
            from matplotlib import widgets
            # Checking backend... (omitted detailed check for brevity, assuming environment is correct)
            
            fig, ax = plt.subplots(figsize=(10, 5)) 
            # Margin for controls on left
            fig.subplots_adjust(left=0.25, right=0.95, top=0.92)

            extent = (gx[0] - x_res / 2.0, gx[-1] + x_res / 2.0, gy[0] - y_res / 2.0, gy[-1] + y_res / 2.0)

            # Initial state
            init_model = possible_models[0]
            init_gs = game_state_values[0]
            init_net = is_net_empty_values[0]
            
            heat0 = results.get((init_model, init_gs, init_net), np.full(XX.shape, np.nan))
            im = ax.imshow(heat0, extent=extent, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, zorder=1)
            title = ax.set_title(f'{init_model} — {init_gs} | net: {init_net}', fontsize=10)
            ax.axis('off')
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('xG probability')

            # Controls
            # Model Selector
            model_ax = plt.axes((0.02, 0.65, 0.18, 0.25))
            model_ax.set_title('Model', fontsize=10)
            r_model = widgets.RadioButtons(model_ax, possible_models, active=0)

            # Game State Selector
            gs_labels = [str(v) for v in game_state_values]
            gs_ax = plt.axes((0.02, 0.35, 0.18, 0.25))
            gs_ax.set_title('Game State', fontsize=10)
            r_gs = widgets.RadioButtons(gs_ax, gs_labels, active=0)

            # Empty Net Selector
            net_labels = [str(v) for v in is_net_empty_values]
            net_ax = plt.axes((0.02, 0.10, 0.18, 0.20))
            net_ax.set_title('Empty Net', fontsize=10)
            r_net = widgets.RadioButtons(net_ax, net_labels, active=0)
            
            selected = {'model': init_model, 'gs': init_gs, 'net': init_net}

            def update_plot():
                m = selected['model']
                g = selected['gs']
                n = selected['net']
                h = results.get((m, g, n), np.full(XX.shape, np.nan))
                im.set_data(h)
                title.set_text(f'{m} — {g} | net: {n}')
                fig.canvas.draw_idle()

            def on_model(label):
                selected['model'] = label
                update_plot()
            def on_gs(label):
                # map label back to value
                try: idx = gs_labels.index(label); val = game_state_values[idx]
                except: val = label
                selected['gs'] = val
                update_plot()
            def on_net(label):
                try: idx = net_labels.index(label); val = is_net_empty_values[idx]
                except: val = int(label)
                selected['net'] = val
                update_plot()

            r_model.on_clicked(on_model)
            r_gs.on_clicked(on_gs)
            r_net.on_clicked(on_net)
            
            print('Interactive mode: Select Model, Game State, and Net Status.')
            plt.show(block=True)
            
        except Exception as e:
            print(f"Interactive mode failed: {e}")

    return results

def analyze_game(game_id, clf=None):
    try:
        from . import nhl_api
    except ImportError:
        nhl_api = None
    from . import parse
    # Default CSV and feature set for analysis; ensure final_features available
    csv_path = 'data/processed/20252026/20252026.csv'
    features = ['distance', 'angle_deg', 'game_state', 'is_net_empty']

    if clf is None:
        # Try to load a persisted classifier; if that fails, train
        model_path = 'web/static/xg_model.joblib'
        if not model_path:
            model_path = 'web/static/xg_model.joblib'
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
    # usage: python fit_xgs.py [--interactive]
    interactive_mode = '--interactive' in sys.argv
    
    # 1. Load Data
    print("Loading all available season data...")
    try:
        df_all = load_all_seasons_data()
    except Exception as e:
        print(f"Data loading failed: {e}")
        sys.exit(1)
        
    # Ensure is_goal exists for stratification
    if 'is_goal' not in df_all.columns and 'event' in df_all.columns:
        df_all['is_goal'] = (df_all['event'] == 'goal').astype(int)

    # 2. Define Models
    # Current Baseline: 500 trees, standard features
    baseline_conf = ModelConfig(
        name='Baseline',
        features=['distance', 'angle_deg', 'game_state', 'is_net_empty', 'shot_type'],
        n_estimators=500,
        description="Standard Random Forest on distance/angle/context/shot_type"
    )

    configs = [baseline_conf]

    # 3. Train & Compare
    # Split separately here to ensure all models see same data
    train_df, test_df = train_test_split(df_all, test_size=0.2, random_state=42, stratify=df_all['is_goal'])
    
    models, results = compare_models(configs, train_df, test_df)
    
    print("\n--- Model Comparison Results ---")
    print(results.to_string(index=False))
    
    # 4. Save Baseline (Legacy Support)
    # The 'Baseline' model is what we want to use for the app by default
    if 'Baseline' in models:
        clf = models['Baseline']
        # Save to analysis location
        model_path = 'analysis/xgs/xg_model.joblib'
        meta_path = model_path + '.meta.json'
        
        try:
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(clf, model_path)
            
            # Save metadata
            # For get_clf to work perfectly with 'load', it expects 'final_features' to be the processed column list.
            # We can quickly generate it:
            _, final_features, cat_map = clean_df_for_model(train_df.head(10), baseline_conf.features)
                
            meta = {
                'final_features': final_features, 
                'categorical_levels_map': cat_map,
                'model_config': baseline_conf.to_dict()
            }
            
            with open(meta_path, 'w', encoding='utf-8') as fh:
                json.dump(meta, fh)
            print(f"\nSaved Baseline model to {model_path}")
            
            # Also copy to static for web app if needed? 
            # Previous code referenced 'web/static/xg_model.joblib' in analyze_game?
            # analyze_game defaults to 'web/static/xg_model.joblib'
            # Let's save there too to be helpful.
            web_path = 'web/static/xg_model.joblib'
            Path(web_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(clf, web_path)
            with open(web_path + '.meta.json', 'w', encoding='utf-8') as fh:
                json.dump(meta, fh)
                
        except Exception as e:
            print(f"Failed to save models: {e}")

    # 5. Debug / Interactive
    print("\nGenerating heatmaps...")
    configs_map = {c.name: c for c in configs}
    debug_model(models, model_configs=configs_map, interactive=interactive_mode)

