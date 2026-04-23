"""fit_xgs.py

Simple, well-documented script to fit a lightweight xG-like model (shot -> goal
probability) using one primary feature (distance to goal). The goal is clarity
and a good starting point for extension.

Usage (from project root):
    python fit_xgs.py

The script will read 'data/processed/20252026/20252026.csv' by default, train a Random Forest on
`dist_center` -> `is_goal`, print evaluation metrics, and save a calibration
plot to `os.path.join(ANALYSIS_DIR, 'xgs/xg_likelihood.png')`.
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
import os
print("DEBUG: LOADED PUCK.FIT_XGS modification check")
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
    from sklearn.calibration import calibration_curve
except Exception:  # pragma: no cover - graceful fallback for environment missing sklearn
    RandomForestClassifier = None
    train_test_split = None
    accuracy_score = log_loss = roc_auc_score = brier_score_loss = None
    calibration_curve = None

# Import Nested Model
try:
    # Try relative import first (module mode)
    from .fit_nested_xgs import NestedXGClassifier
except ImportError:
    try:
        # Try direct import (script mode, assuming dir is in path)
        import fit_nested_xgs
        NestedXGClassifier = fit_nested_xgs.NestedXGClassifier
    except ImportError as e:
        print(f"Warning: Could not import NestedXGClassifier: {e}")
        NestedXGClassifier = None

# Import XGBoost Nested Model
try:
    from . import fit_xgboost_nested
    XGBNestedXGClassifier = fit_xgboost_nested.XGBNestedXGClassifier
except ImportError:
    try:
        import fit_xgboost_nested
        XGBNestedXGClassifier = fit_xgboost_nested.XGBNestedXGClassifier
    except ImportError:
        XGBNestedXGClassifier = None

# Import Config for valid Data Directory
try:
    from . import config as puck_config
except ImportError:
    try:
        import config as puck_config
    except ImportError:
        # Provide a dummy config if strictly standalone and config missing (rare)
        class DummyConfig:
            DATA_DIR = 'data'
            ANALYSIS_DIR = 'analysis'
        puck_config = DummyConfig()

try:
    from . import nhl_api
except ImportError:
    try:
        import nhl_api
    except ImportError:
        nhl_api = None

# --- Simple module-level caching for the trained classifier ---
_GLOBAL_CLF = None
_GLOBAL_FINAL_FEATURES = None
_GLOBAL_CATEGORICAL_LEVELS_MAP = None

class SingleXGClassifier:
    """Wrapper for standard RandomForest to align interface with NestedXGClassifier.
    Accepts DataFrame for prediction and enforces exclusion of blocked shots.
    """
    def __init__(self, clf, features: List[str], raw_features: List[str] = None):
        self.clf = clf
        self.features = features # Encoded features
        self.raw_features = raw_features or ['distance', 'angle_deg', 'game_state', 'shot_type']
        
    def predict_proba(self, X):
        """
        Args:
            X: pd.DataFrame (numeric/encoded features) or np.ndarray
        """
        import pandas as pd
        import numpy as np
        
        # If array, assume it matches features order and just predict
        if isinstance(X, (np.ndarray, np.generic)):
            return self.clf.predict_proba(X)
            
        if not isinstance(X, pd.DataFrame):
             raise ValueError("Input must be DataFrame or numpy array.")
             
        # Enforce Blocked Shot = 0.0 Logic?
        # We need 'event' column to do this. 
        # But 'clean_df_for_model' might have dropped it or it might not be in 'features'.
        # For safety/performance in heavy loops (like analyze.py), we assume X is the MODEL input.
        # But if 'event' is in columns, we can use it.
        
        mask_blocked = None
        if 'event' in X.columns:
            mask_blocked = (X['event'] == 'blocked-shot')
        elif 'event_code' in X.columns:
            # If event is encoded? 'clean_df_for_model' doesn't usually encode event unless requested.
            pass
            
        # Ensure is_home is present
        if 'is_home' not in X.columns and 'team_id' in X.columns and 'home_id' in X.columns:
            X['is_home'] = (X['team_id'].astype(str) == X['home_id'].astype(str)).astype(int)
        elif 'is_home' not in X.columns:
            X['is_home'] = 0

        # Extract features for RF
        # Ensure columns exist
        missing = [f for f in self.features if f not in X.columns]
        if missing:
             # Try to generate missing encoded columns if raw columns exist
             for m in missing:
                 if m.endswith('_code'):
                     base = m[:-5]
                     if base in X.columns:
                         # We need the categorical map to encode
                         # But we don't have it here easily unless we store it in self.
                         pass
             
             missing_after = [f for f in self.features if f not in X.columns]
             if missing_after:
                 raise KeyError(f"Missing features: {missing_after}")
             
        vals = X[self.features].values
        probs = self.clf.predict_proba(vals)
        
        # Override blocked shots to 0.0
        if mask_blocked is not None and mask_blocked.any():
            # Class 0 = Not Goal? Class 1 = Goal?
            # probs is [n_samples, n_classes]. 
            # We want to set Goal Prob (col 1) to 0.0. 
            # And Non-Goal (col 0) to 1.0.
            probs[mask_blocked, :] = 0.0
            if probs.shape[1] > 1:
                probs[mask_blocked, 0] = 1.0 # P(No Goal) = 1
                # P(Goal) matches 0.0 already
            
        return probs

    def predict(self, X):
         probs = self.predict_proba(X)
         return np.argmax(probs, axis=1)

@dataclass
class ModelConfig:
    """Configuration for an xG model variant."""
    name: str
    features: List[str]
    feature_set_name: Optional[str] = None
    n_estimators: int = 500  # Increased default for better baseline performance
    max_depth: Optional[int] = None
    min_samples_leaf: int = 1
    description: str = ""
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != 'name'}

def load_all_seasons_data(base_dir: str = None, seasons: list = None, min_season: int = None) -> pd.DataFrame:
    """Load and concatenate all season CSVs found in data/{season}.
    
    Args:
        base_dir: Base directory to look for data.
        seasons: Optional list of season strings/ints to load.
        min_season: Optional minimum season integer to load (inclusive).
    """
    if base_dir is None:
        base_dir = puck_config.DATA_DIR

    base_path = Path(base_dir)

    # 1. If not found in CWD, try Project Root (robust to script location)
    if not base_path.exists() and not base_path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        alt_path = project_root / base_dir
        if alt_path.exists():
            print(f"Found data directory at: {alt_path}")
            base_path = alt_path

    if not base_path.exists():
        # 2. Fallback check for single file in project root
        project_root = Path(__file__).resolve().parent.parent
        fallback = project_root / 'data/20252026/20252026_df.csv'
        if fallback.exists():
            print(f"Data directory generic load failed, but found specific season file: {fallback}")
            return pd.read_csv(fallback)
            
        raise FileNotFoundError(f"Data directory not found. Looked for '{base_dir}' in CWD ({Path.cwd()}) and Project Root ({project_root}).")        
    frames = []
    loaded_stems = set()
    # 1. Look for files directly in base_path matching {year}.csv or {year}_df.csv
    for item in base_path.iterdir():
        if item.is_file() and (item.name.endswith('.csv')):
            # check if starts with digit year
            stem = item.stem
            year_key_str = stem[:-3] if stem.endswith('_df') else stem
            if year_key_str.isdigit():
                year_key = int(year_key_str)
                
                # Filter by seasons/min_season
                if seasons is not None and year_key not in [int(s) for s in seasons]:
                    continue
                if min_season is not None and year_key < int(min_season):
                    continue

                print(f"Loading season file: {item.name}...")
                try:
                    df = pd.read_csv(item)
                    frames.append(df)
                    loaded_stems.add(year_key_str)
                    print(f"DEBUG: Step 1 loaded {year_key_str} from {item.name}")
                    continue # already handled
                except Exception as e:
                    print(f"Failed to load {item}: {e}")

    # 2. Look for {year}/{year}_df.csv structure
    # 2. Look for nested structure using glob (more robust)
    # 2. Look for nested structure using glob (more robust)
    print(f"DEBUG: Globbing {base_path} for */*.csv...")
    # Matches data/20142015/20142015_df.csv or data/20142015/20142015.csv
    for csv_path in base_path.glob("*/*.csv"):
         year_search_str = csv_path.parent.name
         # Strict check: Must be 8 digits (e.g. 20142015)
         if not (year_search_str.isdigit() and len(year_search_str) == 8):
             # vprint(f"DEBUG: Skipping non-season dir {year_search}")
             continue

         year_search = int(year_search_str)
         
         # Filter by seasons/min_season
         if seasons is not None and year_search not in [int(s) for s in seasons]:
             continue
         if min_season is not None and year_search < int(min_season):
             continue

         if year_search_str in loaded_stems:
             print(f"DEBUG: Skipping {year_search_str} (already loaded flat)")
             continue
         
         print(f"DEBUG: Compiling {csv_path}...")
         try:
             df = pd.read_csv(csv_path)
             frames.append(df)
             loaded_stems.add(year_search)
         except Exception as e:
             print(f"Failed to load {csv_path}: {e}")
    
    if not frames:
        print("No season data found in data/.")
        raise FileNotFoundError("No season data found.")

    full_df = pd.concat(frames, ignore_index=True)
    print(f"Total loaded rows: {len(full_df)}")

    # Enrich with Handedness
    full_df = enrich_data_with_bios(full_df)

    return full_df

def enrich_data_with_bios(df: pd.DataFrame) -> pd.DataFrame:
    """Add player handedness (shoots_catches) and role (shooter_role) to the DataFrame."""
    if nhl_api and 'player_id' in df.columns and 'game_id' in df.columns:
        # print("Enriching with player bios...")
        try:
            # derive season start year from first 4 chars of game_id
            df['temp_season_start'] = df['game_id'].astype(str).str[:4]
            # filter out non-digit
            # We work on a copy to determine unique seasons, but map back to original
            starts = df['temp_season_start']
            # handle potential non-numeric or NaN
            mask = starts.astype(str).str.isdigit()
            unique_starts = starts[mask].astype(int).unique()
            
            master_map = {}
            for start_year in unique_starts:
                # Basic sanity check on year
                if start_year < 1900 or start_year > 2100:
                    continue
                season_str = f"{start_year}{start_year + 1}"
                bios = nhl_api.get_season_player_bios(season_str)
                master_map.update(bios)
            
            # Helper to safely get value from nested map
            def get_bio_val(pid_val, field, default=None):
                if pd.isna(pid_val): return default
                try:
                    # IDs clean up (handle float/str/int)
                    # The map keys are strings (e.g. "8478402")
                    clean_id = str(int(float(pid_val)))
                except:
                    clean_id = str(pid_val)
                    
                entry = master_map.get(clean_id)
                if not entry:
                    return default
                return entry.get(field, default)

            # 1. Handedness
            df['shoots_catches'] = df['player_id'].apply(lambda x: get_bio_val(x, 'shootsCatches', 'L'))
            
            # 2. Shooter Role (F vs D)
            # Map positionCode to Role
            def map_role(pos_code):
                if not pos_code: return 'F' # Default to F
                if pos_code == 'D':
                    return 'D'
                return 'F' # C, L, R, G -> F

            df['shooter_role'] = df['player_id'].apply(lambda x: map_role(get_bio_val(x, 'positionCode')))
            
            if 'temp_season_start' in df.columns:
                df.drop(columns=['temp_season_start'], inplace=True)
            # print("Bio enrichment complete.")
            
        except Exception as e:
            print(f"Warning: Bio enrichment failed: {e}")
            if 'shoots_catches' not in df.columns:
                df['shoots_catches'] = 'L'
            if 'shooter_role' not in df.columns:
                df['shooter_role'] = 'F'
    else:
        # ensuring columns exist if we can't enrich
        if 'shoots_catches' not in df.columns:
             df['shoots_catches'] = 'L'
        if 'shooter_role' not in df.columns:
             df['shooter_role'] = 'F'
             
    return df

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


_CLF_MEM_CACHE = {}

def get_clf(out_path: str = None, behavior: str = 'load', *,
            model_type: str = 'single',
            csv_path: str = None,
            n_estimators: int = 200,
            max_depth: int = None,
            min_samples_leaf: int = 1,
            features: list = None,
            feature_set_name: str = None,
            random_state: int = 42,
            data_df: pd.DataFrame = None,
            **kwargs):
    """Train or load a RandomForest classifier for xG.

    Parameters
    - out_path: path to save/load the classifier. If None, defaults based on model_type.
    - behavior: 'train' to train & save, 'load' to load from disk
    - model_type: 'single' (default) or 'nested'. Used to determine default path.
    - csv_path/features/random_state/n_estimators: training params used when behavior='train'
    - data_df: Optional pre-loaded DataFrame to use if behavior='train'
    - kwargs: Additional params for RandomForestClassifier (e.g. max_features)
    """
    # normalize behavior
    b = (behavior or '').strip().lower()
    if b not in ('train', 'load'):
        raise ValueError("behavior must be 'train' or 'load'")

    # Resolve default path based on type
    if out_path is None:
        if model_type == 'nested':
            out_path = os.path.join(puck_config.ANALYSIS_DIR, 'xgs', 'xg_model_xgboost_nested_20202021.joblib')
        else:
            out_path = os.path.join(puck_config.ANALYSIS_DIR, 'xgs', 'xg_model_single.joblib')

    # Check cache for 'load'
    cache_key = (out_path, model_type)
    if b == 'load' and cache_key in _CLF_MEM_CACHE:
        return _CLF_MEM_CACHE[cache_key]

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
        meta = {}
        try:
            with open(meta_path, 'r', encoding='utf-8') as fh:
                meta = json.load(fh)
                final_features = meta.get('final_features')
                categorical_levels_map = meta.get('categorical_levels_map')
        except Exception:
            # metadata missing is not fatal; caller may re-derive
            final_features = None
            categorical_levels_map = None
            
        # WRAPPER LOGIC
        if model_type == 'single' and not isinstance(clf, SingleXGClassifier):
             # Try to resolve raw features
             raw_features = meta.get('raw_features')
             if not raw_features and meta.get('feature_set_name'):
                 try:
                     from . import features as puck_features
                     raw_features = puck_features.get_features(meta['feature_set_name'])
                 except: pass
             
             clf = SingleXGClassifier(clf, final_features, raw_features=raw_features)
        
        # Update Cache
        _CLF_MEM_CACHE[cache_key] = (clf, final_features, categorical_levels_map, meta)
        return clf, final_features, categorical_levels_map

    # else: train
    # resolve features
    if features is None:
        if feature_set_name:
            try:
                from . import features as puck_features
                features = puck_features.get_features(feature_set_name)
            except ImportError:
                import features as puck_features
                features = puck_features.get_features(feature_set_name)
        else:
            features = ['distance', 'angle_deg', 'game_state', 'shot_type']
            feature_set_name = 'default'

    # load data and prepare
    if data_df is not None:
        season_df = data_df
    else:
        season_df = load_data(csv_path)

    season_model_df, final_features, categorical_levels_map = clean_df_for_model(season_df, features)

    # fit model
    clf, X_test, y_test = fit_model(season_model_df, feature_cols=final_features,
                                    random_state=random_state, n_estimators=n_estimators,
                                    max_depth=max_depth, min_samples_leaf=min_samples_leaf, **kwargs)

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
        meta = {
            'final_features': final_features, 
            'categorical_levels_map': categorical_levels_map,
            'feature_set_name': feature_set_name,
            'model_type': model_type,
            'raw_features': features
        }
        with open(meta_path, 'w', encoding='utf-8') as fh:
            json.dump(meta, fh)
    except Exception as e:
        print(f"Warning: failed to save model or metadata to {out_path}: {e}")

    # Re-wrap if training
    if model_type == 'single' and not isinstance(clf, SingleXGClassifier):
         clf = SingleXGClassifier(clf, final_features, raw_features=features)

    # Update Cache
    _CLF_MEM_CACHE[cache_key] = (clf, final_features, categorical_levels_map, meta)
    return clf, final_features, categorical_levels_map

# --- end of module-level caching helpers ---

def get_or_train_clf(force_retrain: bool = False,
                     csv_path: str = None,
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
        features = ['distance', 'angle_deg', 'game_state', 'shot_type']

    # Train a fresh model and cache metadata
    season_df = load_data(csv_path)
    season_model_df, final_features, categorical_levels_map = clean_df_for_model(season_df, features)

    clf, _, _ = fit_model(season_model_df, feature_cols=final_features,
                          random_state=random_state, n_estimators=n_estimators)

    _GLOBAL_CLF = clf
    _GLOBAL_FINAL_FEATURES = final_features
    _GLOBAL_CATEGORICAL_LEVELS_MAP = categorical_levels_map
    return _GLOBAL_CLF, _GLOBAL_FINAL_FEATURES, _GLOBAL_CATEGORICAL_LEVELS_MAP

def clean_df_for_model(df: pd.DataFrame, feature_cols, fixed_categorical_levels: dict = None, encode_method: str = 'integer'):
    """Filter events, encode categorical features, and coerce types.
    
    Args:
        encode_method (str): 'integer' (default) for label encoding (old behavior), 
                             'none' to preserve raw categorical columns (for Nested/OHE models).
                             'onehot' could be added if needed, but 'none' lets downstream handle it.
    """
    shot_attempt_types = ['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot']
    df = df[df['event'].isin(shot_attempt_types)].copy()
    
    # EXCLUDE EMPTY NET SHOTS
    # We do not want to calculate xG for shots on an empty net.
    if 'is_net_empty' in df.columns:
        # Check against 1, True, or '1'
        # Safest way is to convert to numeric, fill 0, check != 1
        # But usually it's integer 0/1. 
        # Let's filter out where is_net_empty is thruthy
        mask_empty = (df['is_net_empty'] == 1) | (df['is_net_empty'] == True)
        if mask_empty.any():
            # print(f"clean_df_for_model: Filtering {mask_empty.sum()} empty net shots.")
            df = df[~mask_empty].copy()

    # EXCLUDE 1v0 and 0v1
    if 'game_state' in df.columns:
        mask_extreme = df['game_state'].isin(['1v0', '0v1'])
        if mask_extreme.any():
            df = df[~mask_extreme].copy()

    # EXCLUDE NON-REGULAR SEASON (02)
    if 'game_id' in df.columns:
        # Game IDs are YYYYTTNNNN, where TT=02 is regular season.
        df['game_id_str'] = df['game_id'].astype(str)
        # Handle cases where game_id might be malformed or too short
        mask_regular = df['game_id_str'].str.len() >= 6
        mask_regular &= df['game_id_str'].str[4:6] == '02'
        if not mask_regular.all():
            # print(f"clean_df_for_model: Filtering {len(df) - mask_regular.sum()} events from non-regular season games.")
            df = df[mask_regular].copy()
        df.drop(columns=['game_id_str'], inplace=True)

    # Ensure is_home is present
    if 'is_home' not in df.columns and 'team_id' in df.columns and 'home_id' in df.columns:
        df['is_home'] = (df['team_id'].astype(str) == df['home_id'].astype(str)).astype(int)
    elif 'is_home' not in df.columns:
        print("Warning: 'is_home' cannot be derived. Filling with 0.")
        df['is_home'] = 0

    # define is_goal as a boolean: True when event equals 'goal'
    df['is_goal'] = df['event'].eq('goal')

    # Determine which requested features are categorical (object dtype)
    # Note: If encode_method='none', we still want to identify them but NOT transform them drastically
    categorical_cols = [col for col in feature_cols if col in df.columns and df[col].dtype == object]
    categorical_dummies_map = {}
    final_feature_cols = list(feature_cols)

    # Encode categorical columns
    categorical_levels_map = {}
    if categorical_cols:
        if encode_method == 'none':
            # For 'none', we might want to fill NaNs but keep them as strings/objects
            for c in categorical_cols:
                # Ensure it exists and isn't NaN (crucial for Blocked Shots -> Unknown)
                 if c not in df.columns:
                     df[c] = ''
                 # We probably want to fillna with something safe if it's missing?
                 # analyze.py does this explicitly for shot_type -> Unknown.
                 # Let's trust caller or existing fill logic?
                 pass
            # We do NOT run one_hot_encode (which does integer coding). 
            # We leave them as is. 
            pass 
        else:
            # Default 'integer' behavior
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
    # BUT if encode_method='none', some features are STRINGS. We must NOT coerce them to numeric.
    if encode_method != 'none':
        numeric_cols = [c for c in final_feature_cols]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    else:
        # Coerce ONLY non-categorical cols
        numeric_cols = [c for c in final_feature_cols if c not in categorical_cols]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Ensure is_goal is integer 0/1
    df['is_goal'] = pd.to_numeric(df['is_goal'], errors='coerce').astype('Int64')

    # Drop rows with missing required values
    # If 'none', string cols are fine.
    missing_cols = [c for c in final_feature_cols if c not in df.columns]
    if missing_cols:
         raise KeyError(f"Missing required columns in input DataFrame: {missing_cols}. Loading form {df.columns}")

    df = df[final_feature_cols + ['is_goal']].dropna().copy()
    df['is_goal'] = df['is_goal'].astype(int)

    return df, final_feature_cols, categorical_levels_map

def one_hot_encode(df: pd.DataFrame, categorical_cols, prefix_sep: str = '_', fill_value: str = '', fixed_mappings: dict = None):
    """Encode categorical columns as integer codes instead of binary dummies.
    ...
    (No changes needed here if we just skip calling it in 'none' mode)
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



def load_data(path: str = None):
    """Load the season CSV and return a cleaned DataFrame.

    Parameters
    - path: CSV path (default uses config.DATA_DIR to find 20252026 csv)
    - feature_cols: a column name or list of column names to use as features
      (default ['dist_center', 'angle_deg', 'game_state', 'is_net_empty']). The function will try common
      alternate names if the requested columns are missing.

    Returns a DataFrame with at least the feature column and `is_goal` target.
    """
    if path is None:
        # try to find a default
        try:
            return load_all_seasons_data()
        except:
             # Default to the primary season file in DATA_DIR
             path = os.path.join(puck_config.DATA_DIR, '20252026', '20252026_df.csv')
        
    df = pd.read_csv(path)


    return df


def fit_model(
    df: pd.DataFrame,
    feature_cols=None,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int = None,
    min_samples_leaf: int = 1,
    progress: bool = False,
    progress_steps: int = 20,
    **kwargs
):
    """Fit a RandomForest on the specified feature and return the trained
    model along with the held-out test split.

    Returns (clf, X_test, y_test).
    """
    if RandomForestClassifier is None:
        raise RuntimeError('scikit-learn is required to run training. Please install scikit-learn.')

    # default feature set if not provided
    if feature_cols is None:
        feature_cols = ['distance', 'angle_deg', 'game_state']
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
        clf = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state, 
            n_jobs=1,
            **kwargs
        )
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

    plot_calibration(y_test, y_prob, path=os.path.join(puck_config.ANALYSIS_DIR, 'xgs', 'xg_likelihood.png'),
                     n_bins= 10)

    metrics = {
        'accuracy': accuracy,
        'log_loss': logloss,
        'roc_auc': rocauc,
        'brier': brier,
    }

    return y_prob, y_pred, metrics





def plot_calibration(y_test, y_prob, path: str = None, n_bins: int = 10):
    """Generate and save a simple calibration/reliability plot.

    The plot shows observed frequency vs predicted probability in bins, and a
    diagonal reference line.
    """
    if path is None:
        path = os.path.join(puck_config.ANALYSIS_DIR, 'xgs', 'xg_likelihood.png')
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
                out_path: str = None, cmap='viridis',
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
    import os
    import math
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
    if out_path is None:
        out_path = os.path.join(puck_config.ANALYSIS_DIR, 'xgs', 'xg_heatmap.png')

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
    
    # Defaults for Shot Type (if model uses it)
    shot_type_values = ['wrist', 'snap', 'slap', 'backhand', 'tip-in', 'deflected']
    default_shot_type = 'wrist'

    from itertools import product
    # We now iterate Shot Types too if the model cares?
    # Or we just add it to the GUI selectors.
    # To keep combinatorics sane, we might stick to ONE shot type for the static generation loop
    # but strictly use the interactive mode for exploring.
    # Actually, let's keep combos simple (GS + Net) but assume a fixed Shot Type for the specific loop 
    # UNLESS interactive mode requests updates.
    
    # Wait, the interactive mode uses `results` cache.
    # So `results` must contain all combos or be generated on fly.
    # Generating on fly is better for interactive mode if combos define dimensionality.
    # But `debug_model` structure pre-calculates `results`.
    # Let's add shot_type to combos only if interactive, or just use default.
    # The user wants proper GUI support.
    
    # If interactive, we will pre-calc A LOT if we multiply by shot types (6 types * 3 GS * 2 Net = 36 maps).
    # That's fast enough for modern CPUs (3000 points * 36 = 100k predictions).
    
    combos = list(product(game_state_values, is_net_empty_values, shot_type_values))
    
    # Loop
    for model_name, clf in models.items():
        conf = model_configs.get(model_name)
        model_feats = conf.features if conf else feature_cols
        
        # Check if model actually USES shot_type
        # If not, we don't need to re-calc for every shot type.
        # But for simplicity of the cache key structure, might be easier to just do it.
        # Or checking overlap.
        uses_shot_type = any('shot_type' in f for f in model_feats)
        
        # Reduced combos if shot_type not used?
        current_combos = combos
        if not uses_shot_type:
            # just use default shot type once
            # But we want the GUI to work consistently.
            # We can store result under (model, gs, net, st)
            pass

        if verbose:
            print(f"debug_model: computing heatmaps for '{model_name}' (uses_shot_type={uses_shot_type})")

        for gs, nne, st in current_combos:
            # Skip redundant calcs if model ignores shot type
            # optimization: if not uses_shot_type and st != default, just copy default result?
            # Let's just run it, it's cheap prediction.
            
            # Construct DataFrame for Vectorized Prediction
            n_pts = len(xs)
            df_grid = pd.DataFrame({
                'distance': dists,
                'angle_deg': angles,
                'game_state': [gs] * n_pts,
                'is_net_empty': [int(nne)] * n_pts,
                'shot_type': [st] * n_pts,
                'is_home': [1] * n_pts,
                # Add commonly used extra features just in case
                'dist_center': np.hypot(xs, ys)
            })
            
            # Also populate encoded columns if the model requires them
            clf_features = getattr(clf, 'features', model_feats)
            for f in clf_features:
                if f not in df_grid.columns and f.endswith('_code'):
                    base = f[:-5]
                    if base in df_grid.columns:
                        val = df_grid[base].iloc[0]
                        code = category_value_to_code(f, val)
                        df_grid[f] = code
            
            # Ensure all needed columns exist (fill NaN/0 for others)
            # This is important for some models that look for diverse columns
            for f in clf_features:
                if f not in df_grid.columns:
                    df_grid[f] = 0

            try:
                # Support both array and DF capable classifiers
                # Most robust is to pass DF if it works
                probs = clf.predict_proba(df_grid)[:, 1]
            except Exception as e:
                # Fallback to array construction if DF failed (unlikely for our classifiers)
                # print(f"DF predict failed, trying array: {e}")
                Xgrid = df_grid[clf_features].values
                try:
                    probs = clf.predict_proba(Xgrid)[:, 1]
                except Exception as e2:
                    print(f"Error predicting for {model_name}: {e2}")
                    probs = np.zeros(n_pts)

            # fill heat grid
            heat = np.full(XX.shape, np.nan)
            for (i, j), p in zip(coord_indices, probs):
                heat[i, j] = p

            # Store result key: (model_name, gs, nne, st)
            results[(model_name, gs, nne, st)] = heat


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
    # Save static images
    # Use default shot type for static save
    static_combos = list(product(game_state_values, is_net_empty_values))

    for model_name in models:
        for gs, nne in static_combos:
            st = default_shot_type
            heat = results.get((model_name, gs, nne, st))
            if heat is None: continue
            
            fig, ax = plt.subplots(figsize=(8, 4.5))
            draw_rink(ax=ax)
            extent = (gx[0] - x_res / 2.0, gx[-1] + x_res / 2.0, gy[0] - y_res / 2.0, gy[-1] + y_res / 2.0)
            im = ax.imshow(heat, extent=extent, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, zorder=1)
            ax.set_title(f'{model_name} xG\ngs: {gs} | net: {nne} | {st}', fontsize=10)
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Construct logical filename
            import os
            base, ext = os.path.splitext(out_path)
            if ext == '': ext = '.png'
            safe_name = model_name.replace(' ', '_').lower()
            gs_tag = str(gs).replace(' ', '_')
            save_path = f"{base}_{safe_name}_gs-{gs_tag}_net-{nne}_st-{st}{ext}"
            
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
            init_st = shot_type_values[0]
            
            heat0 = results.get((init_model, init_gs, init_net, init_st), np.full(XX.shape, np.nan))
            im = ax.imshow(heat0, extent=extent, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, zorder=1)
            title = ax.set_title(f'{init_model} — {init_gs} | net: {init_net} | {init_st}', fontsize=10)
            ax.axis('off')
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('xG probability')

            # Controls
            # Model Selector
            model_ax = plt.axes((0.02, 0.70, 0.18, 0.20))
            model_ax.set_title('Model', fontsize=10)
            r_model = widgets.RadioButtons(model_ax, possible_models, active=0)

            # Game State Selector
            gs_labels = [str(v) for v in game_state_values]
            gs_ax = plt.axes((0.02, 0.45, 0.18, 0.20))
            gs_ax.set_title('Game State', fontsize=10)
            r_gs = widgets.RadioButtons(gs_ax, gs_labels, active=0)

            # Shot Type Selector
            st_ax = plt.axes((0.02, 0.20, 0.18, 0.20))
            st_ax.set_title('Shot Type', fontsize=10)
            r_st = widgets.RadioButtons(st_ax, shot_type_values, active=0)

            # Empty Net Selector
            net_labels = [str(v) for v in is_net_empty_values]
            net_ax = plt.axes((0.02, 0.05, 0.18, 0.10))
            net_ax.set_title('Empty Net', fontsize=10)
            r_net = widgets.RadioButtons(net_ax, net_labels, active=0)
            
            selected = {'model': init_model, 'gs': init_gs, 'net': init_net, 'st': init_st}

            def update_plot():
                m = selected['model']
                g = selected['gs']
                n = selected['net']
                s = selected['st']
                h = results.get((m, g, n, s), np.full(XX.shape, np.nan))
                im.set_data(h)
                title.set_text(f'{m} — {g} | net: {n} | {s}')
                fig.canvas.draw_idle()

            def on_model(label):
                selected['model'] = label
                update_plot()
            def on_gs(label):
                try: idx = gs_labels.index(label); val = game_state_values[idx]
                except: val = label
                selected['gs'] = val
                update_plot()
            def on_st(label):
                selected['st'] = label
                update_plot()
            def on_net(label):
                try: idx = net_labels.index(label); val = is_net_empty_values[idx]
                except: val = int(label)
                selected['net'] = val
                update_plot()

            r_model.on_clicked(on_model)
            r_gs.on_clicked(on_gs)
            r_st.on_clicked(on_st)
            r_net.on_clicked(on_net)
            
            print('Interactive mode: Select Model, Game State, Shot Type, and Net Status.')
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
    csv_path = 'data/20252026/20252026_df.csv'
    features = ['distance', 'angle_deg', 'game_state', 'is_net_empty']

    if clf is None:
        # Try to load a persisted classifier; if that fails, train
        model_path = 'analysis/xgs/xg_model.joblib'
        if not model_path:
            model_path = 'analysis/xgs/xg_model.joblib'
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

    # 2. Prepare Data Split (Must match Baseline's training split implicitly via random_state)
    print("\nPreparing Train/Test split (random_state=42)...")
    train_df, test_df = train_test_split(df_all, test_size=0.2, random_state=42, stratify=df_all['is_goal'])

    models = {}
    results = []

    # 3. Load Baseline Model
    baseline_path = 'analysis/xgs/xg_model.joblib'
    print(f"\n--- Loading 'Baseline' from {baseline_path} ---")
    
    clf_baseline = None
    final_feats_baseline = None
    cat_map_baseline = None

    try:
        clf_baseline, final_feats_baseline, cat_map_baseline = get_clf(baseline_path, behavior='load')
        print("Baseline loaded.")
    except Exception:
        print("Baseline model not found or failed to load. Training new Baseline...")
        try:
             # Train Baseline
             # Note: get_clf will handle feature cleaning
             clf_baseline, final_feats_baseline, cat_map_baseline = get_clf(
                 baseline_path, 
                 behavior='train', 
                 features=['distance', 'angle_deg', 'game_state', 'is_net_empty'],
                 data_df=train_df, # Use our pre-split training data (get_clf will internally split it again, but that's safe)
                 n_estimators=200
             )
        except Exception as ex:
             print(f"Failed to train Baseline: {ex}")

    if clf_baseline:
        models['Baseline'] = clf_baseline
        
        # If we just trained it, cat_map_baseline is correct.
        # If we loaded it, we might want to re-learn mapping from training data to be safe, 
        # or just trust the loaded map. The original code re-learned it. 
        # But if we trained, we don't need to relearn.
        
        baseline_feats_input = ['distance', 'angle_deg', 'game_state', 'is_net_empty']
        
        # For consistency, let's map the test set using the map we have (loaded or trained)
        if not cat_map_baseline:
             # Re-learn if missing from load
             print(f"Re-learning categorical mapping from training data...")
             _, _, cat_map_baseline = clean_df_for_model(train_df.copy(), baseline_feats_input)
        
        # Now clean test_df using the mapping
        test_df_bl, _, _ = clean_df_for_model(test_df.copy(), baseline_feats_input, fixed_categorical_levels=cat_map_baseline)
        
        # Use final_feats_baseline 
        if not final_feats_baseline:
             _, final_feats_baseline, _ = clean_df_for_model(train_df.head(1).copy(), baseline_feats_input)

        X_test_bl = test_df_bl[final_feats_baseline].values
        y_test_bl = test_df_bl['is_goal'].values
        
        _, _, metrics_bl = evaluate_model(clf_baseline, X_test_bl, y_test_bl)
        metrics_bl['Model'] = 'Baseline'
        metrics_bl['Features'] = str(final_feats_baseline)
        metrics_bl['N Features'] = len(final_feats_baseline)
        results.append(metrics_bl)
        print(f"  -> Log Loss: {metrics_bl['log_loss']:.4f}, AUC: {metrics_bl['roc_auc']:.4f}")
    else:
        print("Skipping Baseline comparison.")


    # 4. Train or Load 'With Shot Type' Model
    print(f"\n--- Model: 'With Shot Type' ---")
    st_path = 'analysis/xgs/xg_model_shot_type.joblib'
    shot_type_conf = ModelConfig(
        name='With Shot Type',
        features=['distance', 'angle_deg', 'game_state', 'is_net_empty', 'shot_type'],
        n_estimators=500,
        description="Random Forest including shot_type"
    )
    
    clf_st = None
    # Try loading first
    if Path(st_path).exists():
        print(f"Found existing model at {st_path}. Loading...")
        try:
            clf_st, final_feats_st, cat_map_st = get_clf(st_path, 'load')
        except Exception as e:
            print(f"Load failed ({e}). Will retrain.")
            
    if clf_st is None:
        print("Training 'With Shot Type'...")
        # Prepare Data
        train_df_st, final_feats_st, cat_map_st = clean_df_for_model(train_df.copy(), shot_type_conf.features)
        X_train_st = train_df_st[final_feats_st].values
        y_train_st = train_df_st['is_goal'].values
        
        # Fit
        clf_st = RandomForestClassifier(
            n_estimators=shot_type_conf.n_estimators,
            random_state=42,
            n_jobs=-1
        )
        clf_st.fit(X_train_st, y_train_st)
        
        # Save immediately
        print(f"Saving 'With Shot Type' model to {st_path}...")
        try:
            Path(st_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(clf_st, st_path)
            # Save metadata
            meta_st = {
                'final_features': final_feats_st, 
                'categorical_levels_map': cat_map_st,
                'model_config': shot_type_conf.to_dict()
            }
            with open(st_path + '.meta.json', 'w', encoding='utf-8') as fh:
                json.dump(meta_st, fh)
        except Exception as e:
            print(f"Failed to save: {e}")

    models['With Shot Type'] = clf_st
    
    # Evaluate (always evaluate on current test set to ensure fair comparison)
    # We need to clean test/train to get correct columns regardless of load/train path
    # Use final_feats_st and cat_map_st from whichever path we took
    
    # Re-clean test set using known map
    test_df_st, _, _ = clean_df_for_model(test_df.copy(), shot_type_conf.features, fixed_categorical_levels=cat_map_st)
    X_test_st = test_df_st[final_feats_st].values
    y_test_st = test_df_st['is_goal'].values

    _, _, metrics_st = evaluate_model(clf_st, X_test_st, y_test_st)
    metrics_st['Model'] = 'With Shot Type'
    metrics_st['Features'] = str(final_feats_st)
    metrics_st['N Features'] = len(final_feats_st)
    results.append(metrics_st)
    print(f"  -> Log Loss: {metrics_st['log_loss']:.4f}, AUC: {metrics_st['roc_auc']:.4f}")

    # (Skip re-saving if loaded, already done above if trained)

    # 7. Train 'Nested xG' Model (Comparison)
    if NestedXGClassifier:
        print(f"\n--- Training 'Nested xG' ---")
        # Nested Model needs special data preparation: it requires 'event' column for training
        # and raw categoricals (as object) to perform its own pd.get_dummies inside fit().
        
        nested_conf_dummy = ModelConfig(
            name='Nested Prep',
            features=['distance', 'angle_deg', 'game_state', 'is_net_empty', 'is_home', 'shot_type']
        )
        
        # 1. Fill NaNs in shot_type with 'Unknown' explicitly so we know what string to look for.
        train_df_n = train_df.copy()
        test_df_n = test_df.copy()
        
        unknown_label = 'Unknown'
        train_df_n['shot_type'] = train_df_n['shot_type'].fillna(unknown_label)
        test_df_n['shot_type'] = test_df_n['shot_type'].fillna(unknown_label)
        
        # We need to filter out empty net and extreme events just like clean_df_for_model does.
        train_df_n, _, _ = clean_df_for_model(train_df_n, nested_conf_dummy.features, encode_method='none')
        test_df_n, _, _ = clean_df_for_model(test_df_n, nested_conf_dummy.features, encode_method='none')

        # Add 'event' column back (aligned via index) since clean_df_for_model usually filters it.
        # But wait, clean_df_for_model actually filters rows (like empty net), so we pull 'event'
        # from the ORIGINAL subset matching the final index.
        train_df_n['event'] = train_df.loc[train_df_n.index, 'event']
        test_df_n['event'] = test_df.loc[test_df_n.index, 'event']
        
        print(f"  Nested xG: Identified 'Unknown' shot_type implicitly as '{unknown_label}'.")

        # Instantiate & Fit
        clf_nested = NestedXGClassifier(features=nested_conf_dummy.features, n_estimators=500, random_state=42, unknown_shot_type_val=unknown_label)
        
        print("  Fitting Nested xG (Block->Accuracy->Finish)...")
        clf_nested.fit(train_df_n) 
        models['Nested xG'] = clf_nested
        
        # Evaluate
        print("  Evaluating Nested xG...")
        y_prob_nested = clf_nested.predict_proba(test_df_n)[:, 1]
        y_test_nested = test_df_n['is_goal'].values        
        auc_n = roc_auc_score(y_test_nested, y_prob_nested)
        ll_n = log_loss(y_test_nested, y_prob_nested)
        
        metrics_n = {
            'Model': 'Nested xG',
            'log_loss': ll_n,
            'roc_auc': auc_n,
            'brier': brier_score_loss(y_test_nested, y_prob_nested),
            'accuracy': accuracy_score(y_test_nested, (y_prob_nested >= 0.5).astype(int)),
            'N Features': '3 Layers',
            'Features': 'Nested(Block->Acc->Finish)'
        }
        results.append(metrics_n)
        print(f"  -> Log Loss: {ll_n:.4f}, AUC: {auc_n:.4f}")
        
        # Save Nested Model
        nested_path = 'analysis/nested_xgs/nested_xg_model.joblib'
        print(f"Saving 'Nested xG' model to {nested_path}...")
        try:
            Path(nested_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(clf_nested, nested_path)
        except Exception as e:
            print(f"Failed to save Nested model: {e}")

    else:
        print("NestedXGClassifier could not be imported. Skipping.")


    # 8. Train 'Nested XGB' Model (New)
    try:
        if XGBNestedXGClassifier:
            print(f"\n--- Training 'Nested XGBoost' ---")
            # XGBNested handles dataframe natively
            # We don't need manual encoding of columns, just raw strings
            
            # Ensure shot_type has no nan (filled with Unknown)
            train_df_xgb = train_df.copy()
            train_df_xgb['shot_type'] = train_df_xgb['shot_type'].fillna('Unknown')
            # Ensure event column exists
            
            clf_xgb = XGBNestedXGClassifier(n_estimators=500, random_state=42)
            clf_xgb.fit(train_df_xgb)
            
            # Evaluate
            y_prob_xgb = clf_xgb.predict_proba(test_df.copy())[:, 1]
            y_test_xgb = test_df['is_goal'].values
            
            auc_xgb = roc_auc_score(y_test_xgb, y_prob_xgb)
            ll_xgb = log_loss(y_test_xgb, y_prob_xgb)
            metrics_xgb = {
                'Model': 'Nested XGBoost',
                'log_loss': ll_xgb,
                'roc_auc': auc_xgb,
                'brier': brier_score_loss(y_test_xgb, y_prob_xgb),
                'accuracy': accuracy_score(y_test_xgb, (y_prob_xgb >= 0.5).astype(int)),
                'N Features': '3 Layers',
                'Features': 'Nested XGB'
            }
            results.append(metrics_xgb)
            models['Nested XGBoost'] = clf_xgb
            print(f"  -> Log Loss: {ll_xgb:.4f}, AUC: {auc_xgb:.4f}")
            
            # Save
            save_path_xgb = 'analysis/nested_xgs/xgb_nested_xg_model.joblib'
            Path(save_path_xgb).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(clf_xgb, save_path_xgb)
    
    except Exception as e:
        print(f"Failed to train XGBoost Nested: {e}")
        # import traceback
        # traceback.print_exc()

    # 9. Compare Results (Final)
    print("\n--- Final Model Comparison Results ---")
    results_df = pd.DataFrame(results)
    cols = ['Model', 'log_loss', 'roc_auc', 'brier', 'accuracy']
    print(results_df[cols].to_string(index=False))


    # 10. Generate Comparative Heatmaps
    print("\nGenerating heatmaps...")
    
    # Config map for debug_model
    configs_map = {
        'Baseline': ModelConfig(name='Baseline', features=['distance', 'angle_deg', 'game_state', 'is_net_empty']),
        'With Shot Type': shot_type_conf,
    }
    
    if 'Nested xG' in models:
        # We need to map options for it, even if we hacked the features
        configs_map['Nested xG'] = ModelConfig(name='Nested xG', features=['distance', 'angle_deg', 'game_state', 'is_net_empty', 'shot_type'])
        
    if 'Nested XGBoost' in models:
        configs_map['Nested XGBoost'] = ModelConfig(name='Nested XGBoost', features=['distance', 'angle_deg', 'game_state', 'is_net_empty', 'shot_type'])
    
    
    combined_cat_map = {}
    if cat_map_baseline: combined_cat_map.update(cat_map_baseline)
    if cat_map_st: combined_cat_map.update(cat_map_st)
    
    debug_model(models, model_configs=configs_map, categorical_levels_map=combined_cat_map, interactive=interactive_mode)


