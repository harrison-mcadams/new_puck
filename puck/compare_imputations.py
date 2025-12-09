import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def load_data(path_pattern: str = 'data/**/*.csv') -> pd.DataFrame:
    """Load and concatenate all available season data. Expects data/{year}/*.csv"""
    import glob
    files = glob.glob(path_pattern, recursive=True)
    
    season_files = {}
    for f in files:
        p = Path(f)
        parent_name = p.parent.name
        if parent_name.isdigit():
             if p.name.endswith('_df.csv') or p.name == f"{parent_name}.csv":
                 if parent_name not in season_files:
                     season_files[parent_name] = f
                 else:
                     current_path = season_files[parent_name]
                     if not current_path.endswith('_df.csv') and f.endswith('_df.csv'):
                         season_files[parent_name] = f

    data_files = list(season_files.values())
    
    if not data_files:
        # Fallback for dev environment testing
        fallback = 'data/20252026/20252026_df.csv'
        if Path(fallback).exists():
            data_files = [fallback]
        else:
            raise FileNotFoundError("Could not find any season CSV files.")
            
    logger.info(f"Loading data from {len(data_files)} files...")
    dfs = []
    for f in data_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to read {f}: {e}")
            
    if not dfs:
        raise ValueError("No data loaded.")
        
    full_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(full_df)} total rows.")
    return full_df

def calculate_geometry(df_in: pd.DataFrame, x_col='x', y_col='y', net_x=89, net_y=0):
    """
    Recalculate distance and angle for a given set of x,y coordinates relative to the net.
    Returns Series for distance and angle.
    """
    # Distance
    # simple euclidean to (89, 0)
    # Note: Using 'adj_x' usually aligns everything to one side, but let's assume raw X has nets at +89 and -89.
    # Actually, standardizing is safer.
    # Usually data is normalized so attacking net is always +89 or similar.
    # Let's assume absolute distance to nearest net? 
    # Or assume data is already oriented. The 'distance' column exists, so we can mimic it.
    # fit_nested_xgs uses the existing 'distance' column.
    # Standard NHL API data usually has attacking zone coordinates normalized.
    # Let's trust that Attacking Net is at X=89.
    
    dx = df_in[x_col] - net_x
    dy = df_in[y_col] - net_y
    dist = np.sqrt(dx**2 + dy**2)
    
    # Angle calculation
    # 0 degrees is straight on? Or 90?
    # Usually: abs(arctan(y/x))
    # Note: dividing by dx (which is negative).
    # Angle should be 0 at center line (y=0) and high at goal line (y large).
    # angle = |arctan(y / (x-89))|
    # Convert to degrees
    
    # Safe division
    with np.errstate(divide='ignore', invalid='ignore'):
         angle_rad = np.arctan(np.abs(dy / dx))
         angle_deg = np.degrees(angle_rad)
    
    # Handle x=89 (dx=0) -> 90 degrees
    angle_deg = angle_deg.fillna(90.0)
    
    return dist, angle_deg

def impute_origins(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Returns a copy of df with 'distance' and 'angle_deg' updated for BLOCKED shots.
    Methods:
    - 'baseline': No change (uses raw block location).
    - 'fixed_15': 15ft back-projection.
    - 'mean_6': 6ft back-projection (HockeyViz approximation).
    """
    df_out = df.copy()
    
    # We only modify blocked shots
    mask_blocked = (df['event'] == 'blocked-shot')
    
    logger.info(f"Applying imputation: {method}")
    
    # Initialize with original
    df_out['imputed_x'] = df_out['x']
    df_out['imputed_y'] = df_out['y']

    if method != 'baseline':
        # Get coordinates of blocks
        bx = df_out.loc[mask_blocked, 'x']
        by = df_out.loc[mask_blocked, 'y']
        
        # Net location (Assumed 89, 0 based on standard API normalization)
        net_x = 89
        net_y = 0
        
        # Vector from Net to Block
        vx = bx - net_x
        vy = by - net_y
        
        # Magnitude
        mag = np.sqrt(vx**2 + vy**2)
        
        # Direction Unit Vector (pointing upstream, away from net)
        ux = vx / mag
        uy = vy / mag
        
        # Handle division by zero
        ux = ux.fillna(0)
        uy = uy.fillna(0)
        
        # Determine imputation distance D
        if method == 'fixed_15':
            d = 15.0
        elif method == 'mean_6':
            d = 5.64 
        else:
            raise ValueError(f"Unknown method {method}")
        
        # Calculate Origin
        ox = bx + (ux * d)
        oy = by + (uy * d)
        
        # Overwrite blocked
        df_out.loc[mask_blocked, 'imputed_x'] = ox
        df_out.loc[mask_blocked, 'imputed_y'] = oy
    
    # Recalculate geometry for ALL rows to ensure consistency
    
    # Recalculate geometry for ALL rows to ensure consistency
    # (Avoids leakage if original 'angle_deg' has specific rounding/origin artifacts)
    new_dist, new_angle = calculate_geometry(df_out, x_col='imputed_x', y_col='imputed_y')
    
    df_out['distance'] = new_dist
    df_out['angle_deg'] = new_angle
    
    return df_out


def train_and_eval(df: pd.DataFrame, method_name: str):
    """Train Block Model and return metrics."""
    
    # 1. Prepare Data
    # Filter for used columns
    cols = ['distance', 'angle_deg', 'game_state_encoded', 'is_net_empty', 'is_blocked']
    X = df[cols[:-1]] # Features
    y = df['is_blocked']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Train
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # 3. Predict
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    # 4. Score
    auc = roc_auc_score(y_test, y_prob)
    ll = log_loss(y_test, y_prob)
    
    logger.info(f"[{method_name}] Result -> AUC: {auc:.4f} | LogLoss: {ll:.4f}")
    return auc, ll

def main():
    logger.info("Starting Imputation Comparison...")
    
    # 1. Load Data
    raw_df = load_data()
    
    # 2. Preprocess Common Features
    # Encode Game State
    raw_df['game_state'] = raw_df['game_state'].fillna('5v5')
    le = LabelEncoder()
    raw_df['game_state_encoded'] = le.fit_transform(raw_df['game_state'].astype(str))
    
    # Target
    raw_df['is_blocked'] = (raw_df['event'] == 'blocked-shot').astype(int)
    
    # Use only shot attempts (Goals, Shots, Misses, Blocks)
    # Filter: blocked-shot, shot-on-goal, goal, missed-shot
    df_attempts = raw_df[raw_df['event'].isin(['blocked-shot', 'shot-on-goal', 'goal', 'missed-shot'])].copy()
    
    logger.info(f"Total Shot Attempts: {len(df_attempts)}")
    
    results = {}
    
    # 3. Compare Methods
    methods = ['baseline', 'fixed_15', 'mean_6']
    
    for m in methods:
        # Create imputed version of dataframe
        df_imputed = impute_origins(df_attempts, m)
        auc, ll = train_and_eval(df_imputed, m)
        results[m] = {'AUC': auc, 'LogLoss': ll}
        
    print("\n--- COMPARISON RESULTS ---")
    res_df = pd.DataFrame(results).T
    print(res_df)
    
    # Determine Winner
    winner = res_df['AUC'].idxmax()
    print(f"\nWinner: {winner}")

if __name__ == "__main__":
    main()
