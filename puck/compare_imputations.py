import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import logging
import matplotlib.pyplot as plt
import os

# Ensure output directory exists
Path('analysis/imputation_research').mkdir(parents=True, exist_ok=True)

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

def calculate_geometry(df_in: pd.DataFrame, x_col='x', y_col='y'):
    """
    Recalculate distance and angle for a given set of x,y coordinates.
    Assumes standard NHL rink with nets at x = +/- 89.
    Calculates geometry to the NEAREST net (assuming that is the target/defended net).
    """
    x = df_in[x_col].values
    y = df_in[y_col].values
    
    # Identify target net based on X coordinate sign
    # If x > 0, net is at +89. If x < 0, net is at -89.
    # If x = 0, default to +89 (rare exact center ice events)
    net_x = np.sign(x) * 89
    net_x[net_x == 0] = 89
    
    dx = x - net_x
    dy = y - 0 # Net Y is always 0
    
    dist = np.sqrt(dx**2 + dy**2)
    
    # Angle calculation
    # angle = |arctan(y / dx)|
    # dx is distance from net along X axis.
    with np.errstate(divide='ignore', invalid='ignore'):
         angle_rad = np.arctan(np.abs(dy / dx))
         angle_deg = np.degrees(angle_rad)
    
    # Handle dx=0 (shot from goal line extended) -> 90 degrees
    angle_deg = np.nan_to_num(angle_deg, nan=90.0)
    
    return dist, angle_deg

def get_unblocked_density(df: pd.DataFrame):
    """
    Calculates a 2D density map of unblocked shot attempts.
    Returns density array, x_edges, y_edges.
    """
    mask_unblocked = df['event'].isin(['shot-on-goal', 'goal', 'missed-shot'])
    
    # Use only the half-rink for density calculation, as it's symmetric
    # And we'll apply it based on the side of the block
    x_coords = df.loc[mask_unblocked, 'x'].abs() # Use absolute x for density
    y_coords = df.loc[mask_unblocked, 'y']
    
    # Define bins for the half-rink
    x_bins = np.arange(0, 100, 1) # 1ft bins from 0 to 100
    y_bins = np.arange(-45, 46, 1) # 1ft bins from -45 to 45
    
    density, x_edges, y_edges = np.histogram2d(x_coords, y_coords, bins=[x_bins, y_bins], density=True)
    
    return density, x_edges, y_edges

def impute_origins(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Returns a copy of df with 'distance' and 'angle_deg' updated for BLOCKED shots.
    methods: 'baseline', 'mean_X' (e.g. mean_6, mean_15), 'data_bayes'
    """
    df_out = df.copy()
    
    # Initialize with original
    df_out['imputed_x'] = df_out['x']
    df_out['imputed_y'] = df_out['y']
    
    # We only mod blocked shots
    mask_blocked = (df['event'] == 'blocked-shot')
    
    if method == 'baseline':
        pass # No change to coords
        
    elif method.startswith('mean_') or method == 'data_bayes':
        logger.info(f"Applying imputation: {method}")
        
        # Get coordinates of blocks
        bx = df_out.loc[mask_blocked, 'x'].to_numpy()
        by = df_out.loc[mask_blocked, 'y'].to_numpy()
        
        # Dynamic Net location: +/- 89 based on block side
        net_x = np.sign(bx) * 89
        net_x[net_x == 0] = 89
        net_y = 0
        
        # Vector from Net to Block
        vx = bx - net_x
        vy = by - net_y
        
        # Magnitude & Direction
        mag = np.sqrt(vx**2 + vy**2)
        
        # Avoid div by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            ux = vx / mag
            uy = vy / mag
        ux = np.nan_to_num(ux)
        uy = np.nan_to_num(uy)
        
        if method.startswith('mean_'):
            try:
                dist_val = float(method.split('_')[1])
            except ValueError:
                raise ValueError(f"Invalid format for method {method}")
            
            # Calculate D (Fixed mean distance)
            d = np.full_like(mag, dist_val)
            
        elif method == 'data_bayes':
            # 1. Prior: Unblocked Shot Density
            density, x_edges, y_edges = get_unblocked_density(df)
            
            # 2. Candidates: 0 to 60ft upstream (finer resolution)
            candidates_d = np.linspace(0, 60, 20) # 3ft steps
            
            best_d = np.zeros_like(mag)
            best_score = np.full_like(mag, -1.0)
            
            sigma = 20.0 # Tuning parameter for "Block Gap" likelihood
            
            for dist_candidate in candidates_d:
                # Candidate coords
                cx = bx + (ux * dist_candidate)
                cy = by + (uy * dist_candidate)
                
                # Digitize to find bin indices
                # Need to use absolute x for density lookup
                x_idx = np.searchsorted(x_edges, np.abs(cx)) - 1
                y_idx = np.searchsorted(y_edges, cy) - 1
                
                # Valid indices only
                valid_mask = (x_idx >= 0) & (x_idx < density.shape[0]) & \
                             (y_idx >= 0) & (y_idx < density.shape[1])
                
                # Prior P(Shot from here)
                prior = np.zeros_like(mag)
                prior[valid_mask] = density[x_idx[valid_mask], y_idx[valid_mask]]
                
                # Likelihood P(Block | Gap) -> Gaussian Decay
                # dist_candidate is the gap distance
                likelihood = np.exp(-(dist_candidate**2) / (2 * sigma**2))
                
                # Posterior Score
                score = prior * likelihood
                
                # Update best
                better_mask = (score > best_score)
                best_d[better_mask] = dist_candidate
                best_score[better_mask] = score[better_mask]
            
            d = best_d
            
        else:
            raise ValueError(f"Unknown method {method}")
        
        # Calculate Origin
        ox = bx + (ux * d)
        oy = by + (uy * d)
        
        # Overwrite blocked
        df_out.loc[mask_blocked, 'imputed_x'] = ox
        df_out.loc[mask_blocked, 'imputed_y'] = oy
        
    else:
        # Fallback or error
        raise ValueError(f"Unknown method {method}")
        
    # Recalculate geometry for ALL rows
    new_dist, new_angle = calculate_geometry(df_out, x_col='imputed_x', y_col='imputed_y')
    df_out['distance'] = new_dist
    df_out['angle_deg'] = new_angle
    
    return df_out

def plot_imputations(df: pd.DataFrame, method: str, out_path: Path):
    """Generate diagnostic plot for imputation."""
    plt.figure(figsize=(10, 6))
    
    # 1. Plot Prior (Unblocked Shots) as background hexbin/density
    mask_unblocked = df['event'].isin(['shot-on-goal', 'goal', 'missed-shot'])
    plt.hexbin(df.loc[mask_unblocked, 'x'], df.loc[mask_unblocked, 'y'], 
               gridsize=30, cmap='Greys', mincnt=1, alpha=0.3, label='Unblocked (Prior)')
    
    # 2. Plot Blocked Shots (Imputed Locations)
    mask_blocked = (df['event'] == 'blocked-shot')
    
    # Sample clearly
    sample_df = df[mask_blocked].sample(min(1000, mask_blocked.sum()), random_state=42)
    
    plt.scatter(sample_df['imputed_x'], sample_df['imputed_y'], 
                c='blue', s=10, alpha=0.6, label=f'Blocked ({method})')
    
    # Draw Net
    plt.scatter([89], [0], c='red', marker='D', s=100, label='Net')
    
    plt.title(f"Imputation Method: {method}")
    plt.xlabel("X (ft)")
    plt.ylabel("Y (ft)")
    plt.xlim(-100, 100) # Full rink view roughly for raw coords
    plt.ylim(-45, 45)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(out_path)
    plt.close()
    logger.info(f"Saved plot: {out_path}")

def plot_comparison_arrows(raw_df, methods_data: dict, out_path: Path):
    """Plot arrows from Block -> Imputed Origin for different methods."""
    plt.figure(figsize=(12, 8))
    
    mask_blocked = (raw_df['event'] == 'blocked-shot')
    sample_indices = raw_df[mask_blocked].sample(min(50, mask_blocked.sum()), random_state=99).index
    
    # Plot rink bg proxy (Unblocked shots)
    mask_unblocked = raw_df['event'].isin(['shot-on-goal', 'goal', 'missed-shot'])
    plt.hexbin(raw_df.loc[mask_unblocked, 'x'], raw_df.loc[mask_unblocked, 'y'], 
               gridsize=30, cmap='Greys', mincnt=1, alpha=0.2, label='Density')

    colors = {'mean_20': 'blue', 'data_bayes': 'red', 'baseline': 'green'}
    
    # Loop through sampled events
    for idx in sample_indices:
        bx = raw_df.loc[idx, 'x']
        by = raw_df.loc[idx, 'y']
        
        # Plot Block
        plt.plot(bx, by, 'kx', markersize=5)
        
        # Plot arrows for each method
        for m_name, m_df in methods_data.items():
            if m_name == 'baseline': continue
            
            ix = m_df.loc[idx, 'imputed_x']
            iy = m_df.loc[idx, 'imputed_y']
            
            c = colors.get(m_name, 'purple')
            # Draw arrow
            plt.arrow(bx, by, ix-bx, iy-by, color=c, alpha=0.6, 
                      head_width=1.5, length_includes_head=True)
            
    # Dummy legends
    for m_name, c in colors.items():
        if m_name in methods_data:
             plt.plot([], [], color=c, label=m_name)
             
    plt.title("Imputation Correction Vectors (Sample)")
    plt.xlim(-100, 100)
    plt.ylim(-45, 45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logger.info(f"Saved comparison plot: {out_path}")

def train_and_eval(df: pd.DataFrame, method_name: str):
    """Train Block Model and return metrics."""
    
    # 0. Sanity Check Metric: Mean Imputed Distance for Blocked Shots
    mask_blocked = (df['event'] == 'blocked-shot')
    mean_dist_blocked = df.loc[mask_blocked, 'distance'].mean()
    
    # Plotting (skip for sweep if too many, or keep to see visual drift)
    # Keeping it because visuals are good
    plot_path = Path(f'analysis/imputation_research/plot_{method_name}.png')
    plot_imputations(df, method_name, plot_path)
    
    # 1. Prepare Data
    # Filter for used columns
    cols = ['distance', 'angle_deg', 'game_state_encoded', 'is_net_empty', 'is_blocked']
    X = df[cols[:-1]].fillna(0) # Simple fill for safety
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
    
    logger.info(f"[{method_name}] AUC: {auc:.4f} | Dist: {mean_dist_blocked:.1f}ft")
    return auc, ll, mean_dist_blocked

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
    methods_dfs = {}
    
    # 3. Compare: Baseline vs Mean_20 (Winner) vs Data_Bayes (Challenger)
    methods = ['baseline', 'mean_20', 'data_bayes']
    
    for m in methods:
        # Create imputed version of dataframe
        df_imputed = impute_origins(df_attempts, m)
        methods_dfs[m] = df_imputed # Store for comparison plot
        
        auc, ll, dist = train_and_eval(df_imputed, m)
        results[m] = {'AUC': auc, 'LogLoss': ll, 'MeanDist': dist}
        
    print("\n--- COMPARISON RESULTS ---")
    res_df = pd.DataFrame(results).T
    print(res_df)
    
    # Generate Comparison Arrows
    plot_comparison_arrows(df_attempts, methods_dfs, Path('analysis/imputation_research/comparison_arrows.png'))

if __name__ == "__main__":
    main()
