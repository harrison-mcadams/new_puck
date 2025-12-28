
import pandas as pd
import numpy as np
import math
import sys
import os

def calculate_geometry(df_in: pd.DataFrame, x_col='x', y_col='y', net_x=89, net_y=0):
    """
    Recalculate distance and angle for a given set of x,y coordinates relative to the net.
    Returns Series for distance and angle.
    """
    try:
        from .rink import calculate_distance_and_angle
    except ImportError:
        try:
            from rink import calculate_distance_and_angle
        except ImportError:
            # Fallback for scripts running from project root
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from puck.rink import calculate_distance_and_angle
    
    # Apply standard geometry calculation to each row
    # results = df_in.apply(lambda row: calculate_distance_and_angle(row[x_col], row[y_col], net_x, net_y), axis=1)
    
    # Efficient vectorized approach if possible, but for xG consistency we can use apply 
    # since this is usually called on chunks or during training.
    
    # Standard puck geometry logic from rink.py:
    # distance = math.hypot(x - goal_x, y - goal_y)
    # angle_deg = (-math.degrees(math.atan2(cross, dot))) % 360.0
    
    # Let's use the helper directly
    res = df_in.apply(lambda r: calculate_distance_and_angle(r[x_col], r[y_col], net_x, net_y), axis=1)
    dist = res.apply(lambda x: x[0])
    angle_deg = res.apply(lambda x: x[1])
    
    return dist, angle_deg

def impute_blocked_shot_origins(df: pd.DataFrame, method: str = 'mean_6', 
                                x_col='x', y_col='y') -> pd.DataFrame:
    """
    Returns a copy of df with 'distance' and 'angle_deg' updated for BLOCKED shots.
    The 'x' and 'y' columns in the output are NOT updated (original block location kept),
    but 'imputed_x' and 'imputed_y' are added.
    
    Methods:
    - 'mean_6': 5.64ft back-projection (HockeyViz approximation).
    - 'fixed_15': 15ft back-projection.
    """
    df_out = df.copy()
    
    # We only modify blocked shots
    mask_blocked = (df['event'] == 'blocked-shot')
    if not mask_blocked.any():
        return df_out

    # Initialize with original
    df_out['imputed_x'] = df_out[x_col]
    df_out['imputed_y'] = df_out[y_col]

    # Get coordinates of blocks
    bx = df_out.loc[mask_blocked, x_col]
    by = df_out.loc[mask_blocked, y_col]
    
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
        # Default fallback or error?
        d = 0.0
    
    # Calculate Origin
    ox = bx + (ux * d)
    oy = by + (uy * d)
    
    # Update ONLY where ox/oy are valid (not NaN)
    # If source x/y was NaN, result is NaN. We don't want to overwrite valid distance/angle with NaN 
    # (which later gets filled to 0 => High xG).
    
    mask_valid_coords = mask_blocked & bx.notna() & by.notna()
    
    if mask_valid_coords.any():
        df_out.loc[mask_valid_coords, 'imputed_x'] = ox[mask_valid_coords]
        df_out.loc[mask_valid_coords, 'imputed_y'] = oy[mask_valid_coords]
        
        # Recalculate geometry ONLY for valid updates
        new_dist, new_angle = calculate_geometry(df_out.loc[mask_valid_coords], x_col='imputed_x', y_col='imputed_y')
        
        df_out.loc[mask_valid_coords, 'distance'] = new_dist
        df_out.loc[mask_valid_coords, 'angle_deg'] = new_angle
    
    return df_out
