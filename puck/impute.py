
import pandas as pd
import numpy as np
import math

def calculate_geometry(df_in: pd.DataFrame, x_col='x', y_col='y', net_x=89, net_y=0):
    """
    Recalculate distance and angle for a given set of x,y coordinates relative to the net.
    Returns Series for distance and angle.
    """
    dx = df_in[x_col] - net_x
    dy = df_in[y_col] - net_y
    dist = np.sqrt(dx**2 + dy**2)
    
    # Angle calculation
    # We want degrees, 0 at center line? Standard logic matches puck.parse?
    # Usually in this codebase: abs(degrees(atan2(y, x_dist_to_goal)))
    # But let's stick to the logic from compare_imputations which seemed to work.
    
    with np.errstate(divide='ignore', invalid='ignore'):
         angle_rad = np.arctan(np.abs(dy / dx))
         angle_deg = np.degrees(angle_rad)
    
    # Handle x=89 (dx=0) -> 90 degrees
    angle_deg = angle_deg.fillna(90.0)
    
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
    
    # Overwrite blocked
    df_out.loc[mask_blocked, 'imputed_x'] = ox
    df_out.loc[mask_blocked, 'imputed_y'] = oy
    
    # Recalculate geometry for BLOCKED shots using imputed coordinates
    # We leave non-blocked geometry alone (assuming it was correct from source)
    # UNLESS we want to be safe and recalc everything? 
    # Let's only recalc blocked rows to minimize side effects.
    
    new_dist, new_angle = calculate_geometry(df_out.loc[mask_blocked], x_col='imputed_x', y_col='imputed_y')
    
    df_out.loc[mask_blocked, 'distance'] = new_dist
    df_out.loc[mask_blocked, 'angle_deg'] = new_angle
    
    return df_out
