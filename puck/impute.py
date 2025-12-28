
import pandas as pd
import numpy as np
import math
import sys
import os


try:
    from .rink import calculate_distance_and_angle
except ImportError:
    try:
        from rink import calculate_distance_and_angle
    except ImportError:
        # Fallback for scripts running from project root
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        try:
            from puck.rink import calculate_distance_and_angle
        except ImportError:
            # Final fallback: define it or error? 
            # Re-defining here to be safe if all imports fail
            def calculate_distance_and_angle(x, y, goal_x, goal_y=0.0):
                distance = math.hypot(x - goal_x, y - goal_y)
                vx, vy = x - goal_x, y - goal_y
                if goal_x < 0: rx, ry = 0.0, 1.0
                else: rx, ry = 0.0, -1.0
                cross = rx * vy - ry * vx
                dot = rx * vx + ry * vy
                angle_deg = (-math.degrees(math.atan2(cross, dot))) % 360.0
                return distance, angle_deg

def calculate_geometry(df_in: pd.DataFrame, x_col='x', y_col='y', net_x=89, net_y=0):
    """
    Recalculate distance and angle for a given set of x,y coordinates relative to the net.
    Returns Series for distance and angle.
    """
    # Standard puck geometry logic from rink.py:
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
    
    # Deduce Net location per row (89, 0) or (-89, 0) based on existing distance.
    # Blocked shots in our data already have 'distance' calculated relative to the correct goal.
    if 'distance' in df_out.columns:
        d1 = np.sqrt((bx - 89)**2 + by**2)
        d2 = np.sqrt((bx + 89)**2 + by**2)
        net_x = np.where(np.abs(d1 - df_out.loc[mask_blocked, 'distance']) < np.abs(d2 - df_out.loc[mask_blocked, 'distance']), 89, -89)
    else:
        # Fallback: Guess based on x coordinate if distance is missing
        net_x = np.where(bx > 0, 89, -89)
    
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
    mask_valid_coords = mask_blocked & bx.notna() & by.notna()
    
    if mask_valid_coords.any():
        df_out.loc[mask_valid_coords, 'imputed_x'] = ox[mask_valid_coords]
        df_out.loc[mask_valid_coords, 'imputed_y'] = oy[mask_valid_coords]
        
        # Determine net_x for the valid updates for recalculation
        valid_net_x = net_x[mask_valid_coords.loc[mask_blocked].values] if isinstance(net_x, np.ndarray) else net_x
        
        # Recalculate geometry ONLY for valid updates
        # Since net_x varies, we can't use the vectorized helper as easily if it assumes fixed net.
        # But calculate_geometry is a helper in this file that can take net_x.
        
        # Let's update calculate_geometry to handle varying net_x if needed, 
        # or just loop if small or use row-wise net_x.
        
        # The easiest is to use row-wise net_x in a small loop or map for recalculation.
        res = df_out.loc[mask_valid_coords].apply(
            lambda r: calculate_distance_and_angle(
                r['imputed_x'], r['imputed_y'], 
                89 if np.abs(np.sqrt((r['imputed_x']-89)**2 + r['imputed_y']**2) - r['distance']) < 15 else -89,
                0
            ), axis=1
        )
        # Note: comparison above is slightly tricky as 'distance' was the OLD distance.
        # Better: use the net_x we already deduced.
        
        # Re-deduce or pass through net_x? 
        # Let's just use the net_x we found earlier.
        
        blocks_idx = df_out.index[mask_valid_coords]
        for idx in blocks_idx:
            # Re-run same logic to find goal for this specific row
            r = df_out.loc[idx]
            g_x = 89 if np.abs(np.sqrt((r[x_col]-89)**2 + r[y_col]**2) - r['distance']) < np.abs(np.sqrt((r[x_col]+89)**2 + r[y_col]**2) - r['distance']) else -89
            new_d, new_a = calculate_distance_and_angle(r['imputed_x'], r['imputed_y'], g_x, 0)
            df_out.at[idx, 'distance'] = new_d
            df_out.at[idx, 'angle_deg'] = new_a
    
    return df_out
