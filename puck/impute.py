
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
    Updates 'imputed_x', 'imputed_y', 'distance', 'angle_deg'.
    
    Logic:
    1. NON-BLOCKED SHOTS:
       - imputed_x = x
       - imputed_y = y
       - distance/angle = UNTOUCHED (keep original)
       
    2. BLOCKED SHOTS:
       - imputed_x, imputed_y = back-projected coordinates
       - distance, angle = RECALCULATED from imputed coords
    """
    df_out = df.copy()
    
    # 1. Initialize imputed cols with original
    # (Matches user requirement: imputation should return same x/y for non-blocked)
    df_out['imputed_x'] = df_out[x_col]
    df_out['imputed_y'] = df_out[y_col]

    # Mask for blocked shots
    mask_blocked = (df['event'] == 'blocked-shot')
    if not mask_blocked.any():
        return df_out

    # 2. Logic for BLOCKED only
    bx = df_out.loc[mask_blocked, x_col]
    by = df_out.loc[mask_blocked, y_col]
    
    # Need existing distance to deduce net location
    # If distance missing, we guess based on side of rink
    if 'distance' in df_out.columns:
        # Distance to (89,0) vs (-89,0)
        d1 = np.sqrt((bx - 89)**2 + by**2)
        d2 = np.sqrt((bx + 89)**2 + by**2)
        # Compare calculated distance to stored distance
        # We assume stored distance is correct-ish relative to "some" net
        old_dist = df_out.loc[mask_blocked, 'distance']
        # closer match wins
        net_x = np.where(np.abs(d1 - old_dist) < np.abs(d2 - old_dist), 89, -89)
    else:
        # Fallback: Guess based on x coordinate (standard NHL coords, +x is one side)
        # Usually positive x is offensive zone for home? It varies. 
        # But usually blocking happens in defensive zone.
        # Safe fallback: assume nearest net.
        net_x = np.where(bx > 0, 89, -89)
    
    net_y = 0
    
    # Vector from Net to Block
    vx = bx - net_x
    vy = by - net_y
    mag = np.sqrt(vx**2 + vy**2)
    
    # Unit Vector
    ux = vx / mag
    uy = vy / mag
    
    # Fill NaNs (div by zero)
    ux = ux.fillna(0)
    uy = uy.fillna(0)
    
    # Distance to project back
    # "Smooth Point Prior" Strategy:
    # For deep blocks (< 30ft), we assume the shot originated from the 'Point' or 'High Slot'.
    # We sample target distances from a Normal Distribution (mean=55ft, std=8ft) to create a natural spread.
    # This prevents artificial "walls" or detectable patterns while eliminating False Slot Shots.
    
    # NEW (Verification Step Correction):
    # If the block is very close to the goal line (e.g. mag < 30 and ux is small), purely radial projection
    # sends the imputed point to the boards (x=89, y=42). 
    # We blend the radial vector with a "Center Pull" vector for deep blocks to bias origins towards the Point.
    
    if method == 'mean_6':
        # 1. Identify "Deep Blocks" (e.g. < 30ft)
        is_deep = (mag < 30.0)
        
        # 2. Generate Target Distances (Normal Dist ~ 55ft)
        target_dists = np.random.normal(loc=55.0, scale=8.0, size=len(bx))
        
        # 3. Calculate Projection Distance
        d_proj_deep = np.maximum(5.64, target_dists - mag)
        d_proj = np.where(is_deep, d_proj_deep, 5.64)
        
        # 4. Modify Direction Vectors (Deep Blocks Only)
        # Vector pointing from Net to Center Ice (0,0) is (-sign(net_x), 0)
        # We blend the observed vector (ux, uy) with this center vector
        # Blend factor alpha depends on depth? Let's use constant 0.5 for deep blocks to ensure significant pull.
        
        # Target Vector: (-sign(net_x), 0)
        # If net_x is 89, target is (-1, 0).
        t_ux = -np.sign(net_x) 
        t_uy = 0.0
        
        # Blend Factor (0.0 = Raw, 1.0 = Pure Point)
        # Use 0.5 for deep blocks, 0.0 otherwise
        alpha = np.where(is_deep, 0.5, 0.0)
        
        # Blend
        ux_blend = (1 - alpha) * ux + alpha * t_ux
        uy_blend = (1 - alpha) * uy + alpha * t_uy
        
        # Re-Normalize
        mag_blend = np.sqrt(ux_blend**2 + uy_blend**2)
        # Handle zero mag (unlikely unless ux=1, t_ux=-1 and alpha=0.5 -> cancellations)
        # t_ux is -1. ux is usually -1 (slot) or 0 (side).
        # if ux = 1 (shot from behind net?), then -1 + 1 cancels.
        # Fallback to t_ux if blend is zero
        ux_final = np.where(mag_blend < 1e-3, t_ux, ux_blend / mag_blend)
        uy_final = np.where(mag_blend < 1e-3, t_uy, uy_blend / mag_blend)
        
    else:
        d_proj = 15.0 if method == 'fixed_15' else 0.0
        ux_final = ux
        uy_final = uy
    
    # Apply projection
    ox = bx + (ux_final * d_proj)
    oy = by + (uy_final * d_proj)
    
    # RINK BOUNDARIES (Clamp to valid ice)
    # Standard NHL Rink: X +/- 100, Y +/- 42.5
    ox = np.clip(ox, -99.0, 99.0)
    oy = np.clip(oy, -42.0, 42.0)
    
    # Update Imputed Coordinates (for BLOCKED only)
    df_out.loc[mask_blocked, 'imputed_x'] = ox
    df_out.loc[mask_blocked, 'imputed_y'] = oy
    
    # 3. Recalculate Distance & Angle (for BLOCKED only)
    # We use the SAME net_x we deduced above
    # Calculate using vectorized numpy for speed/simplicity
    
    # New vectors
    dx_new = df_out.loc[mask_blocked, 'imputed_x'] - net_x
    dy_new = df_out.loc[mask_blocked, 'imputed_y'] - net_y
    
    # New Distance
    new_dist = np.hypot(dx_new, dy_new)
    df_out.loc[mask_blocked, 'distance'] = new_dist
    
    # New Angle
    # Use standard NHL angle logic (from rink.py usually)
    # If rink.py import failed, we use fallback math.
    # Logic: Angle is degrees from center line.
    
    # We'll use apply/lambda with our helper if available, or direct math if easy.
    # The helper calculate_distance_and_angle handles the specific sign conventions.
    
    # Let's map row-wise to be safe and consistent with rink.py logic
    # We need to zip imputed_x, imputed_y, and net_x
    
    def get_new_metrics(row, nx):
        return calculate_distance_and_angle(row['imputed_x'], row['imputed_y'], nx, 0)

    # We need to align net_x with the dataframe index
    # net_x is a numpy array matching mask_blocked rows
    
    idxs = df_out[mask_blocked].index
    
    # Create temporary DF to apply over
    temp_df = df_out.loc[mask_blocked, ['imputed_x', 'imputed_y']].copy()
    temp_df['net_x'] = net_x
    
    res = temp_df.apply(lambda r: calculate_distance_and_angle(r['imputed_x'], r['imputed_y'], r['net_x'], 0), axis=1)
    
    # Update
    df_out.loc[idxs, 'distance'] = res.apply(lambda x: x[0])
    df_out.loc[idxs, 'angle_deg'] = res.apply(lambda x: x[1])

    return df_out
