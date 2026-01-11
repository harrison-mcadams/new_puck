import pandas as pd
import numpy as np
import math
import sys
import os
import json

_BLOCKED_SHOT_MODEL = None
# Relative path to data directory
_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'data', 'blocked_shot_model.json')

_BLOCKED_SHOT_MODEL_F = None
_BLOCKED_SHOT_MODEL_D = None
_QUANTILE_MAPPINGS = None

try:
    from .rink import calculate_distance_and_angle
except ImportError:
    # Fallback for scripts running from project root
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    try:
        from puck.rink import calculate_distance_and_angle
    except ImportError:
        # Minimal fallback definition if imports fail
        def calculate_distance_and_angle(x, y, goal_x, goal_y=0.0):
            distance = math.hypot(x - goal_x, y - goal_y)
            vx, vy = x - goal_x, y - goal_y
            if goal_x < 0: rx, ry = 0.0, 1.0
            else: rx, ry = 0.0, -1.0
            cross = rx * vy - ry * vx
            dot = rx * vx + ry * vy
            angle_deg = (-math.degrees(math.atan2(cross, dot))) % 360.0
            return distance, angle_deg

def load_blocked_models():
    global _BLOCKED_SHOT_MODEL_F, _BLOCKED_SHOT_MODEL_D
    # Load separate models
    path_f = _MODEL_PATH.replace('.json', '_F.json')
    path_d = _MODEL_PATH.replace('.json', '_D.json')
    
    if _BLOCKED_SHOT_MODEL_F is None:
        if os.path.exists(path_f):
            try:
                with open(path_f, 'r') as f:
                    _BLOCKED_SHOT_MODEL_F = json.load(f)
            except Exception:
                _BLOCKED_SHOT_MODEL_F = {}
        else:
            # Fallback to single model if split not found?
             pass

    if _BLOCKED_SHOT_MODEL_D is None:
        if os.path.exists(path_d):
            try:
                with open(path_d, 'r') as f:
                    _BLOCKED_SHOT_MODEL_D = json.load(f)
            except Exception:
                _BLOCKED_SHOT_MODEL_D = {}
        else:
            pass
            
    return _BLOCKED_SHOT_MODEL_F, _BLOCKED_SHOT_MODEL_D

def _smooth_coordinates(x, y, scale=1.0, base_sigma=2.0):
    """
    Applies Gaussian noise to coordinates to represent uncertainty.
    
    Args:
        x, y: Coordinates
        scale: Multiplier for the noise (e.g. distance ratio)
        base_sigma: Base standard deviation in feet (default 2ft)
        
    Returns:
        (x_smooth, y_smooth)
    """
    sigma = base_sigma * scale
    # Cap sigma to reasonable limits
    sigma = min(sigma, 6.0)
    
    noise_x = np.random.normal(0, sigma)
    noise_y = np.random.normal(0, sigma)
    return x + noise_x, y + noise_y

def impute_blocked_shot_origins(df_shots: pd.DataFrame, method: str = 'empirical_model', 
                                x_col='x', y_col='y', role_col='shooter_role',
                                is_standardized: bool = False) -> pd.DataFrame:
    """
    Updates 'imputed_x', 'imputed_y', 'distance', 'angle_deg'.
    Uses 'shooter_role' (F/D) to select specific blocked shot model.
    
    Args:
        is_standardized: If True, assumes input coordinates are already standardized 
                         to Attacking Right (Goal at +89), meaning Negative X 
                         indicates Defensive Zone (Far Shot), not Left Attack.
                         Disables blind coordinate flipping.
    """
    df_out = df_shots.copy()
    
    # 1. Initialize imputed cols with original
    df_out['imputed_x'] = df_out[x_col].astype(float)
    df_out['imputed_y'] = df_out[y_col].astype(float)
    
    # Capture original PBP Block Location (for visualization/debugging)
    # This stores the location of the BLOCK event (e.g. where the puck hit the shin pad)
    df_out['block_x'] = df_out[x_col].astype(float)
    df_out['block_y'] = df_out[y_col].astype(float)

    # Mask for blocked shots
    mask_blocked = (df_shots['event'] == 'blocked-shot')
    if not mask_blocked.any():
        return df_out

    # 2. Logic for BLOCKED only
    bx = df_out.loc[mask_blocked, x_col].astype(float) # ensure float
    by = df_out.loc[mask_blocked, y_col].astype(float)
    
    # DITHERING (Critical for Angular Smoothness)
    # nhl API coords are discrete integers. We add noise (+/- 0.5ft) 
    # to simulate continuous locations and avoid angular artifacts.
    if len(bx) > 0:
        rng = np.random.default_rng(42) # Fixed seed for reproducibility
        bx += rng.uniform(-0.5, 0.5, size=len(bx))
        by += rng.uniform(-0.5, 0.5, size=len(by))

    # Get roles if available
    # Default F
    roles = pd.Series('F', index=bx.index) 
    if role_col in df_out.columns:
        roles = df_out.loc[mask_blocked, role_col].fillna('F')
    
    # Need existing distance to deduce net location
    if 'distance' in df_out.columns:
        d1 = np.sqrt((bx - 89)**2 + by**2)
        d2 = np.sqrt((bx + 89)**2 + by**2)
        old_dist = df_out.loc[mask_blocked, 'distance']
        net_x = np.where(np.abs(d1 - old_dist) < np.abs(d2 - old_dist), 89, -89)
    else:
        net_x = np.where(bx > 0, 89, -89)
    
    # Logic setup
    load_blocked_models() # Ensure loaded
    
    # Loop imputation
    new_oxs = []
    new_oys = []
    
    for x, y, role in zip(bx, by, roles):
        if pd.isna(x) or pd.isna(y):
            new_oxs.append(np.nan)
            new_oys.append(np.nan)
            continue
            
        # Normalize to Right Attack (Standardize Geometry)
        flipped = False
        
        # If NOT standardized, we assume Negative X needs flip (Attacking Left).
        # If standardized, we TRUST X (Negative X = Defensive Zone).
        should_flip = (x < 0 and not is_standardized)
        
        if should_flip:
            nx, ny = -x, -y
            flipped = True
        else:
            nx, ny = x, y
            
        # Select Model
        # If D -> use D model. Else (F or anything else) -> use F model.
        if role == 'D' and _BLOCKED_SHOT_MODEL_D and _BLOCKED_SHOT_MODEL_D.get('bins'):
             _model = _BLOCKED_SHOT_MODEL_D
        elif _BLOCKED_SHOT_MODEL_F and _BLOCKED_SHOT_MODEL_F.get('bins'):
             _model = _BLOCKED_SHOT_MODEL_F
        else:
             _model = None

        if method == 'quantile_matching' or method == 'cdf_mapping':
             # Quantile / CDF Mapping (Direct Mapping, No Inversion)
             # This assumes that a block at the Nth percentile distance from net 
             # likely comes from a shot at the Nth percentile distance from net.
             
             global _QUANTILE_MAPPINGS
             if '_QUANTILE_MAPPINGS' not in globals() or _QUANTILE_MAPPINGS is None:
                 try:
                     import joblib
                     mpath = os.path.join(os.path.dirname(__file__), 'data', 'quantile_mappings.joblib')
                     if os.path.exists(mpath):
                         _QUANTILE_MAPPINGS = joblib.load(mpath)
                     else:
                         _QUANTILE_MAPPINGS = {}
                 except Exception:
                     _QUANTILE_MAPPINGS = {}
             
             # Get Role Mapping (F vs D)
             mapping = _QUANTILE_MAPPINGS.get(role, _QUANTILE_MAPPINGS.get('global'))
             
             if not mapping and len(_QUANTILE_MAPPINGS) > 0:
                 # Fallback to any available if specific role not found
                 mapping = next(iter(_QUANTILE_MAPPINGS.values()))

             if mapping:
                 # Mapping is based on DISTANCE FROM NET
                 # 'block_dist_net': [values at 0..100 percentiles]
                 # 'origin_dist_net': [values at 0..100 percentiles]
                 
                 net_x_ref = 89.0
                 # Calculate Block Distance from Net
                 # NOTE: nx is normalized to positive side? No, `nx` logic above handled scaling?
                 # impute.py lines 128-132:
                 # if x < 0: nx = -x ...
                 # So `nx` is positive X (0..100). Net is at 89.
                 # ny is raw Y (could be negative).
                 
                 dist_blk = math.hypot(nx - net_x_ref, ny)
                 
                 # Interpolate:
                 # Map Block Distance Percentile -> Origin Distance Percentile
                 # Assumes monotonic relationship (Percentile arrays are sorted 0..100)
                 
                 try:
                     src_vals = mapping['block_dist_net']
                     dst_vals = mapping['origin_dist_net']
                     
                     dist_new = float(np.interp(dist_blk, src_vals, dst_vals))
                 except KeyError:
                     # Check if we have the old format (block_x_dist)?
                     # Fallback to X-mapping if 'block_dist_net' missing (backward compat)
                     if 'block_x_dist' in mapping:
                          # Map X coordinate directly
                          # X is normalized 0-100.
                          # nx is block X.
                          ox = float(np.interp(nx, mapping['block_x_dist'], mapping['origin_x_dist']))
                          
                          # Handle Y
                          abs_ny = abs(ny)
                          oy_abs = float(np.interp(abs_ny, mapping['block_y_dist'], mapping['origin_y_dist']))
                          oy = oy_abs if ny >= 0 else -oy_abs
                          
                          # Skip projection logic below if we mapped coords directly
                          # But we need consistent structure.
                          # Let's just use dist_new logic mainly.
                          # If we only have X/Y mapping, return result here.
                          if flipped:
                             ox, oy = -ox, -oy
                          
                          new_oxs.append(ox)
                          new_oys.append(oy)
                          continue
                          
                     dist_new = dist_blk + 15.0

                 # Project: Origin should be at dist_new from Net
                 # Vector: Net -> Block (Direction)
                 vx = nx - net_x_ref
                 vy = ny
                 mag_v = math.hypot(vx, vy)
                 
                 if mag_v < 1e-3:
                     ux, uy = -1.0, 0.0 # Center ice
                 else:
                     ux, uy = vx / mag_v, vy / mag_v
                     
                 # Origin = Net + Direction * NewDistance
                 ox = net_x_ref + ux * dist_new
                 oy = 0.0 + uy * dist_new
                 
             else:
                 ox, oy = nx, ny
                 
             # Denormalize
             if flipped:
                 ox, oy = -ox, -oy

        elif method == 'empirical_model' and _model:
            bins = _model.get('bins', {})
            meta = _model.get('meta', {})
            bin_size = meta.get('bin_size', 5.0)
            y_offset = meta.get('y_min', -50.0)

            k_x = int(nx // bin_size)
            k_y = int((ny - y_offset) // bin_size)
            key = f"{k_x}_{k_y}"

            if key in bins:
                b_data = bins[key]
                mx, my = b_data['mx'], b_data['my']
                b = bins[key]
                
                # Intra-bin Correction (Geometric "Smearing")
                # We scale the offset based on similar triangles (Net -> Block -> Origin)
                # If blocker moves 1ft, shooter moves >1ft.
                bin_center_x = (k_x + 0.5) * bin_size
                mx = b['mx']
                my = b['my']
                
                # Empirical Variance from model (if available)
                std_x_emp = b.get('std_x', 5.0) # Fallback 5ft if missing
                std_y_emp = b.get('std_y', 5.0)
                
                # Apply Gaussian Noise based on Empirical Variance
                # We trust the model's measured spread.
                # Note: We do NOT use the ratio-based smoothing anymore.
                
                rng = np.random.default_rng()
                ox = rng.normal(mx, std_x_emp)
                oy = rng.normal(my, std_y_emp)
                
            else:
                # Fallback: Straight projection 15ft
                net_x_ref = 89.0
                vx = nx - net_x_ref
                vy = ny
                mag_v = math.hypot(vx, vy)
                d_proj = 15.0
                if mag_v < 1e-3: 
                    ux, uy = -1.0, 0.0
                else:
                    ux, uy = vx/mag_v, vy/mag_v
                
                # Use generic smoothing for fallback
                base_ox = nx + ux * d_proj
                base_oy = ny + uy * d_proj
                ox, oy = _smooth_coordinates(base_ox, base_oy, scale=1.0)

            # Denormalize
            if flipped:
                ox, oy = -ox, -oy
        else:
             # Fallback method (fixed_15) if empirical model unavailable
             nx_fallback = 89 if x > 0 else -89
             vx = x - nx_fallback
             vy = y
             mag = math.hypot(vx, vy)
             d_proj = 15.0
             if mag < 1e-3:
                 ox = x # no move
                 oy = y
             else:
                 ox = x + (vx/mag) * d_proj
                 oy = y + (vy/mag) * d_proj
                 
        new_oxs.append(ox)
        new_oys.append(oy)

    # Assign
    df_out.loc[mask_blocked, 'imputed_x'] = new_oxs
    df_out.loc[mask_blocked, 'imputed_y'] = new_oys
    
    # RINK BOUNDARIES
    # Standard NHL rink is 200x85 ft. Center (0,0). X: -100 to 100. Y: -42.5 to 42.5.
    df_out.loc[mask_blocked, 'imputed_x'] = df_out.loc[mask_blocked, 'imputed_x'].clip(-99.0, 99.0)
    df_out.loc[mask_blocked, 'imputed_y'] = df_out.loc[mask_blocked, 'imputed_y'].clip(-42.0, 42.0)

    # 2.5 SPECIAL LOGIC: Behind the Net
    # If the BLOCK (input) is behind the net, force the ORIGIN (output) X to match the block X.
    # We allow Y to be adjusted (imputed), but we don't want to pull the shot "in front" of the net.
    # Net is at X = +/- 89.
    
    # Identify blocks behind net
    # using 'bx' (original block x)
    mask_behind_net = (df_out['event'] == 'blocked-shot') & (df_out[x_col].abs() > 89.0)
    
    if mask_behind_net.any():
        # Set imputed_x = original x for these rows
        df_out.loc[mask_behind_net, 'imputed_x'] = df_out.loc[mask_behind_net, x_col]
        
    # 3. Recalculate Distance & Angle (for BLOCKED only)
    
    # Create temporary DF to apply over
    idxs = df_out[mask_blocked].index
    
    # Re-infer Net X based on new imputed positions
    # Logic: Imputation transformed relative to "net_x_ref = 89" (normalized).
    # So we calculate metrics using the aligned net X.
    
    # net_x is an array aligned with bx (which was extracted from df_out.loc[mask_blocked, 'x_col'])
    # So it should match idxs
    
    if len(idxs) > 0:
        imputed_xs = df_out.loc[idxs, 'imputed_x'].values
        imputed_ys = df_out.loc[idxs, 'imputed_y'].values
        
        # Calculate new metrics
        # Vectorized or list comp
        new_dists = []
        new_angles = []
        
        # We need net_x for each row.
        # net_x computed earlier is numpy array.
        for i in range(len(idxs)):
            ix = imputed_xs[i]
            iy = imputed_ys[i]
            # Use the net_x we decided on earlier
            nx = net_x[i]
            
            d, a = calculate_distance_and_angle(ix, iy, nx, 0)
            new_dists.append(d)
            new_angles.append(a)
            
        df_out.loc[idxs, 'distance'] = new_dists
        df_out.loc[idxs, 'angle_deg'] = new_angles

    # 4. Arena Adjustments (Optional)
    # Skip if we are already using adjusted coordinates as input (prevents double adjustment)
    if '_adj' in x_col:
         return df_out
         
    col_season = None
    col_team = None
    
    if 'season' in df_out.columns:
        col_season = 'season'
    elif 'game_id' in df_out.columns:
        col_season = 'game_id'
        
    if 'home_team' in df_out.columns:
        col_team = 'home_team'
    elif 'home_abb' in df_out.columns:
        col_team = 'home_abb'
        
    if col_season and col_team:
        try:
            from .arena_adjustments import adjust_shot
            
            def get_season(val):
                s = str(val)
                if len(s) >= 4:
                    try:
                        start_year = int(s[:4])
                        return f"{start_year}{start_year+1}" 
                    except ValueError:
                        return s
                return str(val)

            subset = df_out[mask_blocked]
            if not subset.empty:
                for (season_val, team_val), group in subset.groupby([col_season, col_team]):
                    if col_season == 'game_id':
                        true_season = get_season(season_val)
                    else:
                        true_season = str(season_val)
                    true_team = str(team_val)
                    
                    res_adj = group.apply(lambda r: adjust_shot(r['imputed_x'], r['imputed_y'], true_team, true_season), axis=1)
                    
                    idx_grp = group.index
                    new_xs = res_adj.apply(lambda t: t[0])
                    new_ys = res_adj.apply(lambda t: t[1])
                    
                    df_out.loc[idx_grp, 'imputed_x'] = new_xs
                    df_out.loc[idx_grp, 'imputed_y'] = new_ys
                    
                # Re-run dist/angle for modified rows
                idxs = df_out[mask_blocked].index
                # derive net from new x
                nx_new = np.where(df_out.loc[idxs, 'imputed_x'] > 0, 89, -89)
                
                vals_x = df_out.loc[idxs, 'imputed_x'].values
                vals_y = df_out.loc[idxs, 'imputed_y'].values
                
                dists_adj = []
                angles_adj = []
                for i in range(len(idxs)):
                    d, a = calculate_distance_and_angle(vals_x[i], vals_y[i], nx_new[i], 0)
                    dists_adj.append(d)
                    angles_adj.append(a)
                    
                df_out.loc[idxs, 'distance'] = dists_adj
                df_out.loc[idxs, 'angle_deg'] = angles_adj

        except ImportError:
            pass
            
    return df_out
