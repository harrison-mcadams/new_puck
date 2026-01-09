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

def impute_blocked_shot_origins(df_shots: pd.DataFrame, method: str = 'empirical_model', 
                                x_col='x', y_col='y', role_col='shooter_role') -> pd.DataFrame:
    """
    Updates 'imputed_x', 'imputed_y', 'distance', 'angle_deg'.
    Uses 'shooter_role' (F/D) to select specific blocked shot model.
    """
    df_out = df_shots.copy()
    
    # 1. Initialize imputed cols with original
    df_out['imputed_x'] = df_out[x_col]
    df_out['imputed_y'] = df_out[y_col]

    # Mask for blocked shots
    mask_blocked = (df_shots['event'] == 'blocked-shot')
    if not mask_blocked.any():
        return df_out

    # 2. Logic for BLOCKED only
    bx = df_out.loc[mask_blocked, x_col]
    by = df_out.loc[mask_blocked, y_col]
    
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
        if x < 0:
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

        if method == 'cdf_mapping':
             # CDF / Quantile Mapping
             # Load mappings if needed
             global _CDF_MAPPINGS
             if '_CDF_MAPPINGS' not in globals() or _CDF_MAPPINGS is None:
                 try:
                     import joblib
                     mpath = os.path.join(os.path.dirname(__file__), 'data', 'cdf_mappings.joblib')
                     if os.path.exists(mpath):
                         _CDF_MAPPINGS = joblib.load(mpath)
                     else:
                         _CDF_MAPPINGS = {}
                 except Exception:
                     _CDF_MAPPINGS = {}
             
             mapping = _CDF_MAPPINGS.get(role, _CDF_MAPPINGS.get('global'))
             
             if mapping:
                 cdf_blk = mapping['cdf_block']
                 icdf_org = mapping['icdf_origin']
                 
                 net_x_ref = 89.0
                 dist_blk = math.hypot(nx - net_x_ref, ny)
                 
                 # Clip/Percentile
                 try:
                     pct = float(cdf_blk(dist_blk))
                 except ValueError:
                     pct = 0.5
                 
                 # Map
                 try:
                     dist_new = float(icdf_org(pct))
                 except ValueError:
                     dist_new = dist_blk + 15.0
                     
                 # Project
                 vx = nx - 92.0 # Focal point behind net
                 vy = ny
                 mag_v = math.hypot(vx, vy)
                 
                 if mag_v < 1e-3:
                     ux, uy = -1.0, 0.0
                 else:
                     ux, uy = vx / mag_v, vy / mag_v
                     
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
                
                # Intra-bin Correction
                bin_center_x = (k_x + 0.5) * bin_size
                bin_center_y = (k_y + 0.5) * bin_size + y_offset
                diff_x = nx - bin_center_x
                diff_y = ny - bin_center_y
                ox = mx + diff_x
                oy = my + diff_y
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
                ox = nx + ux * d_proj
                oy = ny + uy * d_proj

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
    df_out.loc[mask_blocked, 'imputed_x'] = df_out.loc[mask_blocked, 'imputed_x'].clip(-99.0, 99.0)
    df_out.loc[mask_blocked, 'imputed_y'] = df_out.loc[mask_blocked, 'imputed_y'].clip(-42.0, 42.0)
    
    # 3. Recalculate Distance & Angle (for BLOCKED only)
    
    # Create temporary DF to apply over
    idxs = df_out[mask_blocked].index
    
    # Re-infer Net X based on new imputed positions? 
    # Or keep the prior net_x assumption?
    # Logic: Imputation transformed relative to "net_x_ref = 89" (normalized).
    # So we should calculate relative to the net closest to IMPUTED position?
    # Or strict NHL geometry.
    # Let's use the net_x derived earlier but properly aligned
    
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
