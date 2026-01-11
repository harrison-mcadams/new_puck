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
_GLOBAL_SHOT_PRIOR = None

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

def load_global_shot_prior():
    global _GLOBAL_SHOT_PRIOR
    path = os.path.join(os.path.dirname(__file__), 'data', 'global_shot_prior.json')
    if _GLOBAL_SHOT_PRIOR is None:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    _GLOBAL_SHOT_PRIOR = json.load(f)
            except Exception:
                _GLOBAL_SHOT_PRIOR = {}
    return _GLOBAL_SHOT_PRIOR

def _smooth_coordinates(x, y, scale=1.0, base_sigma=2.0):
    """
    Applies Gaussian noise to coordinates to represent uncertainty.
    """
    sigma = base_sigma * scale
    # Cap sigma to reasonable limits
    sigma = min(sigma, 6.0)
    
    noise_x = np.random.normal(0, sigma)
    noise_y = np.random.normal(0, sigma)
    return x + noise_x, y + noise_y

def impute_blocked_shot_origins(df_shots: pd.DataFrame, method: str = 'empirical_model', 
                                x_col='x', y_col='y', role_col='shooter_role',
                                is_standardized: bool = False,
                                alpha: float = 0.0) -> pd.DataFrame:
    """
    Updates 'imputed_x', 'imputed_y', 'distance', 'angle_deg'.
    """
    df_out = df_shots.copy()
    
    # 1. Initialize imputed cols with original
    df_out['imputed_x'] = df_out[x_col].astype(float)
    df_out['imputed_y'] = df_out[y_col].astype(float)
    
    df_out['block_x'] = df_out[x_col].astype(float)
    df_out['block_y'] = df_out[y_col].astype(float)

    # Mask for blocked shots
    mask_blocked = (df_shots['event'] == 'blocked-shot')
    if not mask_blocked.any():
        return df_out

    # 2. Logic for BLOCKED only
    bx = df_out.loc[mask_blocked, x_col].astype(float) 
    by = df_out.loc[mask_blocked, y_col].astype(float)
    
    if len(bx) > 0:
        rng_dither = np.random.default_rng(42)
        bx += rng_dither.uniform(-0.5, 0.5, size=len(bx))
        by += rng_dither.uniform(-0.5, 0.5, size=len(by))

    roles = pd.Series('F', index=bx.index) 
    if role_col in df_out.columns:
        roles = df_out.loc[mask_blocked, role_col].fillna('F')
    
    if 'distance' in df_out.columns:
        d1 = np.sqrt((bx - 89)**2 + by**2)
        d2 = np.sqrt((bx + 89)**2 + by**2)
        old_dist = df_out.loc[mask_blocked, 'distance']
        net_x = np.where(np.abs(d1 - old_dist) < np.abs(d2 - old_dist), 89, -89)
    else:
        net_x = np.where(bx > 0, 89, -89)
    
    load_blocked_models() 
    prior = load_global_shot_prior() if (method == 'mixture_model' or alpha > 0) else None
    
    # Pre-flatten global prior
    p_flat, x_mids_flat, y_mids_flat = None, None, None
    if method == 'mixture_model' or alpha > 0:
        if prior and 'weights' in prior:
            p_flat = np.array(prior['weights']).ravel()
            p_flat = p_flat / p_flat.sum()
            x_mids_flat = np.repeat(prior['x_mids'], len(prior['y_mids']))
            y_mids_flat = np.tile(prior['y_mids'], len(prior['x_mids']))
        else:
            alpha = 0.0 
    
    rng = np.random.default_rng(42)
    new_oxs = []
    new_oys = []
    
    for i, (x, y, role) in enumerate(zip(bx, by, roles)):
        if pd.isna(x) or pd.isna(y):
            new_oxs.append(np.nan)
            new_oys.append(np.nan)
            continue
            
        flipped = False
        should_flip = (x < 0 and not is_standardized)
        
        if should_flip:
            nx, ny = -x, -y
            flipped = True
        else:
            nx, ny = x, y
            
        if role == 'D' and _BLOCKED_SHOT_MODEL_D and _BLOCKED_SHOT_MODEL_D.get('bins'):
             _model = _BLOCKED_SHOT_MODEL_D
        elif _BLOCKED_SHOT_MODEL_F and _BLOCKED_SHOT_MODEL_F.get('bins'):
             _model = _BLOCKED_SHOT_MODEL_F
        else:
             _model = None

        method_to_use = method
        # If alpha > 0, we force mixture logic
        if alpha > 0:
            method_to_use = 'mixture_model'

        if method_to_use == 'mixture_model':
            if p_flat is not None and x_mids_flat is not None and y_mids_flat is not None and rng.random() < alpha:
                idx = rng.choice(len(p_flat), p=p_flat)
                ox = x_mids_flat[idx] + rng.uniform(-1, 1)
                oy = y_mids_flat[idx] + rng.uniform(-1, 1)
            else:
                # Local Empirical
                if _model:
                    bins = _model.get('bins', {})
                    meta = _model.get('meta', {})
                    bin_size = meta.get('bin_size', 5.0)
                    y_offset = meta.get('y_min', -50.0)
                    k_x = int(nx // bin_size)
                    k_y = int((ny - y_offset) // bin_size)
                    key = f"{k_x}_{k_y}"
                    if key in bins:
                        b = bins[key]
                        ox = rng.normal(b['mx'], b.get('std_x', 5.0))
                        oy = rng.normal(b['my'], b.get('std_y', 5.0))
                    else:
                        vx, vy = (nx-89.0), ny
                        mag = math.hypot(vx, vy)
                        if mag > 0: vx, vy = vx/mag, vy/mag
                        ox, oy = _smooth_coordinates(nx + vx*15, ny + vy*15)
                else:
                    ox, oy = nx, ny
        elif method == 'empirical_model' and _model:
            bins = _model.get('bins', {})
            meta = _model.get('meta', {})
            bin_size = meta.get('bin_size', 5.0)
            y_offset = meta.get('y_min', -50.0)
            k_x = int(nx // bin_size)
            k_y = int((ny - y_offset) // bin_size)
            key = f"{k_x}_{k_y}"
            if key in bins:
                b = bins[key]
                ox = rng.normal(b['mx'], b.get('std_x', 5.0))
                oy = rng.normal(b['my'], b.get('std_y', 5.0))
            else:
                vx, vy = (nx-89.0), ny
                mag = math.hypot(vx, vy)
                if mag > 0: vx, vy = vx/mag, vy/mag
                ox, oy = _smooth_coordinates(nx + vx*15, ny + vy*15)
        elif method in ['quantile_matching', 'cdf_mapping']:
             # (Keeping original logic for completeness)
             global _QUANTILE_MAPPINGS
             if _QUANTILE_MAPPINGS is None:
                 try:
                     import joblib
                     mpath = os.path.join(os.path.dirname(__file__), 'data', 'quantile_mappings.joblib')
                     _QUANTILE_MAPPINGS = joblib.load(mpath) if os.path.exists(mpath) else {}
                 except Exception: _QUANTILE_MAPPINGS = {}
             
             mapping = _QUANTILE_MAPPINGS.get(role, _QUANTILE_MAPPINGS.get('global'))
             if mapping:
                 dist_blk = math.hypot(nx - 89.0, ny)
                 try:
                     dist_new = float(np.interp(dist_blk, mapping['block_dist_net'], mapping['origin_dist_net']))
                 except: dist_new = dist_blk + 15.0
                 vx, vy = nx - 89.0, ny
                 mag = math.hypot(vx, vy)
                 ux, uy = (vx/mag, vy/mag) if mag > 1e-3 else (-1.0, 0.0)
                 ox, oy = 89.0 + ux * dist_new, uy * dist_new
             else: ox, oy = nx, ny
        else:
             # Fallback
             nx_fallback = 89 if x > 0 else -89
             vx, vy = x - nx_fallback, y
             mag = math.hypot(vx, vy)
             if mag < 1e-3: ox, oy = x, y
             else: ox, oy = x + (vx/mag)*15.0, y + (vy/mag)*15.0
                  
        if flipped: ox, oy = -ox, -oy
        new_oxs.append(ox)
        new_oys.append(oy)

    df_out.loc[mask_blocked, 'imputed_x'] = new_oxs
    df_out.loc[mask_blocked, 'imputed_y'] = new_oys
    df_out.loc[mask_blocked, 'imputed_x'] = df_out.loc[mask_blocked, 'imputed_x'].clip(-99.0, 99.0)
    df_out.loc[mask_blocked, 'imputed_y'] = df_out.loc[mask_blocked, 'imputed_y'].clip(-42.0, 42.0)

    mask_behind_net = (df_out['event'] == 'blocked-shot') & (df_out[x_col].abs() > 89.0)
    if mask_behind_net.any():
        df_out.loc[mask_behind_net, 'imputed_x'] = df_out.loc[mask_behind_net, x_col]
        
    idxs = df_out[mask_blocked].index
    if len(idxs) > 0:
        imputed_xs = df_out.loc[idxs, 'imputed_x'].values
        imputed_ys = df_out.loc[idxs, 'imputed_y'].values
        new_dists, new_angles = [], []
        for i in range(len(idxs)):
            d, a = calculate_distance_and_angle(imputed_xs[i], imputed_ys[i], net_x[i], 0)
            new_dists.append(d)
            new_angles.append(a)
        df_out.loc[idxs, 'distance'] = new_dists
        df_out.loc[idxs, 'angle_deg'] = new_angles

    if '_adj' in x_col: return df_out
    # (Arena Adj logic follows...)
    col_season = 'season' if 'season' in df_out.columns else ('game_id' if 'game_id' in df_out.columns else None)
    col_team = 'home_team' if 'home_team' in df_out.columns else ('home_abb' if 'home_abb' in df_out.columns else None)
    
    if col_season and col_team:
        try:
            from .arena_adjustments import adjust_shot
            subset = df_out[mask_blocked]
            for (season_val, team_val), group in subset.groupby([col_season, col_team]):
                true_season = f"{str(season_val)[:4]}{int(str(season_val)[:4])+1}" if col_season == 'game_id' else str(season_val)
                res_adj = group.apply(lambda r: adjust_shot(r['imputed_x'], r['imputed_y'], str(team_val), true_season), axis=1)
                df_out.loc[group.index, 'imputed_x'] = res_adj.apply(lambda t: t[0])
                df_out.loc[group.index, 'imputed_y'] = res_adj.apply(lambda t: t[1])
            
            idxs = df_out[mask_blocked].index
            nx_new = np.where(df_out.loc[idxs, 'imputed_x'] > 0, 89, -89)
            vals_x, vals_y = df_out.loc[idxs, 'imputed_x'].values, df_out.loc[idxs, 'imputed_y'].values
            dists_adj, angles_adj = [], []
            for i in range(len(idxs)):
                d, a = calculate_distance_and_angle(vals_x[i], vals_y[i], nx_new[i], 0)
                dists_adj.append(d)
                angles_adj.append(a)
            df_out.loc[idxs, 'distance'] = dists_adj
            df_out.loc[idxs, 'angle_deg'] = angles_adj
        except: pass
            
    return df_out
