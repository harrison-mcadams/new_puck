import numpy as np
import pandas as pd

try:
    from .rink import calculate_distance_and_angle, rink_goal_xs
except ImportError:
    # Fallback if relative import fails or rink not found
    def calculate_distance_and_angle(x, y, gx, gy=0.0):
         import math
         return math.hypot(x-gx, y-gy), 0.0
    def rink_goal_xs(): return -89.0, 89.0

def fix_blocked_shot_attribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrects attribution for blocked shots in the provided DataFrame.
    
    The NHL API attributes 'blocked-shot' events to the blocking team (Defense).
    For xG analysis and model training, we want these events attributed to the SHOOTER (Offense).
    
    This function:
    1. Identifies blocked shots.
    2. Swaps 'team_id' (Home <-> Away).
    3. Recalculates 'distance' and 'angle_deg' relative to the SHOOTER's attacking goal.
    
    Args:
        df: DataFrame containing event data. Must have 'event', 'team_id', 'home_id', 'away_id'.
            'home_team_defending_side' is recommended for accurate distance recalculation.
            
    Returns:
        pd.DataFrame: The modified DataFrame (copy).
    """
    df = df.copy()
    
    if 'event' not in df.columns:
        return df
        
    is_blocked = df['event'].astype(str).str.strip().str.lower() == 'blocked-shot'
    
    if not is_blocked.any():
        return df
        
    # --- 1. Swap team_id ---
    # Use safe accessors (handle mixed types by casting to str for comparison)
    t_id = df.get('team_id', pd.Series(dtype=object)).astype(str).values
    h_id = df.get('home_id', pd.Series(dtype=object)).astype(str).values
    a_id = df.get('away_id', pd.Series(dtype=object)).astype(str).values
    
    # Mask where team == home
    mask_home_t = (t_id == h_id)
    # Mask where team == away
    mask_away_t = (t_id == a_id)
    
    # New team ID array initialized with original
    new_t_id = df['team_id'].values.copy()
    
    # Apply swap logic
    # where team == home -> away
    new_t_id[mask_home_t] = df.loc[mask_home_t, 'away_id'].values
    # where team == away -> home
    new_t_id[mask_away_t] = df.loc[mask_away_t, 'home_id'].values
    
    # Assign back to blocked shots ONLY
    df.loc[is_blocked, 'team_id'] = new_t_id[is_blocked]
    
    # --- 1.5. Flip Coordinates (X, Y) ---
    # The raw data likely normalized coordinates assuming the Event Owner (Blocker) was ATTACKING.
    # Since they were Defending, the event is actually in the opposite zone.
    # We flip X and Y to place the event in the correct physical location (Defensive Zone of Blocker).
    # This ensures consistency when we treat the Shooter as the Attacker.
    df.loc[is_blocked, 'x'] *= -1
    df.loc[is_blocked, 'y'] *= -1
    
    # --- 2. Recalculate Distance and Angle ---
    # Now team_id is the Shooter. We need distance to the Goal the Shooter is Attacking.
    
    if 'home_team_defending_side' in df.columns:
         sides = df['home_team_defending_side'].astype(str).str.lower().values
         
         # Rink Goals
         try:
             lg_x, rg_x = rink_goal_xs()
         except:
             lg_x, rg_x = -89.0, 89.0
             
         # Determine Goal X for each blocked shot
         # Logic based on PRE-SWAP masks (cleaner state)
         # mask_away_t: Blocker was Away -> Shooter is Home.
         # mask_home_t: Blocker was Home -> Shooter is Away.
         
         # Logic Table:
         # Shooter | Home Defends | Attack Goal
         # Home    | Left         | Right (89)
         # Home    | Right        | Left (-89)
         # Away    | Left         | Left (-89)
         # Away    | Right        | Right (89)
         
         # target X array
         target_xs = np.full(len(df), np.nan)
         
         cond_side_left = (sides == 'left')
         cond_side_right = (sides == 'right')
         
         # 1. Shooter Home (Blocker Away) + Side Left -> Right Goal (89)
         m1 = mask_away_t & cond_side_left
         target_xs[m1] = rg_x
         
         # 2. Shooter Home (Blocker Away) + Side Right -> Left Goal (-89)
         m2 = mask_away_t & cond_side_right
         target_xs[m2] = lg_x
         
         # 3. Shooter Away (Blocker Home) + Side Left -> Left Goal (-89)
         m3 = mask_home_t & cond_side_left
         target_xs[m3] = lg_x
         
         # 4. Shooter Away (Blocker Home) + Side Right -> Right Goal (89)
         m4 = mask_home_t & cond_side_right
         target_xs[m4] = rg_x
         
         # Apply to blocked shots
         bx = df.loc[is_blocked, 'x'].values
         by = df.loc[is_blocked, 'y'].values
         gxs = target_xs[is_blocked]
         
         # Recalculate metrics
         new_dists = []
         new_angles = []
         
         for i in range(len(bx)):
             gx = gxs[i]
             if np.isnan(gx):
                 # Side unknown
                 new_dists.append(np.nan)
                 new_angles.append(np.nan)
             else:
                 try:
                     d, a = calculate_distance_and_angle(bx[i], by[i], gx, 0.0)
                     new_dists.append(d)
                     new_angles.append(a)
                 except:
                     new_dists.append(np.nan)
                     new_angles.append(np.nan)
         
         df.loc[is_blocked, 'distance'] = new_dists
         df.loc[is_blocked, 'angle_deg'] = new_angles
         
    return df
