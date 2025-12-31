import pandas as pd
import numpy as np
from puck.edge import filter_data_to_goal_moment

def infer_possession_events(df_pos: pd.DataFrame, threshold_ft: float = 6.0) -> pd.DataFrame:
    """
    Infers possession events based on player-puck proximity.
    
    Args:
        df_pos: DataFrame with columns [frame_idx, entity_type, entity_id, team_id, x, y]
                Note: Coordinates should ideally be in feet.
        threshold_ft: Distance in feet to consider 'possession'.
        
    Returns:
        DataFrame containing possession events:
        [event_idx, start_frame, end_frame, player_id, team_id, duration_frames, avg_dist_ft]
    """
    # Ensure expected columns
    required = ['frame_idx', 'entity_type', 'x', 'y']
    if not all(col in df_pos.columns for col in required):
        return pd.DataFrame() # Empty if invalid input

    # 0. Trim Post-Goal Data
    df_pos = filter_data_to_goal_moment(df_pos)

    # 1. Separate Puck and Players
    df_puck = df_pos[df_pos['entity_type'] == 'puck'].set_index('frame_idx')[['x', 'y']]
    df_players = df_pos[df_pos['entity_type'] == 'player']
    
    if df_puck.empty or df_players.empty:
        return pd.DataFrame()

    frames = sorted(df_puck.index.intersection(df_players['frame_idx'].unique()))
    
    frame_status = []
    
    # 2. Frame-by-frame analysis
    # Vectorized approach is harder here due to per-frame player sets, but we can try to optimize later.
    # For now, loop is safe for single goal events (~100-300 frames).
    
    for frame in frames:
        pk = df_puck.loc[frame]
        pl = df_players[df_players['frame_idx'] == frame]
        
        if pl.empty: continue
        
        # Calculate distances
        # Assuming x, y are comparable (feet)
        dx = pl['x'] - pk['x']
        dy = pl['y'] - pk['y']
        dists = np.sqrt(dx**2 + dy**2)
        
        min_dist = dists.min()
        
        if min_dist <= threshold_ft:
            closest_idx = dists.idxmin()
            closest = pl.loc[closest_idx]
            frame_status.append({
                'frame': frame,
                'player_id': closest['entity_id'],
                'team_id': closest['team_id'],
                'dist': min_dist
            })
        else:
            frame_status.append({
                'frame': frame,
                'player_id': None, # Loose
                'team_id': None,
                'dist': min_dist
            })
            
    if not frame_status:
        return pd.DataFrame()
        
    df_status = pd.DataFrame(frame_status)
    
    # 3. Collapse to Events
    # Group consecutive frames where player_id is the same
    # We treat 'None' (Loose) as a distinct state
    
    # Fill NA for grouping
    df_status['pid_fill'] = df_status['player_id'].fillna('LOOSE')
    df_status['grp'] = (df_status['pid_fill'] != df_status['pid_fill'].shift()).cumsum()
    
    events = []
    
    for grp_id, group_df in df_status.groupby('grp'):
        pid = group_df['player_id'].iloc[0]
        
        # Skip Loose puck segments for the "Possession Event" list
        # (Or keep them if we want to track 'Loose' time)
        # Let's keep them but label them clearly 
        is_loose = pd.isna(pid)
        
        start_f = group_df['frame'].min()
        end_f = group_df['frame'].max()
        
        events.append({
            'start_frame': start_f,
            'end_frame': end_f,
            'player_id': 'LOOSE' if is_loose else pid,
            'team_id': 'LOOSE' if is_loose else group_df['team_id'].iloc[0],
            'duration_frames': len(group_df),
            'avg_dist_ft': group_df['dist'].mean(),
            'is_possession': not is_loose
        })
        
    return pd.DataFrame(events)
