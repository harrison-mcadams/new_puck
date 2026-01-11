
import pandas as pd
import numpy as np
import joblib
import os
import sys

def train_quantile_mappings():
    # Path to batch data
    data_path = os.path.join('analysis', 'blocked_shots', 'blocked_shots_summary_batch.csv')
    if not os.path.exists(data_path):
        print(f"Error: Data not found at {data_path}")
        return

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows.")
    
    # Filter for high quality tracking
    # Use 'score' if available (older/mixed data might use best_match_score)
    if 'score' in df.columns:
        df = df[df['score'] > 0.4].copy()
    
    # Ensure distance_to_shooter exists
    if 'distance_to_shooter' not in df.columns:
         print("Error: 'distance_to_shooter' column missing.")
         return

    print(f"Filtered (Score>0.4): {len(df)} rows.")
    
    # Reconstruction of Origin
    # Standardize Geometry: Net at X=89, Absolute Y.
    # The batch data has 'x', 'y' relative to 0,0 center.
    # We should normalize all to "Right Attack" (X > 0)
    
    bx_orig = df['x'].values
    by_orig = df['y'].values
    
    # Flip to positive X side
    bx_raw = np.abs(bx_orig)
    by_raw = df['y'].values # Keep Y sign for now, but symmetry usually implies abs(Y) for dists
    # Actually, for distribution matching, we prefer operating in the 0-100, 0-42.5 quadrant
    by_abs = np.abs(by_orig)
    
    d_shooter = df['distance_to_shooter'].values
    
    # Vector from Net (89, 0) to Block (bx_raw, by_raw)
    # Note: Origin reconstruction logic assumes the block is BETWEEN net and shooter.
    # Vector: Net -> Block. 
    #   vx = bx - 89
    #   vy = by - 0
    # Extending this vector by Dist_to_Shooter gives Origin.
    
    net_x = 89.0
    vx = bx_raw - net_x
    vy = by_abs - 0.0 # to center y
    
    mag = np.hypot(vx, vy)
    
    # Handle mathematical singularities (block at net center)
    mask_zero = mag < 1e-3
    vx[mask_zero] = -1.0 # Default towards center ice
    vy[mask_zero] = 0.0
    mag[mask_zero] = 1.0
    
    ux = vx / mag
    uy = vy / mag
    
    # Calculate Origin
    ox_raw = bx_raw + ux * d_shooter
    oy_raw = by_abs + uy * d_shooter # Y is also extended away
    
    # Add to DF
    df['calc_block_x'] = bx_raw # 0..89 (approx)
    df['calc_block_y'] = by_abs # 0..42
    df['calc_origin_x'] = ox_raw # Likely < 89 (towards blue line) or even negative
    df['calc_origin_y'] = oy_raw
    
    # Calculate Distances from Net (Net X=89, Y=0)
    # This is the primary metric for quantile mapping
    df['dist_block_net'] = np.hypot(df['calc_block_x'] - 89, df['calc_block_y'])
    df['dist_origin_net'] = np.hypot(df['calc_origin_x'] - 89, df['calc_origin_y'])

    # Debug Stats
    print("\n--- Usage Stats ---")
    print(df[['dist_block_net', 'dist_origin_net']].describe())

    # Roles to process
    roles = ['F', 'D']
    mappings = {}
    percentiles = np.arange(101) # 0..100 integer percentiles
    
    for role in roles:
        # Filter by role
        # Try 'shooter_role' from header we saw earlier
        target_col = 'shooter_role'
        if target_col not in df.columns:
            print("Warning: shooter_role not found, looking for 'role'")
            if 'role' in df.columns:
                target_col = 'role'
            else:
                 print("Error: No role column found.")
                 continue

        subset = df[df[target_col] == role]
        
        if len(subset) < 50:
            print(f"Warning: Insufficient data for role {role} ({len(subset)}). skipping.")
            continue
            
        print(f"\nProcessing ROle: {role} (N={len(subset)})")
        
        # We Map: Block Distance -> Origin Distance
        # "Direct Mapping": Pct(BlockDist) -> Pct(OriginDist)
        # So we just need the distributions of both.
        
        d_blk = subset['dist_block_net'].values
        d_org = subset['dist_origin_net'].values
        
        # Clean NaNs
        d_blk = d_blk[~np.isnan(d_blk)]
        d_org = d_org[~np.isnan(d_org)]
        
        # Calculate Percentiles
        # Store the VALUES at each percentile (0..100)
        # We also store X/Y specific distributions if we want to do coordinate-wise mapping
        # But distance-based is usually more robust for "pushing back".
        # Let's check if the impute logic wants X/Y or Dist.
        # The prompt asked for "Close/Far" logic which implies Distance.
        # But previous impute.py had X-mapping code.
        # Let's provide X-mapping (Distance from net along X axis) as well just in case.
        # Normalized X: 89 is Net. 0 is Center.
        # "Distance from Net X" = 89 - x
        
        # Let's save distributions for: 
        # 1. Net Distance (scalar)
        # 2. X Coordinate (normalized 0-100)
        # 3. Y Coordinate (normalized 0-42)
        
        # X: (Note: Lower X is FURTHER from net in 0..89 coords)
        bx = subset['calc_block_x'].values
        ox = subset['calc_origin_x'].values
        
        # Y: 
        by = subset['calc_block_y'].values
        oy = subset['calc_origin_y'].values

        mapping = {
            'block_dist_net': np.percentile(d_blk, percentiles),
            'origin_dist_net': np.percentile(d_org, percentiles),
            
            'block_x': np.percentile(bx, percentiles),
            'origin_x': np.percentile(ox, percentiles), # Note: Origin X distribution might differ shape
            
            'block_y': np.percentile(by, percentiles),
            'origin_y': np.percentile(oy, percentiles),
        }
        
        mappings[role] = mapping
        print(f"  > Saved distribution stats for {role}")
        
    # Save
    out_dir = os.path.join('puck', 'data')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'quantile_mappings.joblib')
    joblib.dump(mappings, out_path)
    print(f"\nMappings saved to {out_path}")

if __name__ == "__main__":
    train_quantile_mappings()
