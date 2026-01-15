
import pandas as pd
import numpy as np

def debug_logic():
    # Setup single row reproducing the "Normal Block" case
    # Normal Block: X=61.0. Blocker=16(Away). DefSide=Left.
    # Corrected: TeamID=6(Home).
    
    data = {
        'event': ['blocked-shot'],
        'team_id': [6],      # Home (Corrected Shooter)
        'home_id': [6],      # Home
        'away_id': [16],     # Away
        'home_team_defending_side': ['left'],
        'x': [61.0],
        'y': [-16.0]
    }
    
    df = pd.DataFrame(data)
    
    print("--- INPUT DATA ---")
    print(df.to_dict(orient='records')[0])
    
    # Logic from data_pipeline.py
    
    # 1. Def Side Map
    def_side_map = df['home_team_defending_side'].astype(str).str.lower().str.strip().map({
        'left': -1, 
        'right': 1
    })
    print(f"\nDefSideMap:\n{def_side_map}")
    
    # 2. Is Home
    # Explicitly verify types
    print(f"\nTypes: team_id={df['team_id'].dtype}, home_id={df['home_id'].dtype}")
    
    is_home = (df['team_id'] == df['home_id'])
    print(f"\nIsHome:\n{is_home}")
    
    # 3. Side Multiplier
    side_multiplier = np.where(is_home, -1, 1)
    print(f"\nSideMultiplier (np.where):\n{side_multiplier}")
    
    # 4. Attacking Side
    attacking_side = def_side_map * side_multiplier
    print(f"\nAttackingSide:\n{attacking_side}")
    
    # 5. Flip Decision
    mask_flip = (attacking_side == -1)
    print(f"\nMaskFlip:\n{mask_flip}")
    
    if mask_flip.any():
        print("\nDECISION: FLIP (Attacking Left)")
        flip_x = df['x'] * -1
        print(f"Result X: {flip_x.values[0]}")
    else:
        print("\nDECISION: NO FLIP (Attacking Right)")
        print(f"Result X: {df['x'].values[0]}")

if __name__ == "__main__":
    debug_logic()
