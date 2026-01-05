import pandas as pd
import numpy as np

def analyze_crossings():
    csv_path = 'data/edge_goals/20242025/game_2024020202_goal_328_positions.csv'
    df = pd.read_csv(csv_path)
    puck = df[df['entity_type'] == 'puck'].sort_values('frame_idx')
    
    # Normalize if needed (though we use abs checks)
    # Check Net Location
    print("Checking for Goal Crossings (abs(x) > 89)...")
    
    in_net = puck[puck['x'].abs() > 89.0]
    
    if in_net.empty:
        print("No goal crossings found!")
        return
        
    frames = in_net['frame_idx'].values
    
    # Group into continuous sequences
    # A sequence is continuous if frame[i] == frame[i-1] + 1 (allow small gaps for tracking loss?)
    # Let's simple iteration.
    
    sequences = []
    if len(frames) > 0:
        current_seq = [frames[0]]
        for f in frames[1:]:
            if f == current_seq[-1] + 1:
                current_seq.append(f)
            else:
                sequences.append(current_seq)
                current_seq = [f]
        sequences.append(current_seq)
        
    print(f"Found {len(sequences)} distinct 'In Net' sequences.")
    
    for i, seq in enumerate(sequences):
        start_f = min(seq)
        end_f = max(seq)
        duration = end_f - start_f + 1
        
        # Get start timestamp
        row = puck[puck['frame_idx'] == start_f]
        ts = row['timestamp'].iloc[0] if not row.empty else 'N/A'
        
        print(f"Sequence {i+1}: Frames {start_f}-{end_f} (Dur: {duration} frames). Start TS: {ts}")
        
        # Check coords at start
        print(f"  Start Coords: ({row['x'].iloc[0]:.2f}, {row['y'].iloc[0]:.2f})")

if __name__ == "__main__":
    analyze_crossings()
