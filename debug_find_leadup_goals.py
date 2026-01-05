import pandas as pd
import glob
import os

def find_leadup_goals():
    files = glob.glob('data/edge_goals/20242025/*_positions.csv')
    print(f"Scanning {len(files)} files...")
    
    found_count = 0
    for f in files[:500]:  # Scan first 500 to find examples
        try:
            df = pd.read_csv(f)
            if df.empty: continue
            
            # Filter for puck
            dp = df[df['entity_type'] == 'puck'].sort_values('frame_idx')
            if dp.empty: continue
            
            # Check last position
            last_row = dp.iloc[-1]
            last_x = last_row['x']
            
            # Normalize if needed (raw coords are 0-100/0-200ish usually? Or -100 to 100?)
            # Usually Edge data in these CSVs might be raw or normalized.
            # Let's check ranges. If abs(x) > 80 (feet) or close to max range?
            # Standard rink is -100 to 100.
            # Raw API is 0-200ish?
            
            is_near_net = False
            if abs(last_x) > 80: # Standard coordinates
                is_near_net = True
            elif last_x > 180 or last_x < 20: # Raw coordinates?
                 # If raw is 0-200.
                 pass
            
            # Let's print the range for the first file to be sure of units
            if found_count == 0:
                 print(f"File {f} X Range: {dp['x'].min():.1f} to {dp['x'].max():.1f}")

            # Let's just look for "End X" being "extreme" (near end of rink)
            # Assuming standard feet for now as that's what other scripts seemed to expect/convert
            if abs(last_x) > 80:
                print(f"Found Lead-Up Candidate: {f} (End X: {last_x:.1f})")
                found_count += 1
                if found_count >= 10:
                    break
        except Exception as e:
            pass

if __name__ == "__main__":
    find_leadup_goals()
