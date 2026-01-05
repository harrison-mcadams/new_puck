import pandas as pd
import os

data_dir = r"c:\Users\harri\Desktop\new_puck\data\edge_goals"
on_puck_file = os.path.join(data_dir, "mod_baseline_on_puck.csv")
off_puck_file = os.path.join(data_dir, "mod_baseline_off_puck.csv")

def inspect_baseline(file_path, name):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        print(f"\n--- {name} Baseline ---")
        # Near goal: x ~ 85, y ~ 0
        near_goal = df[(df['x_bin'] >= 80) & (df['y_bin'].abs() <= 10)]
        print("Near Goal (80+, |y|<=10):")
        print(near_goal.to_string())
        
        # Neutral zone: x ~ 0, y ~ 0
        neutral = df[(df['x_bin'].between(0, 20)) & (df['y_bin'].abs() <= 10)]
        print("\nNeutral Zone (0-20, |y|<=10):")
        print(neutral.to_string())
        
        # Perimeter/Point: x ~ 60, y ~ 30
        point = df[(df['x_bin'].between(55, 65)) & (df['y_bin'].abs() >= 25)]
        print("\nPoint/Wall (55-65, |y|>=25):")
        print(point.to_string())
    else:
        print(f"File not found: {file_path}")

inspect_baseline(on_puck_file, "On-Puck")
inspect_baseline(off_puck_file, "Off-Puck")
