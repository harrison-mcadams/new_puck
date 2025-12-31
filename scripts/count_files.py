import glob
import os

files = glob.glob(r"c:\Users\harri\Desktop\new_puck\data\edge_goals\*positions.csv")
print(f"Total Position Files Found: {len(files)}")
if len(files) < 10:
    print("Files found:", [os.path.basename(f) for f in files])
