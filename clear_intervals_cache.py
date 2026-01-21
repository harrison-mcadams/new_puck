
import os
import glob

# Explicit list of all known zero-shift outliers to patch
GAMES_TO_CLEAR = list(range(2025020640, 2025020660)) + [2025020597, 2025020624, 2025020598, 2025020599, 2025020600,
    2025020602, 2025020603, 2025020712
]

base_dir = "c:/Users/harri/Desktop/new_puck/data/20252026/game_intervals"

print(f"Flushing `game_intervals` cache for {len(GAMES_TO_CLEAR)} games...")
count = 0
for gid in GAMES_TO_CLEAR:
    path = f"{base_dir}/{gid}.json"
    if os.path.exists(path):
        try:
            os.remove(path)
            # print(f"Deleted {gid}.json")
            count += 1
        except Exception as e:
            print(f"Error {gid}: {e}")

print(f"Deleted {count} stale cache files.")
