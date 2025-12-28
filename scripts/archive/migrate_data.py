
import os
import shutil
from pathlib import Path

base_data = Path('data')
processed_data = Path('data/processed')

# Process years 2014 to 2024 (start years) -> 20142015 to 20242025
# 20252026 is already in processed.

for start_year in range(2014, 2025):
    season_str = f"{start_year}{start_year + 1}"
    source_dir = base_data / season_str
    source_file = source_dir / f"{season_str}_df.csv"
    
    if source_file.exists():
        dest_dir = processed_data / season_str
        dest_file = dest_dir / f"{season_str}.csv"
        
        print(f"Migrating {season_str}...")
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source_file), str(dest_file))
            print(f"  Moved to {dest_file}")
            
            # Optionally remove empty source dir? 
            # User didn't explicitly say delete, but "move" implies it.
            # I'll check if source dir is empty later or just leave it for safety.
        except Exception as e:
            print(f"  Error moving {season_str}: {e}")
    else:
        print(f"Skipping {season_str}: Source {source_file} not found.")
