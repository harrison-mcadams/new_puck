
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from puck import fit_xgs, config

def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / 'data'
    print(f"Project Root: {project_root}")
    print(f"Data Dir: {data_dir}")
    print(f"Exists: {data_dir.exists()}")

    # MANUAL TEST
    manual_path = data_dir / "20142015/20142015_df.csv"
    print(f"Checking manual path: {manual_path}, Exists: {manual_path.exists()}")
    if manual_path.exists():
        try:
            df_m = pd.read_csv(manual_path)
            print(f"MANUAL LOAD SUCCESS. Rows: {len(df_m)}")
        except Exception as e:
            print(f"MANUAL LOAD FAILED: {e}")
            
    print("--- Manual Iteration ---")
    if data_dir.exists():
        for item in sorted(data_dir.iterdir()):
            if item.is_dir() and item.name.isdigit():
                print(f"Found season dir: {item.name}")
                csv_1 = item / f"{item.name}_df.csv"
                csv_2 = item / f"{item.name}.csv"
                print(f"  Checking {csv_1}: {csv_1.exists()}")
                print(f"  Checking {csv_2}: {csv_2.exists()}")
    
    print("--- Calling fit_xgs.load_all_seasons_data ---")
    try:
        df = fit_xgs.load_all_seasons_data(base_dir=str(data_dir))
        print(f"Successfully loaded {len(df)} rows.")
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
