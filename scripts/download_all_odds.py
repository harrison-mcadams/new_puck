import os
import subprocess
import glob
import pandas as pd

# List of seasons to harvest
SEASONS = [
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
    "2025-2026"
]

# We are interested in standard NHL Over/Under lines
# Usually NHL closing lines are around 5.5, 6.0, 6.5. We will grab the main ones.
MARKETS = "over_under_5_5,over_under_6_5"

OUTPUT_DIR = "data/raw_odds"
FINAL_CSV = "data/historical_odds.csv"

def run_harvester():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for season in SEASONS:
        print(f"[*] Starting OddsHarvester for NHL season: {season}")
        output_name = f"{OUTPUT_DIR}/nhl_{season}"
        
        # Build the CLI command
        cmd = [
            "oddsharvester",
            "historic",
            "-s", "ice-hockey",
            "-l", "nhl",
            "--season", season,
            "-m", MARKETS,
            "-f", "csv",
            "-o", output_name,
            "--headless"
        ]
        
        print(f"Executing: {' '.join(cmd)}")
        # Run the command
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print(f"[+] Successfully harvested {season}")
        else:
            print(f"[-] Error harvesting {season}")

def combine_csvs():
    print(f"[*] Combining CSVs into {FINAL_CSV}...")
    all_files = glob.glob(f"{OUTPUT_DIR}/*.csv")
    
    if not all_files:
        print("[-] No CSV files found to combine.")
        return

    df_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            df['source_file'] = os.path.basename(f)
            df_list.append(df)
        except Exception as e:
            print(f"[-] Failed to read {f}: {e}")
            
    if df_list:
        final_df = pd.concat(df_list, ignore_index=True)
        os.makedirs(os.path.dirname(FINAL_CSV), exist_ok=True)
        final_df.to_csv(FINAL_CSV, index=False)
        print(f"[+] Combined {len(all_files)} files into {FINAL_CSV}")
        print(f"Total rows: {len(final_df)}")
        
if __name__ == "__main__":
    run_harvester()
    combine_csvs()
    print("\n[!] All done! You can now analyze your historical odds.")
