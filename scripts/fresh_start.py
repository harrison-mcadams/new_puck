#!/usr/bin/env python3
"""
Fresh Start Routine for Raspberry Pi
====================================

This script performs a complete system refresh:
1. Purges old data (via backfill_seasons.py)
2. Re-scrapes all seasons (via backfill_seasons.py)
3. Re-trains xG models (via backfill_seasons.py)
4. Runs daily analysis for the current season (via daily.py)

Usage:
    python scripts/fresh_start.py
"""

import argparse
import sys
import os
import subprocess
import time

def main():
    parser = argparse.ArgumentParser(description='Fresh Start Routine')
    parser.add_argument('--resume', action='store_true', help='Resume from existing data (skip full wipe)')
    args = parser.parse_args()

    # Ensure we are in the project root
    # We assume this script is in 'scripts/', so root is one up.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    print(f"Working Directory: {os.getcwd()}")
    
    start_time = time.time()
    
    print("\n########################################################")
    print("###           STARTING FRESH SYSTEM REFRESH          ###")
    if args.resume:
        print("###                (RESUME MODE)                     ###")
    print("########################################################\n")
    
    try:
        # STEP 1: BACKFILL (Purge + Download)
        # -----------------------------------
        print(">>> STEP 1: Running Backfill (Purge + Download)...")
        # backfill_seasons.py is in the scripts/ folder
        backfill_script = os.path.join(script_dir, 'backfill_seasons.py')
        
        # We run it as a subprocess to keep environments clean and allow it to manage its own memory
        cmd = [sys.executable, '-u', backfill_script]
        if args.resume:
            cmd.append('--resume')
            
        subprocess.run(cmd, check=True)
        
        print("\n>>> STEP 1 COMPLETE: Data refreshed.\n")
        
        # STEP 2: TRAIN & COMPARE MODELS
        # ------------------------------
        print(">>> STEP 2: Training & Comparing Models...")
        train_script = os.path.join(script_dir, 'train_and_compare_models.py')
        
        subprocess.run([sys.executable, '-u', train_script], check=True)
        
        print("\n>>> STEP 2 COMPLETE: Models trained and dashboard generated.\n")
        
        # STEP 3: DAILY ANALYSIS (Current Season)
        # ---------------------------------------
        print(">>> STEP 3: Running Daily Analysis for Current Season...")
        daily_script = os.path.join(script_dir, 'daily.py')
        
        # We can pass --season if needed, but daily.py defaults to 20252026
        # Let's be explicit just in case
        current_season = "20252026"
        
        subprocess.run([sys.executable, '-u', daily_script, '--season', current_season], check=True)
        
        print("\n>>> STEP 3 COMPLETE: Daily analysis finished.\n")
        
    except subprocess.CalledProcessError as e:
        print(f"\n!!! ERROR: A subprocess failed with exit code {e.returncode} !!!")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"\n!!! ERROR: An unexpected error occurred: {e} !!!")
        sys.exit(1)

    elapsed = time.time() - start_time
    print("########################################################")
    print(f"###           REFRESH COMPLETE ({elapsed:.1f}s)           ###")
    print("########################################################")

if __name__ == "__main__":
    main()
