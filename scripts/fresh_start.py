#!/usr/bin/env python3
"""
Fresh Start Routine for Raspberry Pi
====================================

This script performs a complete system refresh:
1. Purges old data and backfills (data only, parsing included).
2. Re-trains the modern XGBoost model.
3. Runs daily analysis for the current season.

Output is piped into the logs/ directory so it can be viewed in the Web Monitor.

Usage:
    nohup python3 scripts/fresh_start.py > logs/fresh_start_master.log 2>&1 &
"""

import argparse
import sys
import os
import subprocess
import time

def run_step(cmd, log_filename, step_name):
    print(f">>> STEP {step_name}: Running command...")
    print(f"    Command: {' '.join(cmd)}")
    print(f"    Logging to: {log_filename}")
    
    with open(log_filename, 'w') as f:
        # Popen can pipe to file
        try:
            process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            # wait for completion
            process.wait()
            if process.returncode != 0:
                print(f"!!! ERROR: Step {step_name} failed with exit code {process.returncode} !!!")
                print(f"    Check {log_filename} for details.")
                sys.exit(process.returncode)
            print(f">>> STEP {step_name} COMPLETE.\n")
        except Exception as e:
            print(f"!!! ERROR: Failed to execute {cmd}: {e}")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Fresh Start Routine')
    parser.add_argument('--resume', action='store_true', help='Resume from existing data (skip already downloaded seasons)')
    parser.add_argument('--skip-backfill', action='store_true', help='Skip Step 1 (Backfill) entirely')
    parser.add_argument('--turbo', action='store_true', help='Use parallel processing if execution is not on Pi')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    print(f"Working Directory: {os.getcwd()}")
    
    logs_dir = os.path.join(project_root, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    start_time = time.time()
    
    print("\n########################################################")
    print("###           STARTING FRESH SYSTEM REFRESH          ###")
    if args.resume:
        print("###                (RESUME MODE)                     ###")
    print("########################################################\n")
    
    # STEP 1: BACKFILL (Purge + Download)
    if not args.skip_backfill:
        log_backfill = os.path.join(logs_dir, 'backfill.log')
        cmd1 = [sys.executable, '-u', os.path.join(script_dir, 'backfill_seasons.py')]
        if args.resume:
             cmd1.append('--resume')
        run_step(cmd1, log_backfill, "1 (Backfill Data)")
    else:
        print(">>> STEP 1 SKIPPED (--skip-backfill requested)\n")

    # STEP 2: TRAIN XGBOOST MODEL (Modern Era nested model)
    log_train = os.path.join(logs_dir, 'train_xgboost.log')
    cmd2 = [sys.executable, '-u', os.path.join(script_dir, 'train_xgboost_nested_20202021.py')]
    run_step(cmd2, log_train, "2 (Train XGBoost Model)")

    # STEP 3: DAILY ANALYSIS (Current Season)
    # The default behavior handles today's fetching.
    log_daily = os.path.join(logs_dir, 'daily.log')
    cmd3 = [sys.executable, '-u', os.path.join(script_dir, 'daily.py')]
    # Pass turbo flag if specified (useful for local dev test of fresh_start)
    if args.turbo:
        cmd3.append('--turbo')
    run_step(cmd3, log_daily, "3 (Run Daily Analysis)")

    elapsed = time.time() - start_time
    print("########################################################")
    print(f"###           REFRESH COMPLETE ({elapsed:.1f}s)           ###")
    print("########################################################")
    print("Web dashboard data & models are fully updated.\n")

if __name__ == "__main__":
    main()
