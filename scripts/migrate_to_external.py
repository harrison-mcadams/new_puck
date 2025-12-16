#!/usr/bin/env python3
"""
Migrate Data to External Drive
==============================

This script moves/copies the contents of the local 'data' and 'analysis' 
directories to the external drive mount point defined in scripts/setup_drive.sh 
(/mnt/puck_data).

Usage:
    sudo python3 scripts/migrate_to_external.py
"""

import os
import shutil
import sys
import time

# Define paths explicitly to avoid confusion with config.py's dynamic switching
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

LOCAL_DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
LOCAL_ANALYSIS_DIR = os.path.join(PROJECT_ROOT, 'analysis')

EXTERNAL_MOUNT = '/mnt/puck_data'
EXTERNAL_DATA_DIR = os.path.join(EXTERNAL_MOUNT, 'data')
EXTERNAL_ANALYSIS_DIR = os.path.join(EXTERNAL_MOUNT, 'analysis')

def main():
    if not os.path.exists(EXTERNAL_MOUNT):
        print(f"Error: External mount point {EXTERNAL_MOUNT} not found.")
        print("Please run 'sudo ./scripts/setup_drive.sh' first.")
        sys.exit(1)

    print("===================================================")
    print("      Migrating Data to External Drive")
    print("===================================================")
    print(f"Source:      {PROJECT_ROOT}")
    print(f"Destination: {EXTERNAL_MOUNT}")
    print("---------------------------------------------------")

    # 1. Migrate DATA
    # ----------------
    if os.path.exists(LOCAL_DATA_DIR):
        print(f"\nProcessing 'data' directory...")
        size_mb = get_size_mb(LOCAL_DATA_DIR)
        print(f"  Size: {size_mb:.1f} MB")
        
        if not os.path.exists(EXTERNAL_DATA_DIR):
            print(f"  Creating {EXTERNAL_DATA_DIR}...")
            os.makedirs(EXTERNAL_DATA_DIR, exist_ok=True)
            
        print("  Copying files (this may take a while)...")
        try:
            # Using copytree with dirs_exist_ok=True (requires Python 3.8+)
            # On older python we might need a custom loop, but Pi usually has 3.9+
            copy_tree_contents(LOCAL_DATA_DIR, EXTERNAL_DATA_DIR)
            print("  [OK] Data copy complete.")
        except Exception as e:
            print(f"  [ERROR] Failed to copy data: {e}")
    else:
        print("\n'data' directory not found locally. Skipping.")

    # 2. Migrate ANALYSIS
    # --------------------
    if os.path.exists(LOCAL_ANALYSIS_DIR):
        print(f"\nProcessing 'analysis' directory...")
        size_mb = get_size_mb(LOCAL_ANALYSIS_DIR)
        print(f"  Size: {size_mb:.1f} MB")
        
        if not os.path.exists(EXTERNAL_ANALYSIS_DIR):
            os.makedirs(EXTERNAL_ANALYSIS_DIR, exist_ok=True)
            
        print("  Copying files...")
        try:
            copy_tree_contents(LOCAL_ANALYSIS_DIR, EXTERNAL_ANALYSIS_DIR)
            print("  [OK] Analysis copy complete.")
        except Exception as e:
            print(f"  [ERROR] Failed to copy analysis: {e}")
    else:
        print("\n'analysis' directory not found locally. Skipping.")

    print("\n---------------------------------------------------")
    print("Migration finished.")
    print("Verify the contents at:", EXTERNAL_MOUNT)
    print("If everything looks good, you can manually delete the local 'data' and 'analysis' folders to save space.")
    print("===================================================")

def get_size_mb(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)

def copy_tree_contents(src, dst):
    """
    Copies contents of src to dst recursively.
    """
    if sys.version_info >= (3, 8):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        # Fallback for older python
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                if not os.path.exists(d):
                    os.makedirs(d)
                copy_tree_contents(s, d)
            else:
                shutil.copy2(s, d)

if __name__ == "__main__":
    main()
