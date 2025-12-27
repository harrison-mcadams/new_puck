
import subprocess
import sys
import os

def main():
    # Ensure we are in the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    print("===========================================")
    print("  SEQUENCE START: REPARSE & RETRAIN")
    print("===========================================")

    print("\n>>> STEP 1: Reparsing all data (scripts/reparse_all.py)...")
    try:
        # Run unbuffered
        subprocess.run([sys.executable, '-u', 'scripts/reparse_all.py'], check=True)
    except subprocess.CalledProcessError:
        print("\n!!! FAILURE: Reparse script failed. Aborting sequence. !!!")
        sys.exit(1)
    except KeyboardInterrupt:
         print("\nOperation cancelled by user.")
         sys.exit(1)

    print("\n>>> STEP 1 COMPLETE: Data regenerated.")
    print("\n>>> STEP 2: Training and Comparing Models (scripts/train_and_compare_models.py)...")
    
    try:
        subprocess.run([sys.executable, '-u', 'scripts/train_and_compare_models.py'], check=True)
    except subprocess.CalledProcessError:
        print("\n!!! FAILURE: Training script failed. !!!")
        sys.exit(1)

    print("\n===========================================")
    print("  SEQUENCE COMPLETE: SUCCESS")
    print("===========================================")

if __name__ == "__main__":
    main()
