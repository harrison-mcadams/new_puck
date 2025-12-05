import dtale
import pandas as pd
import os
import sys

def view_df(df, title="DataFrame View"):
    """
    Opens a pandas DataFrame in the default web browser using D-Tale.
    """
    print(f"Opening DataFrame '{title}' in D-Tale...")
    # dtale.show returns a D-Tale instance. 
    # .open_browser() opens it in the default browser.
    d = dtale.show(df, name=title)
    d.open_browser()
    
    # If running as a script, we need to keep the process alive.
    # If running interactively, this isn't needed, but for a script it is.
    # However, dtale runs in a background thread usually. 
    # For a CLI tool, we might want to block.
    if __name__ == "__main__":
        print("Press Ctrl+C to exit")
        try:
            # Keep alive
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Exiting...")

def view_csv(csv_path):
    """Loads a CSV and opens it in D-Tale."""
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            view_df(df, title=os.path.basename(csv_path))
        except Exception as e:
            print(f"Error reading CSV: {e}")
    else:
        print(f"File not found: {csv_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if file_path.endswith('.csv'):
            view_csv(file_path)
        else:
            print("Usage: python view_utils.py <path_to_csv>")
    else:
        print("Usage: python view_utils.py <path_to_csv>")
