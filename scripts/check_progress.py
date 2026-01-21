
import os
import time
import glob

def check_progress():
    cache_dir = os.path.join('data', '20252026', 'partials')
    if not os.path.exists(cache_dir):
        print(f"Directory not found: {cache_dir}")
        return

    files = glob.glob(os.path.join(cache_dir, '*.npz'))
    total = len(files)
    
    # Check modification time
    now = time.time()
    fresh_count = 0
    for f in files:
        mtime = os.path.getmtime(f)
        if (now - mtime) < 30 * 60: # modified in last 30 mins
            fresh_count += 1
            
    print(f"Total NPZ files: {total}")
    print(f"Freshly updated (last 30m): {fresh_count}")

if __name__ == "__main__":
    check_progress()
