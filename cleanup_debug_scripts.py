
import os

files = [
    'debug_zero_shifts.py',
    'debug_zero_shifts_log.txt',
    'debug_zero_shifts_log_v2.txt',
    'debug_zero_shifts_log_v3.txt',
    'debug_zero_shifts_log_v4.txt',
    'check_game_status.py',
    'status_log.txt',
    'check_html_content.py',
    'check_html_log.txt',
    'check_html_log_v2.txt',
    'check_html_log_v3.txt',
    'check_html_log_v4.txt',
    'debug_zero_shifts_results.csv'
]

for f in files:
    try:
        if os.path.exists(f):
            os.remove(f)
            print(f"Removed {f}")
    except Exception as e:
        print(f"Error removing {f}: {e}")
