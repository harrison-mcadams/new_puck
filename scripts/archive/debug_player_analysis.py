
import subprocess
import sys

cmd = [sys.executable, 'scripts/run_player_analysis.py', '--season', '20252026', '--vmax', '0.02']

with open('player_error.log', 'w') as f:
    try:
        subprocess.run(cmd, stderr=f, stdout=f, check=True)
    except subprocess.CalledProcessError as e:
        pass # The error is in the file
