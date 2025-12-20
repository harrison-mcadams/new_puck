
import sys
import os

# Ensure we can import from current dir
sys.path.append(os.getcwd())

import puck.config as config
from app import ANALYSIS_DIR

print(f"Config ANALYSIS_DIR: {config.ANALYSIS_DIR}")
print(f"App ANALYSIS_DIR: {ANALYSIS_DIR}")
print(f"Does valid file exist? {os.path.exists(os.path.join(ANALYSIS_DIR, 'teams.json'))}")
