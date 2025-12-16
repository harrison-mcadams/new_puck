import os
import platform
import sys

# Try to import psutil, handle if missing
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Detect Platform
IS_MAC = platform.system() == 'Darwin'
IS_LINUX = platform.system() == 'Linux'

# Detect Raspberry Pi (generic linux check is usually enough, but we can be specific)
# Usually Pi has 'aarch64' or 'arm' in machine, but IS_LINUX + low memory is a good proxy.
IS_PI = IS_LINUX and (os.uname().machine.startswith('arm') or os.uname().machine.startswith('aarch64'))

# Memory Thresholds (MB)
try:
    if HAS_PSUTIL:
        TOTAL_RAM_MB = psutil.virtual_memory().total / (1024 * 1024)
    else:
        # Fallback using os.sysconf if on Unix/Linux
        if IS_LINUX or IS_MAC:
            import os
            # _SC_PHYS_PAGES * _SC_PAGE_SIZE
            if hasattr(os, 'sysconf') and 'SC_PHYS_PAGES' in os.sysconf_names and 'SC_PAGE_SIZE' in os.sysconf_names:
                mem_bytes = os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
                TOTAL_RAM_MB = mem_bytes / (1024 * 1024)
            else:
                TOTAL_RAM_MB = 8192 # Assumption for dev machine if checks fail
        else:
            TOTAL_RAM_MB = 8192 # Assumption
except Exception:
    TOTAL_RAM_MB = 4096 # Conservative default

# Config
if IS_PI or TOTAL_RAM_MB < 4500: # Increased threshold slightly to catch 4GB Pi
    # Low Resource Mode (Pi 4 typically 2GB/4GB/8GB, but we assume constrained)
    MAX_WORKERS = 1
    USE_MULTIPROCESSING = False
    GC_FREQUENCY = 1 # Run GC every N items
    BATCH_SIZE = 10
    MEMORY_MODE = 'aggressive'
else:
    # High Performance Mode (Mac)
    MAX_WORKERS = 4
    USE_MULTIPROCESSING = True # Can spawn processes
    GC_FREQUENCY = 50
    BATCH_SIZE = 50
    MEMORY_MODE = 'standard'


# Directory Paths (Absolute)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# External Drive Path Recommendation
EXTERNAL_MOUNT_POINT = '/ssd/new_puck'

if IS_PI and os.path.exists(EXTERNAL_MOUNT_POINT):
    # Use external drive and ensure subdirectories exist
    # We use join to create /mnt/puck_data/data and /mnt/puck_data/analysis
    DATA_DIR = os.path.join(EXTERNAL_MOUNT_POINT, 'data')
    ANALYSIS_DIR = os.path.join(EXTERNAL_MOUNT_POINT, 'analysis')
    # Print a visible confirmation in logs
    if not os.path.exists(DATA_DIR):
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            os.makedirs(ANALYSIS_DIR, exist_ok=True)
        except OSError:
            pass # Permission issues might prevent this, but we try.
else:
    # Default to local project directory
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    ANALYSIS_DIR = os.path.join(BASE_DIR, 'analysis')

CACHE_DIR = os.path.join(DATA_DIR, 'cache')

def get_cache_dir(season):
    return os.path.join(CACHE_DIR, season)

def get_analysis_dir(season):
    return os.path.join(ANALYSIS_DIR, season)

print(f"Config: Mode={MEMORY_MODE}, Workers={MAX_WORKERS}, Platform={'Pi' if IS_PI else 'Mac/Other'}")
