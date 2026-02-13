import sys
import os

print(f"CWD: {os.getcwd()}")
sys.path.insert(0, os.getcwd())

try:
    import puck.mixed_effects
    print(f"Success! Imported from: {puck.mixed_effects.__file__}")
    print(f"Classes: {dir(puck.mixed_effects)}")
except Exception as e:
    print(f"IMPORT ERROR: {e}")
    import traceback
    traceback.print_exc()
