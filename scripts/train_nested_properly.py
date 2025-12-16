
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import puck.fit_nested_xgs as fit_nested

if __name__ == "__main__":
    print("Running training via import to ensure correct pickling path...")
    fit_nested.main()
