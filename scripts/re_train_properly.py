import sys
import os

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import fit_nested_xgs

if __name__ == "__main__":
    print("Retraining via module import...")
    fit_nested_xgs.main()
