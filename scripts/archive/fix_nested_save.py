
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import as module so the class is registered as puck.fit_nested_xgs.NestedXGClassifier
# NOT __main__.NestedXGClassifier
from puck import fit_nested_xgs

if __name__ == "__main__":
    print("Re-training/Re-saving model with correct module path...")
    fit_nested_xgs.main()
