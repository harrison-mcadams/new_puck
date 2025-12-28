
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import puck.parse

print(f"puck.parse file: {puck.parse.__file__}")

if hasattr(puck.parse, '_period_time_to_seconds'):
    print("Found _period_time_to_seconds in puck.parse")
else:
    print("MISSING _period_time_to_seconds in puck.parse")

try:
    from puck.parse import rink_goal_xs
    print("Imported rink_goal_xs successfully")
except ImportError:
    print("Failed to import rink_goal_xs")
except Exception as e:
    print(f"Error importing rink_goal_xs: {e}")
