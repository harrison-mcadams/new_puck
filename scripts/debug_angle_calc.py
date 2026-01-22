
import numpy as np

def calc_angle(x, y):
    goal_x = 89.0
    dx = x - goal_x
    dy = y
    
    rx, ry = 0.0, -1.0 
    cross = rx * dy - ry * dx
    dot = rx * dx + ry * dy
    angle_rad_ccw = np.arctan2(cross, dot)
    angle_deg = (-np.degrees(angle_rad_ccw)) % 360.0
    return angle_deg

if __name__ == "__main__":
    x = 50.0
    print(f"y=1.0: {calc_angle(x, 1.0)}")
    print(f"y=-1.0: {calc_angle(x, -1.0)}")
    print(f"y=0.1: {calc_angle(x, 0.1)}")
    print(f"y=-0.1: {calc_angle(x, -0.1)}")
