filename = r"c:\Users\harri\Desktop\new_puck\data\debug_gravity.log"
try:
    with open(filename, 'r') as f:
        for i in range(50):
            print(f.readline(), end='')
except Exception as e:
    print(e)
