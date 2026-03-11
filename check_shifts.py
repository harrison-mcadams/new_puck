
from puck import timing
import json

def check_shifts():
    gid = 2025020059
    t = 2703 # Period 3, 14:57 (PBP says 303s into period? No, 14:57 is 14:57 left or elapsed? Usually MM:SS elapsed.)
    # 14:57 elapsed in P3 = 40*60 + 14*60 + 57 = 2400 + 840 + 57 = 3297s
    # Total time elapsed s in PBP row was 2703.
    # 2703 - 2400 = 303s = 5mins 03s.
    t = 2703
    
    shifts_res = timing.get_shifts_with_html_fallback(gid)
    all_shifts = shifts_res.get('all_shifts', [])
    
    active = [s for s in all_shifts if s['start_total_seconds'] <= t and s['end_total_seconds'] >= t]
    
    teams = {}
    for s in active:
        tid = s['team_id']
        teams[tid] = teams.get(tid, 0) + 1
        
    print(f"Time {t} active players:")
    for tid, count in teams.items():
        print(f"  Team {tid}: {count}")

if __name__ == "__main__":
    check_shifts()
