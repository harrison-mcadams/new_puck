
### Debugging Case: Goal 2024010039 - 916 (York/Mayfield)

**Problem**: A clear blocked shot at Frame 70 was not being identified.
**Investigation**:
1. **Coordinate Mismatch**: Pre-season tracking data used raw coordinates (X > 1000) while net positions were in rink coordinates. Enabled dynamic normalization.
2. **Angle Wrapping**: The puck crossed the -180/180 degree boundary, causing an artificial `angle_std` spike. Implemented `np.unwrap`.
3. **Proximity Anchoring**: The "Closest Player" fallback was pulling the shot start forward to Frame 70 (opponent proximity 2.5ft) instead of Frame 68 (York proximity 4.0ft). Modified logic to prioritize the earliest valid release.

**Result**: **VERIFIED**
- **Shot Release**: Frame 68 (4.0ft from York).
- **Speed**: 55.4 fps (accelerating to 103 fps).
- **Blocker Proximity**: Traced to **Scott Mayfield** at Frame 70.
- **Score**: 0.513 (Successfully outscored noise and false starts).

![Candidate 2024010039 916 Animation](/C:/Users/harri/.gemini/antigravity/brain/3be1d619-2259-42ba-9092-b0163917e8ec/candidate_2024010039_916.gif)

---

## Batch Processing Results

We executed the refined identification logic across all available Edge goal sequences.

| Season | Goals Processed | Blocked Shots Identified | Success Rate |
| :--- | :--- | :--- | :--- |
| 2024-2025 | 185 | 142 | **76.7%** |

The improved robustness allows the system to handle:
- **Missing IDs**: Automatic fallback to closest player.
- **Pre-Season Data**: Dynamic normalization and coordinate mapping.
- **Noisy Trajectories**: Wrap-aware linearity checks.
