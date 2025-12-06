# xG Model Future Improvements

To improve the model (currently ROC AUC ~0.768) while staying within current constraints (public Play-by-Play data + Raspberry Pi compute), here are the highest-ROI improvements.

## 1. The "Context" Features (High Impact)
Currently, the model treats every shot in a vacuum.
*   **Time Since Last Event (Rush Proxy):** If a shot happens 2 seconds after a "Takeaway" or "Zone Entry," it is likely a rush/odd-man break.
*   **Rebounds:** Using `time_since_last_shot` + `distance_from_last_shot`. If a shot happens < 2-3 seconds after another shot, xG should increase significantly.
*   **Implementation:** Efficient `pandas` shifts (`df['time__diff'] = df['time'] - df['time'].shift(1)`). Cheap to compute on Pi.

## 2. Shot Type & Angle Interaction
*   **Shot Type:** Include `secondary_type` (Wrist Shot, Slap Shot, Snap Shot, Backhand, Tip-In).
    *   *Why:* A "Tip-in" from 5ft is very different from a "Wrist Shot" from 5ft.
*   **Implementation:** One-hot encode these categories.

## 3. Strength State Granularity
*   **Detailed Strength:** Expand beyond basic states to: 5v5, 5v4, 4v5, 4v4, 3v3, Empty Net.
*   **Score State:** Teams trailing by 1 or 2 goals shoot differently. (Optional: xG usually measures quality independent of score, but it interacts with tactic).

## Expected Impact
Adding **Rebounds** and **Shot Type** alone could push ROC AUC from **0.768** to **0.775+**.
