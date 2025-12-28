# Nested xG Model: Comprehensive Performance Insights

**Model:** Nested_All (2014-2026 Data)

**Generated:** Automatically from `scripts/generate_insights_report.py`

---

## Layer 1: Block Model
*Predicts: Probability of a shot being blocked (given it was attempted).*

### Empirical Data (5v5)
| shot_type    |   Attempts |   Block_Pct |   Mean_Dist |
|:-------------|-----------:|------------:|------------:|
| wrist        |     497173 |           0 |      37.554 |
| snap         |     142422 |           0 |      36.999 |
| slap         |     129560 |           0 |      50.91  |
| backhand     |      72533 |           0 |      20.31  |
| tip-in       |      65703 |           0 |      18.024 |
| deflected    |      19930 |           0 |      19.443 |
| wrap-around  |       9381 |           0 |       8.357 |
| poke         |       1148 |           0 |      12.811 |
| bat          |       1029 |           0 |      14.348 |
| between-legs |        162 |           0 |      11.737 |
| cradle       |         28 |           0 |       7.14  |

### Model Logic (Controlled Test)
Comparing a Slot Shot (20ft) vs Point Shot (50ft):
| Shot_Type   |   Pred_Block_Prob_Slot |   Pred_Block_Prob_Point |
|:------------|-----------------------:|------------------------:|
| wrist       |                  0.045 |                    0.37 |
| slap        |                  0.045 |                    0.37 |
| snap        |                  0.045 |                    0.37 |
| backhand    |                  0.045 |                    0.37 |
| tip-in      |                  0.045 |                    0.37 |
| deflected   |                  0.045 |                    0.37 |

> **Insight:** Note how different shot types have different 'blockability' even at the same distance.
---

## Layer 2: Accuracy Model
*Predicts: Probability of hitting the net (given it was NOT blocked).*

### Empirical Data (5v5)
| shot_type    |   Unblocked |   Accuracy_Pct |   Mean_Dist |
|:-------------|------------:|---------------:|------------:|
| wrist        |      497173 |          0.734 |      37.554 |
| snap         |      142422 |          0.735 |      36.999 |
| slap         |      129560 |          0.663 |      50.91  |
| backhand     |       72533 |          0.755 |      20.31  |
| tip-in       |       65703 |          0.549 |      18.024 |
| deflected    |       19930 |          0.561 |      19.443 |
| wrap-around  |        9381 |          0.776 |       8.357 |
| poke         |        1148 |          0.865 |      12.811 |
| bat          |        1029 |          0.558 |      14.348 |
| between-legs |         162 |          0.679 |      11.737 |
| cradle       |          28 |          0.5   |       7.14  |

### Model Logic (Controlled Test)
Predicted Accuracy from the Slot (20ft):
| Shot_Type   |   Pred_Accuracy_Slot |
|:------------|---------------------:|
| wrist       |                0.728 |
| slap        |                0.568 |
| snap        |                0.722 |
| backhand    |                0.594 |
| tip-in      |                0.394 |
| deflected   |                0.863 |

> **Insight:** Wrist shots are significantly more accurate than slap shots. Tip-ins are low accuracy because they are deflections.
---

## Layer 3: Finish Model
*Predicts: Probability of scoring (given it is On Net).*

### Empirical Data (5v5)
| shot_type    |   On_Net |   Shooting_Pct |   Mean_Dist |
|:-------------|---------:|---------------:|------------:|
| wrist        |   364786 |          0.075 |      37.238 |
| snap         |   104737 |          0.092 |      36.223 |
| slap         |    85939 |          0.053 |      50.757 |
| backhand     |    54758 |          0.1   |      20.443 |
| tip-in       |    36067 |          0.15  |      18.688 |
| deflected    |    11182 |          0.154 |      21.418 |
| wrap-around  |     7284 |          0.062 |       7.883 |
| poke         |      993 |          0.136 |      13.066 |
| bat          |      574 |          0.216 |      14.094 |
| between-legs |      110 |          0.082 |      12     |
| cradle       |       14 |          0.286 |       7.06  |

### Model Logic (Controlled Test)
Predicted Shooting % if On Net (Slot, 20ft):
| Shot_Type   |   Pred_Shooting_Pct_Slot |
|:------------|-------------------------:|
| wrist       |                    0.107 |
| slap        |                    0.142 |
| snap        |                    0.133 |
| backhand    |                    0.073 |
| tip-in      |                    0.125 |
| deflected   |                    0.154 |

> **Insight:** Here is where Slap Shots shine. If they hit the net, they are harder to save.
---

## Summary: Total xG
*P(Goal) = P(Unblocked) * P(On Net) * P(Score)*

### Combined xG (Slot, 20ft)
| Shot_Type   |   Total_xG_Slot |
|:------------|----------------:|
| wrist       |          0.0746 |
| slap        |          0.0772 |
| snap        |          0.0916 |
| backhand    |          0.0412 |
| tip-in      |          0.0471 |
| deflected   |          0.1271 |