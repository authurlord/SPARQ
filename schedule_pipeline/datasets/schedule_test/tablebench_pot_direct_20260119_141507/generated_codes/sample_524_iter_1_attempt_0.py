- Total seating in 2004 = sum of seating where introduced ≤ 2004 and (retired ≥ 2004 or retired == '-')
- Total seating in 2008 = sum of seating where introduced ≤ 2008 and (retired ≥ 2008 or retired == '-')
Change = total_seating_2008 - total_seating_2004
- retired ≥ 2004 OR retired == "-"
- retired ≥ 2008 OR retired == "-"
→ Total 2004 = 156 + 148 + 149 = 453
→ Total 2008 = 156 + 180 + 220 + 149 = 705
Change = 705 - 453 = 252
import pandas as pd
df = pd.read_csv('table.csv')