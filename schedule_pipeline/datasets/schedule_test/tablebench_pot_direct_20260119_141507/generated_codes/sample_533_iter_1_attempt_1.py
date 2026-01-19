So the population aged 11–59 = sum of (10–19) + (20–29) + (30–39) + (40–49) + (50–59)
So "60+" = 60–69 + 70–79 + 80+ = 20 + 24 + 14 = 58
But the table says "80 +" → 14, and "60–69" → 20, "70–79" → 24 → so total 60+ = 20 + 24 + 14 = 58
Now, 11–59 = total population minus (0–10) and (60+) → but we must be careful.
- 60+: 60–69 (20) + 70–79 (24) + 80+ (14) = 58
- So total dependent = 41 (0–9) + 58 (60+) = 99
- 11–59: total population minus (0–9) and (60+) = 287 - 41 - 58 = 188
→ Sum = 45 + 47 + 27 + 38 + 31 = 188
So dependency ratio = (0–9 + 60+) / (11–59) = (41 + 58) / 188 = 99 / 188 ≈ 0.5266
5. Dependency ratio = (0–9 + 60+) / (11–59)
import pandas as pd
df = pd.read_csv('table.csv')
total_row = df[df['SPECIFICATION'] == 'I.'].iloc[0]
pop_0_to_9 = total_row['POPULATION (by age group in 2002)_2']  # index 7
pop_60_plus = total_row['POPULATION (by age group in 2002)_13'] + total_row['POPULATION (by age group in 2002)_14'] + total_row['POPULATION (by age group in 2002)_15']  # 60-69, 70-79, 80+
pop_11_to_59 = (