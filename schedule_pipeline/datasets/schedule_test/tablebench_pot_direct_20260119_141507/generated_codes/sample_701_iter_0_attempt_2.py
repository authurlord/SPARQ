import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Convert 'Injuries (US $000)' to numeric, coercing errors to NaN
df['Injuries (US $000)'] = pd.to_numeric(df['Injuries (US $000)'], errors='coerce')
# Find the year with the maximum injuries (ignoring NaN)
max_injury_row = df.dropna(subset=['Injuries (US $000)']).sort_values(by='Injuries (US $000)', ascending=False).iloc[0]
print(f"Final Answer: {max_injury_row['Year']}")