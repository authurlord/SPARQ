import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Convert 'Injuries (US $000)' to numeric, handling missing or non-numeric values
df['Injuries (US $000)'] = pd.to_numeric(df['Injuries (US $000)'], errors='coerce')

# Find the year with the maximum injuries (ignoring NaN)
max_injuries_row = df.loc[df['Injuries (US $000)'].idxmax()]
year_with_max_injuries = max_injuries_row['Year']

print(f"Final Answer: {year_with_max_injuries}")