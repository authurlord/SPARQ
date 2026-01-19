import pandas as pd

df = pd.read_csv('table.csv')
# Clean the 'Injuries (US $000)' column by converting to numeric, coercing errors to NaN
df['Injuries (US $000)'] = pd.to_numeric(df['Injuries (US $000)'], errors='coerce')
# Find the row with the maximum injuries
max_injuries_row = df.loc[df['Injuries (US $000)'].idxmax()]
year_with_max_injuries = max_injuries_row['Year']
print(f"Final Answer: {year_with_max_injuries}")