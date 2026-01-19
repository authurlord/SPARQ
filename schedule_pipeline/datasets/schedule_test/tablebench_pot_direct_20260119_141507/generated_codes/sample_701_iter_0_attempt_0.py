import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'Injuries (US $000)' to numeric, handling non-numeric entries (like 'no data', '2005 (estimate)')
df['Injuries (US $000)'] = pd.to_numeric(df['Injuries (US $000)'].astype(str).str.replace(r'\([^)]*\)', '', regex=True), errors='coerce')

# Find the year with the maximum injuries (excluding NaN)
max_injuries_row = df.loc[df['Injuries (US $000)'].idxmax()]
year_with_max_injuries = max_injuries_row['Year']

print(f"Final Answer: {year_with_max_injuries}")