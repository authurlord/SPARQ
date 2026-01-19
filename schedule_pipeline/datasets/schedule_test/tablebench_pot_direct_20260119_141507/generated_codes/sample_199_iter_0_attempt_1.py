import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert "deaths" column to numeric, replacing text with NaN
deaths_numeric = df['deaths'].str.replace('not known', np.nan).str.replace('100 +', 100).str.replace('30 +', 30).str.replace('200 +', 200).str.replace('none', 0).astype(float)

# Replace any remaining invalid entries with NaN
deaths_numeric = pd.to_numeric(deaths_numeric, errors='coerce')

# Drop rows where deaths are NaN (due to "not known" or invalid)
df_clean = df.dropna(subset=['deaths'])

# Now compute the correlation between "number of major hurricanes" and "deaths"
correlation = df_clean['number of major hurricanes'].corr(deaths_numeric)

print(f"Final Answer: {correlation:.2f}")