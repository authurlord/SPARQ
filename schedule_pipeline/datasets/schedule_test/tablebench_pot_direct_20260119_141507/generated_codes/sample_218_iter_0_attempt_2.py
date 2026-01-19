import pandas as pd

df = pd.read_csv('table.csv')
# Drop the row with NaN for 'Date' since it's not relevant for correlation
df_clean = df.dropna(subset=['Level at Trent Bridge m', 'Peak Flow m3/s'])

# Convert the columns to numeric (in case of formatting issues)
df_clean['Level at Trent Bridge m'] = pd.to_numeric(df_clean['Level at Trent Bridge m'], errors='coerce')
df_clean['Peak Flow m3/s'] = pd.to_numeric(df_clean['Peak Flow m3/s'], errors='coerce')

# Calculate the correlation between water level and peak flow
correlation = df_clean['Level at Trent Bridge m'].corr(df_clean['Peak Flow m3/s'])
print(f"Final Answer: {correlation:.3f}")