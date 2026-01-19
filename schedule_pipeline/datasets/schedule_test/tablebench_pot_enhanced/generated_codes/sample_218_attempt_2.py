import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Level at Trent Bridge m' and 'Peak Flow m3/s' to numeric, handling any non-numeric values
df['Level at Trent Bridge m'] = pd.to_numeric(df['Level at Trent Bridge m'], errors='coerce')
df['Peak Flow m3/s'] = pd.to_numeric(df['Peak Flow m3/s'].str.replace(',', ''), errors='coerce')

# Drop rows with missing values
df.dropna(subset=['Level at Trent Bridge m', 'Peak Flow m3/s'], inplace=True)

# Calculate correlation between water level and peak flow
correlation = df['Level at Trent Bridge m'].corr(df['Peak Flow m3/s'])

print(f"Final Answer: {correlation:.3f}")