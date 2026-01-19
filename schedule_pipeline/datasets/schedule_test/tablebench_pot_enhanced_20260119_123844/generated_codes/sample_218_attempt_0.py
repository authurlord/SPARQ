import pandas as pd

df = pd.read_csv('table.csv')
# Drop the row with None in 'Rank' (the "Normal / Avg flow" row)
df_clean = df.dropna(subset=['Rank'])
# Convert relevant columns to numeric
df_clean['Level at Trent Bridge m'] = pd.to_numeric(df_clean['Level at Trent Bridge m'])
df_clean['Peak Flow m3/s'] = pd.to_numeric(df_clean['Peak Flow m3/s'].str.replace(',', ''))
# Calculate correlation
correlation = df_clean['Level at Trent Bridge m'].corr(df_clean['Peak Flow m3/s'])
print(f"Final Answer: {correlation:.3f}")