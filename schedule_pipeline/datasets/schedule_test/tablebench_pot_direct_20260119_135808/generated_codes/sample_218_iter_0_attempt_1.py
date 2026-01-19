import pandas as pd

df = pd.read_csv('table.csv')
# Drop the row with missing rank (normal/avg flow) for analysis
df_clean = df.dropna(subset=['Rank'])
# Convert columns to numeric for proper sorting
df_clean['Level at Trent Bridge m'] = pd.to_numeric(df_clean['Level at Trent Bridge m'])
df_clean['Peak Flow m3/s'] = pd.to_numeric(df_clean['Peak Flow m3/s'].str.replace(',', ''))
# Sort by water level in ascending order to observe trend
df_sorted = df_clean.sort_values(by='Level at Trent Bridge m')
# Display the sorted data showing how peak flow increases with water level
print(f"Final Answer: {df_sorted[['Level at Trent Bridge m', 'Peak Flow m3/s']].to_string(index=False)}")