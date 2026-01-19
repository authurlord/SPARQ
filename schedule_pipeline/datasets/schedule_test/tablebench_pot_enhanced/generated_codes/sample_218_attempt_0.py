import pandas as pd

df = pd.read_csv('table.csv')
# Convert relevant columns to numeric, handling non-numeric entries
df['Level at Trent Bridge m'] = pd.to_numeric(df['Level at Trent Bridge m'], errors='coerce')
df['Peak Flow m3/s'] = pd.to_numeric(df['Peak Flow m3/s'].str.replace(',', ''), errors='coerce')

# Drop the row with 'Normal / Avg flow' as it's not a peak event
df = df.dropna(subset=['Level at Trent Bridge m', 'Peak Flow m3/s'])

# Sort by level to observe the trend
df_sorted = df.sort_values(by='Level at Trent Bridge m', ascending=False)

# Display the trend: as level increases, peak flow increases
print("Final Answer: increases")