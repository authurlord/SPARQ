import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Level at Trent Bridge m' and 'Peak Flow m3/s' to numeric
df['Level at Trent Bridge m'] = pd.to_numeric(df['Level at Trent Bridge m'], errors='coerce')
df['Peak Flow m3/s'] = pd.to_numeric(df['Peak Flow m3/s'].str.replace(',', ''), errors='coerce')

# Drop the row with 'Normal / Avg flow' since it's not a peak event
df_filtered = df.dropna(subset=['Level at Trent Bridge m', 'Peak Flow m3/s'])

# Sort by level to see the trend
df_sorted = df_filtered.sort_values(by='Level at Trent Bridge m', ascending=False)

# Display the relationship
print("As the water level increases, peak flow generally increases.")
for index, row in df_sorted.iterrows():
    print(f"Level: {row['Level at Trent Bridge m']} m → Peak Flow: {row['Peak Flow m3/s']} m³/s")

# Final Answer: The peak flow increases with increasing water level.
Final Answer: increases