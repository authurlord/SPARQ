import pandas as pd

df = pd.read_csv('table.csv')

# Drop the row with None rank (normal/avg flow) for analysis of extreme events
df_filtered = df.dropna(subset=['Rank'])

# Convert relevant columns to numeric
df_filtered['Level at Trent Bridge m'] = pd.to_numeric(df_filtered['Level at Trent Bridge m'])
df_filtered['Peak Flow m3/s'] = pd.to_numeric(df_filtered['Peak Flow m3/s'])

# Sort by level to observe trend
df_sorted = df_filtered.sort_values(by='Level at Trent Bridge m', ascending=False)

# Display the sorted data to observe the trend
print("Sorted by water level (descending):")
print(df_sorted[['Level at Trent Bridge m', 'Peak Flow m3/s']])

# Final observation: As water level increases, peak flow also increases
print(f"Final Answer: increases")