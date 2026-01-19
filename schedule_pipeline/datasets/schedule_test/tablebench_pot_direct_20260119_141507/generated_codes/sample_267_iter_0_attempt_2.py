import pandas as pd

df = pd.read_csv('table.csv')
# Convert relevant columns to numeric
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'], errors='coerce')
df['col (m)'] = pd.to_numeric(df['col (m)'], errors='coerce')

# Filter for mountains in Morocco with elevation >= 3000 and col > 1500
filtered_df = df[
    (df['country'] == 'morocco') &
    (df['elevation (m)'] >= 3000) &
    (df['col (m)'] > 1500)
]

# Calculate average prominence of filtered peaks
if not filtered_df.empty:
    avg_prominence = filtered_df['prominence (m)'].mean()
else:
    avg_prominence = 0

print(f"Final Answer: {avg_prominence:.1f}")