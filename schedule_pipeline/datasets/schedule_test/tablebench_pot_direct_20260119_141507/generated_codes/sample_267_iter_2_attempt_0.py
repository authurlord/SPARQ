import pandas as pd

df = pd.read_csv('table.csv')

# Convert relevant columns to numeric, handling potential parsing errors
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'], errors='coerce')
df['prominence (m)'] = pd.to_numeric(df['prominence (m)'], errors='coerce')
df['col (m)'] = pd.to_numeric(df['col (m)'], errors='coerce')

# Filter for mountains in Morocco with elevation >= 3000 m and col > 1500 m
filtered_df = df[
    (df['country'] == 'morocco') &
    (df['elevation (m)'] >= 3000) &
    (df['col (m)'] > 1500)
]

# Calculate the average prominence of the filtered peaks
if not filtered_df.empty:
    avg_prominence = filtered_df['prominence (m)'].mean()
    print(f"Final Answer: {avg_prominence:.1f}")
else:
    print("Final Answer: 0")