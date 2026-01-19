import pandas as pd

df = pd.read_csv('table.csv')

# Convert elevation and prominence to numeric, handling any parsing issues
df['elevation (m)'] = pd.to_numeric(df['elevation (m)'], errors='coerce')
df['prominence (m)'] = pd.to_numeric(df['prominence (m)'], errors='coerce')

# Filter mountains in Democratic Republic of the Congo with elevation >= 3000
filtered_df = df[
    (df['country'].str.contains('Democratic Republic of the Congo', case=False)) &
    (df['elevation (m)'] >= 3000)
]

# Calculate average prominence
if not filtered_df.empty:
    avg_prominence = filtered_df['prominence (m)'].mean()
else:
    avg_prominence = 0

print(f"Final Answer: {avg_prominence:.1f}")