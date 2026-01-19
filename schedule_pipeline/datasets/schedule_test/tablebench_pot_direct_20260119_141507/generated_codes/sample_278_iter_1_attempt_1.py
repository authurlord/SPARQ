import pandas as pd

df = pd.read_csv('table.csv')

# Convert necessary columns to numeric, handling errors by converting to float
df['Death toll'] = pd.to_numeric(df['Death toll'], errors='coerce')
df['Magnitude'] = pd.to_numeric(df['Magnitude'], errors='coerce')
df['Depth (km)'] = pd.to_numeric(df['Depth (km)'], errors='coerce')

# Filter rows: death toll >= 1000 and depth < 30
filtered_df = df[(df['Death toll'] >= 1000) & (df['Depth (km)'] < 30)]

# Calculate average magnitude of filtered rows
if not filtered_df.empty:
    avg_magnitude = filtered_df['Magnitude'].mean()
else:
    avg_magnitude = None

print(f"Final Answer: {avg_magnitude:.2f}")