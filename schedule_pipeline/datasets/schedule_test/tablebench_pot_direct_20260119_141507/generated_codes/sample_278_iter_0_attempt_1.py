import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'Death toll' to numeric and filter based on conditions
df['Death toll'] = pd.to_numeric(df['Death toll'], errors='coerce')
filtered_df = df[(df['Death toll'] >= 1000) & (df['Depth (km)'].astype(float) < 30)]

# Calculate average magnitude of filtered rows
if not filtered_df.empty:
    avg_magnitude = filtered_df['Magnitude'].mean()
else:
    avg_magnitude = None

print(f"Final Answer: {avg_magnitude:.1f}")