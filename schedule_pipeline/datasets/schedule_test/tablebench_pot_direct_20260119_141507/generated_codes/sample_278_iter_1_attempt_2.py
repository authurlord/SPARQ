import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'Death toll' to numeric and filter for death toll >= 1000
df['Death toll'] = pd.to_numeric(df['Death toll'], errors='coerce')
filtered_df = df[(df['Death toll'] >= 1000) & (df['Depth (km)'].astype(float) < 30)]

# Convert 'Magnitude' to float and compute average
if not filtered_df.empty:
    avg_magnitude = filtered_df['Magnitude'].astype(float).mean()
else:
    avg_magnitude = None

print(f"Final Answer: {avg_magnitude:.2f}")