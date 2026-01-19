import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'Death toll' to numeric by removing commas and converting to int
df['Death toll'] = df['Death toll'].str.replace(',', '').astype(float)

# Convert 'Magnitude' to float
df['Magnitude'] = df['Magnitude'].astype(float)

# Filter rows where death toll >= 1000 and depth < 30
filtered_df = df[(df['Death toll'] >= 1000) & (df['Depth (km)'].astype(float) < 30)]

# Calculate average magnitude of filtered earthquakes
if not filtered_df.empty:
    avg_magnitude = filtered_df['Magnitude'].mean()
else:
    avg_magnitude = 0

print(f"Final Answer: {avg_magnitude:.2f}")