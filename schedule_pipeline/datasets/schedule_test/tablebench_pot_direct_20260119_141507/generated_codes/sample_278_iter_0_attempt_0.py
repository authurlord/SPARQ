import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'Death toll' to numeric and filter by death toll >= 1000 and depth < 30
df_filtered = df[(df['Death toll'].str.replace(',', '').astype(float) >= 1000) & (df['Depth (km)'].astype(float) < 30)]

# Calculate the average magnitude of filtered earthquakes
average_magnitude = df_filtered['Magnitude'].mean()

print(f"Final Answer: {average_magnitude:.1f}")