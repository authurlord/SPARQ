import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Death toll' to numeric (removing commas) and 'Depth (km)' to float
df['Death toll'] = df['Death toll'].str.replace(',', '').astype(int)
df['Depth (km)'] = df['Depth (km)'].astype(float)

# Filter rows where death toll >= 1000 and depth < 30
filtered_df = df[(df['Death toll'] >= 1000) & (df['Depth (km)'] < 30)]

# Calculate average magnitude
average_magnitude = filtered_df['Magnitude'].astype(float).mean()

print(f"Final Answer: {average_magnitude:.1f}")