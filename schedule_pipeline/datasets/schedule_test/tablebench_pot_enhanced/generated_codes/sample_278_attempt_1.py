import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Death toll' to numeric by removing commas
df['Death toll'] = df['Death toll'].str.replace(',', '').astype(int)
# Filter rows where death toll >= 1000 and depth < 30
filtered_df = df[(df['Death toll'] >= 1000) & (df['Depth (km)'] < 30)]
# Calculate average magnitude
average_magnitude = filtered_df['Magnitude'].mean()
print(f"Final Answer: {average_magnitude:.1f}")