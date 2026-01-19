import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'built' column to integer
df['built'] = pd.to_numeric(df['built'])
# Find the locomotive built in the earliest year
earliest = df.loc[df['built'].idxmin()]
# Find the locomotive built in the latest year
latest = df.loc[df['built'].idxmax()]
# Calculate the difference in build years
year_diff = latest['built'] - earliest['built']
print(f"Final Answer: {earliest['locomotive']}, {year_diff}")