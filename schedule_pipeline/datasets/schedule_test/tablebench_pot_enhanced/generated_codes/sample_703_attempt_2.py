import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'built' column to integer
df['built'] = pd.to_numeric(df['built'])
# Find the locomotive built in the earliest year
earliest_locomotive = df.loc[df['built'].idxmin(), 'locomotive']
# Find the build year of the earliest and latest locomotives
earliest_year = df['built'].min()
latest_year = df['built'].max()
# Calculate the difference
difference = latest_year - earliest_year
print(f"Final Answer: {earliest_locomotive}, {difference}")