import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'built' column to integer
df['built'] = pd.to_numeric(df['built'])
# Find the earliest and latest build years
earliest_year = df['built'].min()
latest_year = df['built'].max()
# Find the locomotive built in the earliest year
earliest_locomotive = df[df['built'] == earliest_year]['locomotive'].iloc[0]
# Calculate the difference
difference = latest_year - earliest_year
print(f"Final Answer: {earliest_locomotive}, {difference}")