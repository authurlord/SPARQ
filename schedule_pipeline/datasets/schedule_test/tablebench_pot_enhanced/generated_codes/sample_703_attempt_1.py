import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'built' column to integer for numerical comparison
df['built'] = pd.to_numeric(df['built'])

# Find the locomotive built in the earliest year
earliest_year = df['built'].min()
earliest_locomotive = df[df['built'] == earliest_year]['locomotive'].values[0]

# Find the locomotive built in the latest year
latest_year = df['built'].max()
latest_locomotive = df[df['built'] == latest_year]['locomotive'].values[0]

# Calculate the difference
year_difference = latest_year - earliest_year

print(f"Final Answer: {earliest_locomotive}, {year_difference}")